"""Telemetría por consulta: latencia por nodo, tokens y costo (reto E sin infraestructura extra).

La medición vive en un ContextVar por request; un callback de LangChain atribuye los tokens al nodo activo.
Si hay claves de Langfuse en el .env, además se envían las trazas a Langfuse.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

from .schemas import Metrics, NodeMetric

current_telemetry: ContextVar["Telemetry | None"] = ContextVar("current_telemetry", default=None)
current_node: ContextVar[str] = ContextVar("current_node", default="?")


class Telemetry:
    def __init__(self, prices: dict[str, tuple[float, float]], free: bool = False):
        self.prices = prices                     # modelo -> (USD/M entrada, USD/M salida)
        self.free = free                         # modelos locales (Ollama): costo cero
        self.t0 = time.perf_counter()
        self.node_ms: dict[str, float] = defaultdict(float)
        self.node_tokens: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])
        self.llm_calls = 0

    def record_node(self, node: str, ms: float) -> None:
        self.node_ms[node] += ms

    def record_llm(self, node: str, model: str, tokens_in: int, tokens_out: int) -> None:
        self.llm_calls += 1
        p_in, p_out = (0.0, 0.0) if self.free else self.prices.get(model, max(self.prices.values(), default=(0, 0)))
        t = self.node_tokens[node]
        t[0] += tokens_in
        t[1] += tokens_out
        t[2] += tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out

    def metrics(self) -> Metrics:
        nodes = [NodeMetric(node=n, ms=round(ms, 1), tokens_in=int(self.node_tokens[n][0]),
                            tokens_out=int(self.node_tokens[n][1]), cost_usd=round(self.node_tokens[n][2], 6))
                 for n, ms in self.node_ms.items()]
        return Metrics(latency_ms=round((time.perf_counter() - self.t0) * 1000, 1),
                       tokens_in=sum(n.tokens_in for n in nodes), tokens_out=sum(n.tokens_out for n in nodes),
                       cost_usd=round(sum(n.cost_usd for n in nodes), 6), llm_calls=self.llm_calls, nodes=nodes)


def timed(name: str) -> Callable:
    """Decorator: mide el nodo y fija el nodo activo para atribuir tokens."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = current_node.set(name)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                tel = current_telemetry.get()
                if tel:
                    tel.record_node(name, (time.perf_counter() - t0) * 1000)
                current_node.reset(token)
        return wrapper
    return deco


def langchain_callback():
    """Adaptador de callbacks de LangChain -> Telemetry (import perezoso)."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _UsageCallback(BaseCallbackHandler):
        def __init__(self) -> None:
            self._models: dict[Any, str] = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):  # noqa: ANN001
            self._models[run_id] = (metadata or {}).get("ls_model_name", "")

        def on_llm_end(self, response, *, run_id, **kwargs):  # noqa: ANN001
            tel = current_telemetry.get()
            if not tel:
                return
            usage = {}
            try:
                usage = response.generations[0][0].message.usage_metadata or {}
            except (AttributeError, IndexError):
                pass
            tel.record_llm(current_node.get(), self._models.pop(run_id, ""),
                           int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))

    return _UsageCallback()


def langfuse_callback(public_key: str | None, secret_key: str | None, host: str | None):
    if not (public_key and secret_key):
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None
    return CallbackHandler()          # lee LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST del entorno

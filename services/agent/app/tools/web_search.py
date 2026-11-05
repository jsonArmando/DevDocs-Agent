"""Herramienta externa tipada (tool calling) con timeout, resultado vacío y proveedores intercambiables.

Patrones: Strategy (DuckDuckGo gratis / Tavily con key) + Adapter (formato común) + Timeout + Fallback.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

log = logging.getLogger("web_search")


class WebSearchArgs(BaseModel):
    """Esquema tipado que ve el LLM al hacer tool calling."""

    query: str = Field(description="Consulta de búsqueda precisa, idealmente en inglés", min_length=2, max_length=300)
    max_results: int = Field(default=5, ge=1, le=8, description="Número de resultados")


@dataclass
class SearchOutcome:
    ok: bool
    results: list[dict] = field(default_factory=list)   # {"title","url","snippet"}
    error: str | None = None                            # "timeout" | "empty" | "provider_error: ..."


Provider = Callable[[str, int], list[dict]]


def duckduckgo_provider(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS                                # gratis, sin API key

    with DDGS() as d:
        raw = d.text(query, max_results=max_results)
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in raw or []]


def tavily_provider_factory(api_key: str) -> Provider:
    def provider(query: str, max_results: int) -> list[dict]:
        import httpx

        r = httpx.post("https://api.tavily.com/search", timeout=10,
                       json={"api_key": api_key, "query": query, "max_results": max_results})
        r.raise_for_status()
        return [{"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
                for x in r.json().get("results", [])]
    return provider


_POOL = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="search")


def run_search(provider: Provider, args: WebSearchArgs, timeout_s: float) -> SearchOutcome:
    """Ejecuta la búsqueda con timeout duro y clasifica los errores para que el agente reaccione."""
    future = _POOL.submit(provider, args.query, args.max_results)
    try:
        results = future.result(timeout=timeout_s)
    except cf.TimeoutError:
        future.cancel()
        log.warning("web_search timeout (%ss): %s", timeout_s, args.query)
        return SearchOutcome(ok=False, error="timeout")
    except Exception as e:                                # red caída, rate limit, cambio de API
        log.warning("web_search error: %s", e)
        return SearchOutcome(ok=False, error=f"provider_error: {type(e).__name__}")
    results = [r for r in results if r.get("url")]
    if not results:
        return SearchOutcome(ok=False, error="empty")
    return SearchOutcome(ok=True, results=results)


def build_provider(name: str, tavily_key: str | None) -> Provider:
    if name == "tavily" and tavily_key:
        return tavily_provider_factory(tavily_key)
    return duckduckgo_provider

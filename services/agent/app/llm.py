"""Puerto LLM del agente y su adaptador LangChain (Ports & Adapters + Strategy de proveedor).

Ahorro de costos por diseño:
* Modelo rápido/barato para decisiones (router, grader en lote, rewriter, planner de búsqueda).
* Modelo grande solo para redactar respuestas (generator, síntesis).
* Con LLM_PROVIDER=ollama todo corre local a costo de API cero.
"""
from __future__ import annotations

from docs_core.config import Settings
from docs_core.models import RetrievedChunk

from . import prompts
from .citations import format_fragments
from .ports import AgentLLM  # noqa: F401  (re-export)
from .schemas import GradeResult, RewriteResult, RouteDecision
from .tools.web_search import WebSearchArgs


def make_chat_model(model: str, s: Settings):
    from langchain.chat_models import init_chat_model

    if s.llm_provider == "ollama":
        return init_chat_model(model, model_provider="ollama", base_url=s.ollama_base_url, temperature=s.llm_temperature)
    return init_chat_model(model, model_provider=s.llm_provider, temperature=s.llm_temperature,
                           timeout=s.llm_timeout_s, max_retries=2)


class LangChainLLM:
    def __init__(self, s: Settings):
        self.s = s
        self.fast = make_chat_model(s.llm_model_fast, s)
        self.smart = make_chat_model(s.llm_model_smart, s)
        self._router = self.fast.with_structured_output(RouteDecision)
        self._grader = self.fast.with_structured_output(GradeResult)
        self._rewriter = self.fast.with_structured_output(RewriteResult)
        self._planner = self._bind_search_tool()

    def _bind_search_tool(self):
        from langchain_core.tools import StructuredTool

        tool = StructuredTool.from_function(
            func=lambda query, max_results=5: None, name="web_search", args_schema=WebSearchArgs,
            description="Search the public web for up-to-date developer/technical information.")
        try:
            return self.fast.bind_tools([tool], tool_choice="web_search")      # forzar la herramienta
        except (NotImplementedError, TypeError, ValueError):
            return self.fast.bind_tools([tool])                                 # proveedores sin tool_choice

    # --------------------------------------------------------------- decisiones (modelo rápido)
    def route(self, history: str, question: str, catalog: list[str]) -> RouteDecision:
        system = prompts.SUPERVISOR.format(catalog="; ".join(catalog[:60]) or "(vacía)")
        human = f"Conversation so far:\n{history or '(none)'}\n\nLast user message: {question}"
        return self._router.invoke([("system", system), ("human", human)])

    def grade(self, question: str, chunks: list[RetrievedChunk]) -> GradeResult:
        frags = format_fragments(chunks, max_chars=900)                         # recorte: el grader no necesita todo
        return self._grader.invoke([("human", prompts.GRADER.format(question=question, fragments=frags))])

    def rewrite(self, question: str, previous: list[str]) -> str:
        r = self._rewriter.invoke([("human", prompts.REWRITER.format(question=question, previous=previous or "—"))])
        return r.query

    def plan_search(self, question: str) -> WebSearchArgs | None:
        msg = self._planner.invoke([("human", prompts.RESEARCH_PLANNER.format(question=question))])
        for call in getattr(msg, "tool_calls", []) or []:
            if call.get("name") == "web_search":
                return WebSearchArgs(**call["args"])
        return None

    # --------------------------------------------------------------- redacción (modelo grande)
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        msg = self.smart.invoke([("human", prompts.GENERATOR.format(question=question, fragments=format_fragments(chunks)))])
        return _text(msg)

    def synthesize(self, question: str, results: list[dict]) -> str:
        res = "\n\n".join(f"[{i}] {r['title']} ({r['url']})\n{r['snippet']}" for i, r in enumerate(results, start=1))
        msg = self.smart.invoke([("human", prompts.RESEARCH_SYNTHESIS.format(question=question, results=res))])
        return _text(msg)


def _text(msg) -> str:  # noqa: ANN001
    content = msg.content
    if isinstance(content, list):                    # algunos proveedores devuelven bloques
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)

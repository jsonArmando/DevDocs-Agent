"""Subagente de investigación: tool calling tipado (web_search) + síntesis con enlaces."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..citations import web_sources
from ..ports import AgentLLM
from ..telemetry import timed
from ..tools.web_search import Provider, WebSearchArgs, run_search

EXTERNAL_PREFIX = "🌐 Fuente externa (búsqueda web):"
_ERRORS = {"timeout": "la búsqueda externa superó el tiempo límite",
           "empty": "la búsqueda externa no devolvió resultados"}


class ResearchState(TypedDict, total=False):
    question: str
    search_query: str
    results: list[dict]
    search_error: str | None
    answer: str
    sources: list[dict]
    contexts: list[str]
    steps: Annotated[list[str], operator.add]


class ResearchAgent:
    def __init__(self, llm: AgentLLM, provider: Provider, timeout_s: float = 8.0, max_results: int = 5):
        self.llm, self.provider, self.timeout_s, self.max_results = llm, provider, timeout_s, max_results

    @timed("research_agent.search")
    def search(self, s: ResearchState) -> dict:
        question = s["question"]
        try:
            args = self.llm.plan_search(question)                    # el LLM llena el esquema tipado
        except Exception:                                            # planner falla: no bloquear la búsqueda
            args = None
        args = args or WebSearchArgs(query=question[:300], max_results=self.max_results)
        outcome = run_search(self.provider, args, self.timeout_s)
        if not outcome.ok and outcome.error == "empty" and args.query != question:
            args = WebSearchArgs(query=question[:300], max_results=self.max_results)   # fallback: consulta literal
            outcome = run_search(self.provider, args, self.timeout_s)
        return {"search_query": args.query, "results": outcome.results, "search_error": outcome.error,
                "steps": ["search"]}

    @timed("research_agent.synthesize")
    def synthesize(self, s: ResearchState) -> dict:
        results = s.get("results") or []
        if not results:
            reason = _ERRORS.get(s.get("search_error") or "", f"error del proveedor ({s.get('search_error')})")
            return {"answer": f"{EXTERNAL_PREFIX} no pude obtener información: {reason}. "
                              "Intenta reformular la pregunta o vuelve a intentarlo en unos minutos.",
                    "sources": [], "steps": ["synthesize"]}
        text = self.llm.synthesize(s["question"], results)
        if not text.strip().startswith("🌐"):
            text = f"{EXTERNAL_PREFIX} {text}"
        answer, sources = web_sources(text, results)
        return {"answer": answer, "sources": [x.model_dump() for x in sources],
                "contexts": [f"{x['title']}: {x['snippet']}" for x in results], "steps": ["synthesize"]}

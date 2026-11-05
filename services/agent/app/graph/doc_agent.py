"""Subagente de documentación: RAG correctivo (retriever -> grader -> [rewriter -> retriever]* -> generator).

Nodos puros sobre un dict de estado: se prueban sin LangGraph (tests/test_doc_agent.py).
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from docs_core.models import RetrievedChunk
from docs_core.ports import RetrieverPort

from ..citations import doc_sources, is_no_context
from ..ports import AgentLLM
from ..telemetry import timed


class DocState(TypedDict, total=False):
    question: str                       # pregunta autónoma (la reescribe el supervisor)
    query: str                          # consulta de búsqueda vigente (puede reescribirse)
    queries: list[str]                  # consultas reescritas en este turno
    doc_ids: list[str] | None
    rewrites: int
    candidates: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    answer: str
    sources: list[dict]
    has_context: bool
    contexts: list[str]
    steps: Annotated[list[str], operator.add]


class DocAgent:
    def __init__(self, llm: AgentLLM, retriever: RetrieverPort, max_rewrites: int = 2, top_n: int = 5):
        self.llm, self.retriever, self.max_rewrites, self.top_n = llm, retriever, max_rewrites, top_n

    @timed("doc_agent.retrieve")
    def retrieve(self, s: DocState) -> dict:
        chunks = self.retriever.retrieve(s["query"], k=self.top_n, doc_ids=s.get("doc_ids"))
        return {"candidates": chunks, "steps": ["retrieve"]}

    @timed("doc_agent.grade")
    def grade(self, s: DocState) -> dict:
        cands = s.get("candidates") or []
        if not cands:
            return {"relevant": [], "steps": ["grade"]}
        result = self.llm.grade(s["question"], cands)                # una sola llamada para todos los chunks
        keep = {g.ref for g in result.grades if g.relevant}
        return {"relevant": [c for i, c in enumerate(cands, start=1) if i in keep], "steps": ["grade"]}

    def after_grade(self, s: DocState) -> Literal["generate", "rewrite", "no_context"]:
        """Arista condicional: generar, reintentar (ciclo controlado) o declarar falta de contexto."""
        if s.get("relevant"):
            return "generate"
        if s.get("rewrites", 0) < self.max_rewrites:
            return "rewrite"
        return "no_context"

    @timed("doc_agent.rewrite")
    def rewrite(self, s: DocState) -> dict:
        previous = [s["question"], *s.get("queries", [])]
        q = self.llm.rewrite(s["question"], previous)
        return {"query": q, "queries": [*s.get("queries", []), q], "rewrites": s.get("rewrites", 0) + 1,
                "steps": ["rewrite"]}

    @timed("doc_agent.generate")
    def generate(self, s: DocState) -> dict:
        relevant = s["relevant"]
        text = self.llm.generate(s["question"], relevant)
        if is_no_context(text):                                      # relevante pero insuficiente
            return {"has_context": False, "answer": "", "sources": [], "steps": ["generate"]}
        answer, sources = doc_sources(text, relevant)
        return {"has_context": True, "answer": answer, "sources": [x.model_dump() for x in sources],
                "contexts": [c.text for c in relevant], "steps": ["generate"]}

    @timed("doc_agent.no_context")
    def no_context(self, s: DocState) -> dict:
        return {"has_context": False, "answer": "", "sources": [], "steps": ["no_context"]}

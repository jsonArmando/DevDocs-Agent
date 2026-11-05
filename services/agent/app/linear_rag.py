"""Baseline obligatorio: RAG lineal (recuperar -> generar), sin router, grader, rewriter ni investigación.

Usa el mismo LLM y el mismo prompt de generación que el grafo para que la comparación sea justa;
recupera en modo denso sin rerank (la configuración "clásica").
"""
from __future__ import annotations

from docs_core.ports import RetrieverPort

from .citations import doc_sources, is_no_context
from .ports import AgentLLM
from .telemetry import timed


class LinearRAG:
    def __init__(self, llm: AgentLLM, retriever: RetrieverPort, top_n: int = 5):
        self.llm, self.retriever, self.top_n = llm, retriever, top_n

    def run(self, question: str, doc_ids: list[str] | None = None) -> dict:
        chunks = timed("linear.retrieve")(lambda: self.retriever.retrieve(
            question, k=self.top_n, doc_ids=doc_ids, mode="dense", rerank=False))()
        text = timed("linear.generate")(lambda: self.llm.generate(question, chunks))()
        if is_no_context(text):
            return {"answer": "No encontré esa información en la documentación indexada.", "sources": [],
                    "contexts": [c.text for c in chunks],
                    "steps": ["retrieve", "generate"]}
        answer, sources = doc_sources(text, chunks)
        return {"answer": answer, "sources": [s.model_dump() for s in sources], "contexts": [c.text for c in chunks],
                "steps": ["retrieve", "generate"]}

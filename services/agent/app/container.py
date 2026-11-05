"""Composition Root: el único lugar que conoce las implementaciones concretas."""
from __future__ import annotations

import sqlite3
import time
from functools import cached_property
from pathlib import Path

from docs_core.config import Settings, get_settings
from docs_core.retrieval import Retriever

from .graph.builder import build_graph
from .graph.doc_agent import DocAgent
from .graph.research_agent import ResearchAgent
from .graph.supervisor import Supervisor
from .linear_rag import LinearRAG
from .llm import LangChainLLM
from .telemetry import Telemetry, langchain_callback, langfuse_callback
from .tools.web_search import build_provider


class Container:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._catalog: tuple[float, list[str]] = (0.0, [])

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(self.s)

    @cached_property
    def llm(self) -> LangChainLLM:
        return LangChainLLM(self.s)

    def catalog(self) -> list[str]:
        """Títulos indexados para el supervisor, cacheados 60 s (evita escanear Qdrant en cada consulta)."""
        ts, titles = self._catalog
        if time.time() - ts > 60:
            try:
                titles = sorted({d.title for d in self.retriever.store.list_documents()})
            except Exception:
                titles = titles or []
            self._catalog = (time.time(), titles)
        return titles

    @cached_property
    def checkpointer(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        Path(self.s.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(self.s.checkpoint_db, check_same_thread=False))

    @cached_property
    def graph(self):
        supervisor = Supervisor(self.llm, self.catalog)
        doc = DocAgent(self.llm, self.retriever, max_rewrites=self.s.max_rewrites, top_n=self.s.top_n)
        research = ResearchAgent(self.llm, build_provider(self.s.search_provider, self.s.tavily_api_key),
                                 timeout_s=self.s.search_timeout_s, max_results=self.s.search_max_results)
        return build_graph(supervisor, doc, research, checkpointer=self.checkpointer)

    @cached_property
    def linear(self) -> LinearRAG:
        return LinearRAG(self.llm, self.retriever, top_n=self.s.top_n)

    def new_telemetry(self) -> Telemetry:
        s = self.s
        prices = {s.llm_model_fast: (s.price_fast_in, s.price_fast_out),
                  s.llm_model_smart: (s.price_smart_in, s.price_smart_out)}
        return Telemetry(prices, free=s.llm_provider == "ollama")

    def callbacks(self) -> list:
        cbs = [langchain_callback()]
        lf = langfuse_callback(self.s.langfuse_public_key, self.s.langfuse_secret_key, self.s.langfuse_host)
        return cbs + ([lf] if lf else [])

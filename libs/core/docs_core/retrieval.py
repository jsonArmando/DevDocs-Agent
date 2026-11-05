"""Retriever: búsqueda densa o híbrida + reranking opcional con cross-encoder local."""
from __future__ import annotations

from .config import Settings, get_settings
from .embeddings import Embedder, Reranker
from .models import RetrievedChunk
from .ports import RetrieverPort  # noqa: F401  (re-export)
from .vectorstore import VectorStore


class Retriever:
    def __init__(self, settings: Settings | None = None, store: VectorStore | None = None,
                 embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.s = settings or get_settings()
        self.store = store or VectorStore(self.s)
        self.embedder = embedder or Embedder(self.s)
        self.reranker = reranker or Reranker(self.s)

    def retrieve(self, query: str, k: int | None = None, doc_ids: list[str] | None = None,
                 mode: str | None = None, rerank: bool | None = None) -> list[RetrievedChunk]:
        k = k or self.s.top_n
        mode = mode or self.s.retrieval_mode
        rerank = self.s.rerank_enabled if rerank is None else rerank
        dense = self.embedder.embed_query(query)
        sparse = self.embedder.sparse_query(query) if mode == "hybrid" else None
        candidates = self.store.search(dense, sparse, limit=max(k, self.s.top_k) if rerank else k, doc_ids=doc_ids)
        if not rerank or not candidates:
            return candidates[:k]
        scores = self.reranker.scores(query, [c.text for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [c.model_copy(update={"score": s}) for c, s in ranked[:k]]

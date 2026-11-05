"""Modelos locales (fastembed/ONNX en CPU): embeddings densos, BM25 disperso y reranker.

Costo de API: cero. Los modelos se descargan una vez al volumen `models_cache`.
Se cargan de forma perezosa y una sola vez por proceso (Singleton por lru_cache).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .config import Settings, get_settings


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class Embedder:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    @property
    def _dense(self):
        return _dense_model(self.s.embedding_model, self.s.models_cache_dir)

    @property
    def _sparse(self):
        return _sparse_model(self.s.sparse_model, self.s.models_cache_dir)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._dense.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._dense.query_embed(text))).tolist()

    def sparse_passages(self, texts: list[str]) -> list[SparseVec]:
        return [SparseVec(e.indices.tolist(), e.values.tolist()) for e in self._sparse.passage_embed(texts)]

    def sparse_query(self, text: str) -> SparseVec:
        e = next(iter(self._sparse.query_embed(text)))
        return SparseVec(e.indices.tolist(), e.values.tolist())


class Reranker:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    def scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = _rerank_model(self.s.rerank_model, self.s.models_cache_dir)
        return [float(x) for x in model.rerank(query, documents)]


@lru_cache(maxsize=2)
def _dense_model(name: str, cache: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name, cache_dir=cache)


@lru_cache(maxsize=2)
def _sparse_model(name: str, cache: str):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=name, cache_dir=cache)


@lru_cache(maxsize=2)
def _rerank_model(name: str, cache: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=name, cache_dir=cache)

"""Repositorio sobre Qdrant: colección con vector denso (HNSW) + vector disperso BM25 (IDF), filtros por metadatos.

Parámetros HNSW (ver docs/adr/0002-vector-db.md):
  m=16            conexiones por nodo: buen equilibrio recall/memoria para < 1M vectores
  ef_construct=128 calidad del grafo al indexar (indexación algo más lenta, mejor recall)
  hnsw_ef=96      exploración en consulta: recall > 0,98 en colecciones pequeñas con latencia de ms
  payload indexes keyword en doc_id y source: filtrar por documento sin escanear
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .config import Settings, get_settings
from .embeddings import SparseVec
from .models import Chunk, DocumentInfo, RetrievedChunk

DENSE, SPARSE = "dense", "bm25"


class VectorStore:
    def __init__(self, settings: Settings | None = None, client: QdrantClient | None = None):
        self.s = settings or get_settings()
        self.client = client or QdrantClient(url=self.s.qdrant_url, api_key=self.s.qdrant_api_key, timeout=30)
        self.col = self.s.collection

    # ------------------------------------------------------------------ esquema
    def ensure_collection(self, retries: int = 30, wait_s: float = 2.0) -> None:
        for attempt in range(retries):                       # Qdrant puede tardar en aceptar conexiones al arrancar
            try:
                exists = self.client.collection_exists(self.col)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(wait_s)
        if exists:
            return
        self.client.create_collection(
            collection_name=self.col,
            vectors_config={DENSE: qm.VectorParams(
                size=self.s.embedding_dim, distance=qm.Distance.COSINE,
                hnsw_config=qm.HnswConfigDiff(m=self.s.hnsw_m, ef_construct=self.s.hnsw_ef_construct),
            )},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        for field in ("doc_id", "source", "content_hash"):
            self.client.create_payload_index(self.col, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD)

    # ------------------------------------------------------------------ escritura
    def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse: list[SparseVec], batch: int = 128) -> None:
        points = [
            qm.PointStruct(
                id=c.id,
                vector={DENSE: d, SPARSE: qm.SparseVector(indices=s.indices, values=s.values)},
                payload={**c.metadata.model_dump(mode="json"), "text": c.text},
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        for i in range(0, len(points), batch):
            self.client.upsert(self.col, points=points[i:i + batch], wait=True)

    def delete_by(self, key: str, value: str) -> None:
        self.client.delete(self.col, points_selector=qm.FilterSelector(filter=_eq(key, value)), wait=True)

    # ------------------------------------------------------------------ lectura
    def hash_exists(self, h: str) -> bool:
        return self.client.count(self.col, count_filter=_eq("content_hash", h), exact=True).count > 0

    def hash_for_source(self, source: str) -> str | None:
        pts, _ = self.client.scroll(self.col, scroll_filter=_eq("source", source), limit=1, with_payload=["content_hash"])
        return pts[0].payload["content_hash"] if pts else None

    def search(self, dense: list[float], sparse: SparseVec | None, limit: int,
               doc_ids: list[str] | None = None) -> list[RetrievedChunk]:
        flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))]) if doc_ids else None
        params = qm.SearchParams(hnsw_ef=self.s.hnsw_ef_search)
        if sparse is None:                                   # modo denso puro (baseline)
            res = self.client.query_points(self.col, query=dense, using=DENSE, limit=limit,
                                           query_filter=flt, search_params=params, with_payload=True)
        else:                                                # híbrido: denso + BM25 fusionados con RRF
            res = self.client.query_points(
                self.col,
                prefetch=[
                    qm.Prefetch(query=dense, using=DENSE, limit=limit * 2, filter=flt, params=params),
                    qm.Prefetch(query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                                using=SPARSE, limit=limit * 2, filter=flt),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit, with_payload=True,
            )
        return [_to_retrieved(p) for p in res.points]

    def list_documents(self) -> list[DocumentInfo]:
        docs: dict[str, dict] = defaultdict(lambda: {"chunks": 0})
        offset = None
        while True:
            pts, offset = self.client.scroll(self.col, limit=512, offset=offset,
                                             with_payload=["doc_id", "source", "title", "file_type", "ingested_at"])
            for p in pts:
                d = docs[p.payload["doc_id"]]
                d.update({k: p.payload[k] for k in ("doc_id", "source", "title", "file_type", "ingested_at")})
                d["chunks"] += 1
            if offset is None:
                break
        return sorted((DocumentInfo(**{**d, "ingested_at": datetime.fromisoformat(d["ingested_at"])}) for d in docs.values()),
                      key=lambda x: x.source)


def _eq(key: str, value: str) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key=key, match=qm.MatchValue(value=value))])


def _to_retrieved(p) -> RetrievedChunk:
    pl = p.payload or {}
    return RetrievedChunk(id=str(p.id), text=pl.get("text", ""), score=float(p.score or 0.0),
                          doc_id=pl.get("doc_id", ""), source=pl.get("source", ""), title=pl.get("title", ""),
                          section=pl.get("section", ""), page=pl.get("page"))

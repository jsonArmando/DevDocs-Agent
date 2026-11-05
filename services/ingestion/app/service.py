"""Caso de uso de ingesta: cargar -> trocear -> vectorizar -> indexar, de forma incremental e idempotente."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from docs_core.chunking import chunk_sections, embed_text
from docs_core.config import Settings, get_settings
from docs_core.embeddings import Embedder
from docs_core.ids import chunk_id, content_hash, decide_ingest_action, doc_id_from_hash
from docs_core.loaders import UnsupportedFileError, load_sections
from docs_core.models import Chunk, ChunkMetadata, IngestResult
from docs_core.vectorstore import VectorStore

log = logging.getLogger("ingestion")


class IngestionService:
    def __init__(self, settings: Settings | None = None, store: VectorStore | None = None,
                 embedder: Embedder | None = None):
        self.s = settings or get_settings()
        self.store = store or VectorStore(self.s)
        self.embedder = embedder or Embedder(self.s)

    def ingest(self, filename: str, data: bytes) -> IngestResult:
        h = content_hash(data)
        doc_id = doc_id_from_hash(h)
        try:
            action = decide_ingest_action(self.store.hash_for_source(filename), self.store.hash_exists(h), h)
            if action == "duplicate":
                return IngestResult(source=filename, doc_id=doc_id, status="duplicate",
                                    detail="Contenido idéntico ya indexado: no se recalculan embeddings")
            file_type, title, sections = load_sections(filename, data)
        except UnsupportedFileError as e:
            return IngestResult(source=filename, doc_id=doc_id, status="rejected", detail=str(e))
        except Exception as e:                                         # PDF corrupto, encoding raro…
            log.exception("Error leyendo %s", filename)
            return IngestResult(source=filename, doc_id=doc_id, status="rejected", detail=f"No se pudo leer: {e}")

        pieces = chunk_sections(sections, self.s.chunk_max_chars, self.s.chunk_overlap_chars, self.s.chunk_min_chars)
        if not pieces:
            return IngestResult(source=filename, doc_id=doc_id, status="rejected",
                                detail="Sin texto extraíble (¿PDF escaneado?)")
        now = datetime.now(UTC)
        chunks = [
            Chunk(
                id=chunk_id(h, i), text=text, embed_text=embed_text(title, sec.section, text),
                metadata=ChunkMetadata(doc_id=doc_id, source=filename, title=title, section=sec.section,
                                       page=sec.page, chunk_index=i, content_hash=h, ingested_at=now,
                                       file_type=file_type),
            )
            for i, (sec, text) in enumerate(pieces)
        ]
        texts = [c.embed_text for c in chunks]
        dense, sparse = self.embedder.embed_passages(texts), self.embedder.sparse_passages(texts)
        if action == "updated":
            self.store.delete_by("source", filename)                    # versión anterior del mismo archivo
        self.store.upsert(chunks, dense, sparse)
        log.info("Ingestado %s (%s): %d chunks", filename, action, len(chunks))
        return IngestResult(source=filename, doc_id=doc_id, status=action, chunks=len(chunks))

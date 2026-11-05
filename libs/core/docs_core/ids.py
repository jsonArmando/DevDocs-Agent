"""Identidad determinista de documentos y chunks: base de la ingesta incremental e idempotente."""
from __future__ import annotations

import hashlib
import uuid

_NAMESPACE = uuid.UUID("5b6f3a1e-9c2d-4f7a-8e11-3d2c7b9a0f42")


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_id_from_hash(h: str) -> str:
    return h[:16]


def chunk_id(content_hash_: str, chunk_index: int) -> str:
    """UUID estable: re-ingestar el mismo contenido produce los mismos IDs (upsert idempotente)."""
    return str(uuid.uuid5(_NAMESPACE, f"{content_hash_}:{chunk_index}"))


def decide_ingest_action(existing_hash_for_source: str | None, hash_exists_anywhere: bool, new_hash: str) -> str:
    """Patrón Idempotent Receiver aplicado a la ingesta.

    - duplicate: el mismo contenido ya está indexado (aunque venga con otro nombre).
    - updated:   el mismo archivo (source) cambió: borrar sus vectores viejos y reindexar.
    - created:   documento nuevo.
    """
    if hash_exists_anywhere:
        return "duplicate"
    if existing_hash_for_source and existing_hash_for_source != new_hash:
        return "updated"
    return "created"

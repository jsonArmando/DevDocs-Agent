"""Puertos livianos (sin dependencias de infraestructura) para inyectar en los agentes."""
from __future__ import annotations

from typing import Protocol

from .models import RetrievedChunk


class RetrieverPort(Protocol):
    """Puerto que usan los agentes; en tests se reemplaza por un fake."""

    def retrieve(self, query: str, k: int | None = None, doc_ids: list[str] | None = None,
                 mode: str | None = None, rerank: bool | None = None) -> list[RetrievedChunk]: ...

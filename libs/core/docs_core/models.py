"""Modelos de dominio compartidos entre servicios."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Section(BaseModel):
    """Unidad lógica antes del chunking: una sección de Markdown o una página de PDF."""

    text: str
    section: str                      # ruta de encabezados ("Tutorial > Query Parameters") o "página 3"
    page: int | None = None


class ChunkMetadata(BaseModel):
    doc_id: str                       # = hash de contenido (identidad del documento)
    source: str                       # nombre del archivo (clave lógica para detectar modificaciones)
    title: str
    section: str
    page: int | None = None
    chunk_index: int
    content_hash: str
    ingested_at: datetime
    file_type: Literal["pdf", "md", "txt"]


class Chunk(BaseModel):
    id: str
    text: str                         # texto que se muestra y se cita
    embed_text: str                   # texto que se vectoriza (con contexto de título y sección)
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    doc_id: str
    source: str
    title: str
    section: str
    page: int | None = None


class IngestResult(BaseModel):
    source: str
    doc_id: str
    status: Literal["created", "updated", "duplicate", "rejected"]
    chunks: int = 0
    detail: str | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    title: str
    file_type: str
    chunks: int
    ingested_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    doc_ids: list[str] | None = None
    mode: Literal["dense", "hybrid"] | None = None
    rerank: bool | None = None
    k: int = Field(default=5, ge=1, le=20)

"""Contratos tipados: salidas estructuradas del LLM y API pública."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["docs", "research", "direct"]
AgentName = Literal["documentation", "research", "direct", "linear_rag"]


# ---------------------------------------------------------------- salidas estructuradas del LLM
class RouteDecision(BaseModel):
    """Decisión del supervisor."""

    route: Route = Field(description="docs: la documentación indexada probablemente lo cubre; research: requiere "
                                     "información externa o reciente; direct: saludo o tema fuera de alcance")
    direct_kind: Literal["greeting", "out_of_scope"] | None = Field(
        default=None, description="Solo si route=direct")
    standalone_question: str = Field(description="La pregunta reescrita para entenderse sin el historial")
    reason: str = Field(description="Justificación breve de la decisión")


class ChunkGrade(BaseModel):
    ref: int = Field(description="Número del fragmento evaluado")
    relevant: bool = Field(description="True si el fragmento ayuda a responder la pregunta")


class GradeResult(BaseModel):
    """Calificación en lote: una sola llamada al LLM para todos los fragmentos (ahorro de tokens y latencia)."""

    grades: list[ChunkGrade]


class RewriteResult(BaseModel):
    query: str = Field(description="Consulta reformulada para búsqueda en documentación técnica en inglés")


# ---------------------------------------------------------------- API
class Source(BaseModel):
    ref: int
    kind: Literal["doc", "web"]
    title: str
    section: str | None = None
    page: int | None = None
    doc_id: str | None = None
    url: str | None = None
    snippet: str


class NodeMetric(BaseModel):
    node: str
    ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class Metrics(BaseModel):
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    llm_calls: int
    nodes: list[NodeMetric]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, description="Si se omite, se crea una conversación nueva")
    doc_ids: list[str] | None = Field(default=None, description="Filtrar la búsqueda a estos documentos")


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    agent: AgentName
    route: Route | None = None
    route_reason: str | None = None
    rewritten_queries: list[str] = []
    steps: list[str]
    sources: list[Source]
    contexts: list[str] = Field(default=[], description="Textos usados para responder (para evaluación de fidelidad)")
    metrics: Metrics | None = None

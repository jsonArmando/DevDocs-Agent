"""Configuración centralizada (12-factor): todo viene de variables de entorno / .env."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Strategy: el proveedor se elige por configuración) ---
    llm_provider: Literal["anthropic", "openai", "groq", "ollama"] = "anthropic"
    llm_model_fast: str = "claude-haiku-5-5"      # router, grader, rewriter, juez de evaluación
    llm_model_smart: str = "claude-sonnet-5-5"    # generator y síntesis de investigación
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    ollama_base_url: str = "http://ollama:11434"

    # --- Embeddings locales (costo cero, CPU, ONNX vía fastembed) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"          # 384 dims
    embedding_dim: int = 384
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    models_cache_dir: str = "/models"

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    collection: str = "devdocs"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_ef_search: int = 96

    # --- Chunking ---
    chunk_max_chars: int = 1400       # ~350 tokens: cabe holgado en los 512 tokens de bge-small
    chunk_overlap_chars: int = 200    # ~15 %
    chunk_min_chars: int = 120

    # --- Recuperación ---
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    rerank_enabled: bool = True
    top_k: int = 12                   # candidatos antes del rerank
    top_n: int = 5                    # chunks que llegan al grader
    max_rewrites: int = 2             # ciclo controlado rewriter -> retriever

    # --- Investigación externa ---
    search_provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    tavily_api_key: str | None = None
    search_timeout_s: float = 8.0
    search_max_results: int = 5

    # --- Servicios ---
    ingestion_url: str = "http://ingestion:8001"
    agent_url: str = "http://agent:8002"
    checkpoint_db: str = "/data/checkpoints.sqlite"
    max_upload_mb: int = 25

    # --- Observabilidad opcional ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    # Precios USD por millón de tokens (entrada, salida) para estimar costo. Ajusta a tu contrato.
    price_fast_in: float = Field(1.0)
    price_fast_out: float = Field(5.0)
    price_smart_in: float = Field(3.0)
    price_smart_out: float = Field(15.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

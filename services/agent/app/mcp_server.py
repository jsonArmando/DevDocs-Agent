"""Reto A: el sistema expuesto como servidor MCP (stdio) para Claude Desktop / Claude Code.

Herramientas:
  search_docs      búsqueda híbrida en la documentación indexada (sin LLM: costo cero)
  ingest_document  ingesta un archivo local (PDF/MD/TXT) vía el servicio de ingesta
  ask              pregunta al grafo agéntico completo (con memoria por thread_id)
"""
from __future__ import annotations

from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from docs_core.config import get_settings

s = get_settings()
mcp = FastMCP("devdocs-agent")


@mcp.tool()
def search_docs(query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[dict]:
    """Busca fragmentos relevantes en la documentación técnica indexada. Devuelve título, sección y texto."""
    r = httpx.post(f"{s.ingestion_url}/search", json={"query": query, "k": k, "doc_ids": doc_ids}, timeout=60)
    r.raise_for_status()
    return [{"title": x["title"], "section": x["section"], "page": x["page"], "doc_id": x["doc_id"],
             "score": round(x["score"], 3), "text": x["text"]} for x in r.json()]


@mcp.tool()
def ingest_document(path: str) -> dict:
    """Ingesta un documento local (PDF, MD o TXT) al índice. Detecta duplicados y versiones modificadas."""
    p = Path(path).expanduser()
    if not p.is_file():
        return {"status": "rejected", "detail": f"No existe el archivo {p}"}
    with p.open("rb") as f:
        r = httpx.post(f"{s.ingestion_url}/documents", files=[("files", (p.name, f))], timeout=300)
    r.raise_for_status()
    return r.json()[0]


@mcp.tool()
def ask(question: str, thread_id: str | None = None) -> dict:
    """Pregunta al asistente agéntico. Devuelve respuesta con citas, agente que respondió y pasos recorridos."""
    r = httpx.post(f"{s.agent_url}/chat", json={"question": question, "thread_id": thread_id}, timeout=180)
    r.raise_for_status()
    d = r.json()
    return {k: d[k] for k in ("answer", "agent", "steps", "sources", "thread_id")}


if __name__ == "__main__":
    mcp.run()          # transporte stdio

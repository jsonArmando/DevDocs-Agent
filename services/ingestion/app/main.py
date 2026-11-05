"""API de ingesta: carga de documentos, listado, borrado y búsqueda de depuración."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from docs_core.config import get_settings
from docs_core.models import DocumentInfo, IngestResult, RetrievedChunk, SearchRequest
from docs_core.retrieval import Retriever

from .service import IngestionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
service = IngestionService(settings)
retriever = Retriever(settings, store=service.store, embedder=service.embedder)


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.store.ensure_collection()
    yield


app = FastAPI(title="DevDocs · Ingestion API", version="1.0.0", lifespan=lifespan,
              description="Ingesta incremental de PDF/MD/TXT a Qdrant con embeddings locales.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "collection": settings.collection}


@app.post("/documents", response_model=list[IngestResult], summary="Ingestar uno o varios documentos")
async def ingest(files: list[UploadFile] = File(...)) -> list[IngestResult]:
    results = []
    for f in files:
        data = await f.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            results.append(IngestResult(source=f.filename or "?", doc_id="-", status="rejected",
                                        detail=f"Supera {settings.max_upload_mb} MB"))
            continue
        # CPU-bound (embeddings): fuera del event loop
        results.append(await run_in_threadpool(service.ingest, f.filename or "sin_nombre.txt", data))
    return results


@app.get("/documents", response_model=list[DocumentInfo], summary="Listar documentos indexados")
def list_documents() -> list[DocumentInfo]:
    return service.store.list_documents()


@app.delete("/documents/{doc_id}", summary="Eliminar un documento del índice")
def delete_document(doc_id: str) -> dict:
    if not any(d.doc_id == doc_id for d in service.store.list_documents()):
        raise HTTPException(404, f"Documento {doc_id} no existe")
    service.store.delete_by("doc_id", doc_id)
    return {"deleted": doc_id}


@app.post("/search", response_model=list[RetrievedChunk], summary="Búsqueda directa (depuración, MCP, evaluación)")
def search(req: SearchRequest) -> list[RetrievedChunk]:
    return retriever.retrieve(req.query, k=req.k, doc_ids=req.doc_ids, mode=req.mode, rerank=req.rerank)

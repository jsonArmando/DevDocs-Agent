"""API del agente: consultas, streaming SSE (reto C), memoria por thread_id, baseline lineal y diagrama del grafo."""
from __future__ import annotations

import contextvars
import json
import logging
import queue
import threading
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_core.messages import HumanMessage

from .container import Container
from .graph.builder import mermaid
from .schemas import ChatRequest, ChatResponse, Source
from .telemetry import current_telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("agent")
c = Container()

app = FastAPI(title="DevDocs · Agent API", version="1.0.0",
              description="Asistente agéntico de documentación: supervisor + subagentes (LangGraph).")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "callbacks": c.callbacks(), "recursion_limit": 25}


def _inputs(req: ChatRequest) -> dict:
    return {"messages": [HumanMessage(content=req.question)], "question": req.question, "doc_ids": req.doc_ids}


def _response(thread_id: str, state: dict, metrics) -> ChatResponse:  # noqa: ANN001
    return ChatResponse(
        thread_id=thread_id, answer=state.get("answer", ""), agent=state.get("agent") or "direct",
        route=state.get("route"), route_reason=state.get("route_reason"),
        rewritten_queries=state.get("rewritten_queries", []), steps=state.get("steps", []),
        sources=[Source(**s) for s in state.get("sources", [])], contexts=state.get("contexts", []), metrics=metrics)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": c.s.llm_provider, "models": [c.s.llm_model_fast, c.s.llm_model_smart]}


@app.post("/chat", response_model=ChatResponse, summary="Consulta en lenguaje natural (grafo agéntico)")
def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or uuid.uuid4().hex
    tel = c.new_telemetry()
    token = current_telemetry.set(tel)
    try:
        state = c.graph.invoke(_inputs(req), _config(thread_id))
    except Exception as e:
        log.exception("Error en el grafo")
        raise HTTPException(502, f"Error procesando la consulta: {type(e).__name__}") from e
    finally:
        current_telemetry.reset(token)
    return _response(thread_id, state, tel.metrics())


@app.post("/chat/stream", summary="Igual que /chat, emitiendo pasos y tokens por SSE")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    thread_id = req.thread_id or uuid.uuid4().hex

    def produce(q: queue.Queue) -> None:
        """Corre el grafo en un hilo propio: la telemetría (ContextVar) vive en un solo contexto."""
        tel = c.new_telemetry()
        current_telemetry.set(tel)
        try:
            q.put(_sse("start", {"thread_id": thread_id}))
            for ns, mode, chunk in c.graph.stream(_inputs(req), _config(thread_id),
                                                  stream_mode=["updates", "messages"], subgraphs=True):
                if mode == "updates":
                    for node in chunk:
                        prefix = ns[-1].split(":")[0] + "." if ns else ""
                        q.put(_sse("step", {"node": f"{prefix}{node}"}))
                elif mode == "messages":
                    msg, meta = chunk
                    if meta.get("langgraph_node") in {"generate", "synthesize"} and getattr(msg, "content", ""):
                        text = msg.content if isinstance(msg.content, str) else "".join(
                            b.get("text", "") for b in msg.content if isinstance(b, dict))
                        if text:
                            q.put(_sse("token", {"text": text}))
            state = c.graph.get_state(_config(thread_id)).values
            q.put(_sse("final", _response(thread_id, state, tel.metrics()).model_dump(mode="json")))
        except Exception as e:
            log.exception("Error en streaming")
            q.put(_sse("error", {"detail": f"{type(e).__name__}: {e}"}))
        finally:
            q.put(None)

    def events() -> Iterator[str]:
        q: queue.Queue = queue.Queue()
        threading.Thread(target=contextvars.copy_context().run, args=(produce, q), daemon=True).start()
        while (item := q.get()) is not None:
            yield item

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/chat/linear", response_model=ChatResponse, summary="Baseline: RAG lineal (recuperar → generar)")
def chat_linear(req: ChatRequest) -> ChatResponse:
    tel = c.new_telemetry()
    token = current_telemetry.set(tel)
    try:
        out = c.linear.run(req.question, req.doc_ids)
    finally:
        current_telemetry.reset(token)
    return ChatResponse(thread_id=req.thread_id or "-", answer=out["answer"], agent="linear_rag",
                        steps=out["steps"], sources=[Source(**s) for s in out["sources"]],
                        contexts=out["contexts"], metrics=tel.metrics())


@app.get("/threads/{thread_id}", summary="Historial de una conversación")
def thread_history(thread_id: str) -> dict:
    snapshot = c.graph.get_state({"configurable": {"thread_id": thread_id}})
    msgs = snapshot.values.get("messages", []) if snapshot else []
    if not msgs:
        raise HTTPException(404, "Conversación no encontrada")
    return {"thread_id": thread_id,
            "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.content,
                          "agent": getattr(m, "name", None)} for m in msgs]}


@app.get("/graph", response_class=PlainTextResponse, summary="Diagrama Mermaid del grafo (con subgrafos)")
def graph_diagram() -> str:
    return mermaid(c.graph)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

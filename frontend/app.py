"""Chat Streamlit: carga de documentos, respuestas en streaming con pasos, agente, fuentes y métricas."""
from __future__ import annotations

import json
import os
import uuid

import httpx
import streamlit as st

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8002")
INGESTION_URL = os.getenv("INGESTION_URL", "http://localhost:8001")
BADGE = {"documentation": "📚 Documentación", "research": "🌐 Investigación externa",
         "direct": "💬 Respuesta directa", "linear_rag": "➡️ RAG lineal"}

st.set_page_config(page_title="DevDocs Agent", page_icon="📚", layout="wide")
st.session_state.setdefault("thread_id", uuid.uuid4().hex)
st.session_state.setdefault("history", [])


# ----------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("📄 Documentos")
    files = st.file_uploader("PDF, Markdown o TXT", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if files and st.button("Ingestar", type="primary"):
        with st.spinner("Troceando y vectorizando…"):
            try:
                r = httpx.post(f"{INGESTION_URL}/documents", timeout=600,
                               files=[("files", (f.name, f.getvalue())) for f in files])
                r.raise_for_status()
                for res in r.json():
                    icon = {"created": "✅", "updated": "♻️", "duplicate": "⏭️"}.get(res["status"], "⚠️")
                    st.write(f"{icon} **{res['source']}** · {res['status']} · {res['chunks']} chunks "
                             f"{('· ' + res['detail']) if res.get('detail') else ''}")
            except httpx.HTTPError as e:
                st.error(f"No se pudo ingestar: {e}")

    try:
        docs = httpx.get(f"{INGESTION_URL}/documents", timeout=15).json()
    except httpx.HTTPError:
        docs = []
        st.warning("Servicio de ingesta no disponible")
    labels = {f"{d['title']} ({d['source']})": d["doc_id"] for d in docs}
    selected = st.multiselect(f"Filtrar búsqueda ({len(docs)} documentos)", list(labels))
    doc_ids = [labels[x] for x in selected] or None

    st.divider()
    mode = st.radio("Modo", ["Agéntico (grafo)", "RAG lineal (baseline)"], horizontal=False)
    streaming = st.toggle("Streaming de pasos (SSE)", value=True)
    st.caption(f"thread_id: `{st.session_state.thread_id[:12]}…`")
    if st.button("Nueva conversación"):
        st.session_state.thread_id, st.session_state.history = uuid.uuid4().hex, []
        st.rerun()


# ----------------------------------------------------------------------------- render
def render_details(d: dict) -> None:
    st.caption(f"{BADGE.get(d['agent'], d['agent'])}"
               + (f" · ruta: `{d['route']}` — {d.get('route_reason') or ''}" if d.get("route") else ""))
    with st.expander(f"🧭 Pasos ({len(d['steps'])})"):
        st.write(" → ".join(f"`{s}`" for s in d["steps"]))
        if d.get("rewritten_queries"):
            st.write("**Consultas reescritas:**", d["rewritten_queries"])
    if d.get("sources"):
        with st.expander(f"📎 Fuentes ({len(d['sources'])})", expanded=True):
            for s in d["sources"]:
                if s["kind"] == "web":
                    st.markdown(f"**[{s['ref']}]** [{s['title']}]({s['url']})")
                else:
                    page = f" · pág. {s['page']}" if s.get("page") else ""
                    st.markdown(f"**[{s['ref']}]** {s['title']} — *{s['section']}*{page}")
                st.caption(s["snippet"][:240] + "…")
    if d.get("metrics"):
        m = d["metrics"]
        st.caption(f"⏱ {m['latency_ms']:.0f} ms · 🔢 {m['tokens_in']}+{m['tokens_out']} tokens · "
                   f"💲 {m['cost_usd']:.5f} USD · {m['llm_calls']} llamadas LLM")


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("details"):
            render_details(turn["details"])


# ----------------------------------------------------------------------------- chat
def stream_answer(payload: dict) -> dict | None:
    steps_box, answer_box = st.status("Pensando…", expanded=True), st.empty()
    text, final = "", None
    with httpx.stream("POST", f"{AGENT_URL}/chat/stream", json=payload, timeout=300) as r:
        event = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
                if event == "step":
                    steps_box.write(f"✓ `{data['node']}`")
                elif event == "token":
                    text += data["text"]
                    answer_box.markdown(text + "▌")
                elif event == "final":
                    final = data
                elif event == "error":
                    st.error(data["detail"])
    steps_box.update(label="Listo", state="complete", expanded=False)
    answer_box.empty()
    return final


if question := st.chat_input("Pregunta sobre la documentación…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    payload = {"question": question, "thread_id": st.session_state.thread_id, "doc_ids": doc_ids}
    with st.chat_message("assistant"):
        try:
            if mode.startswith("RAG lineal"):
                d = httpx.post(f"{AGENT_URL}/chat/linear", json=payload, timeout=300).json()
            elif streaming:
                d = stream_answer(payload)
            else:
                d = httpx.post(f"{AGENT_URL}/chat", json=payload, timeout=300).json()
        except httpx.HTTPError as e:
            d = None
            st.error(f"El agente no respondió: {e}")
        if d:
            st.markdown(d["answer"])
            render_details(d)
            st.session_state.history.append({"role": "assistant", "content": d["answer"], "details": d})

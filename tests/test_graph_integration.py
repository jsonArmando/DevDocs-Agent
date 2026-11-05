"""Integración del grafo real de LangGraph con fakes (se ejecuta en CI; se omite si LangGraph no está instalado)."""
import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.graph.builder import build_graph  # noqa: E402
from app.graph.doc_agent import DocAgent  # noqa: E402
from app.graph.research_agent import ResearchAgent  # noqa: E402
from app.graph.supervisor import Supervisor  # noqa: E402
from app.schemas import RouteDecision  # noqa: E402
from fakes import FakeLLM, FakeRetriever, chunk  # noqa: E402


class RoutingLLM(FakeLLM):
    def route(self, history, question, catalog):
        if "receta" in question:
            return RouteDecision(route="direct", direct_kind="out_of_scope", standalone_question=question, reason="t")
        return RouteDecision(route="docs", standalone_question=question, reason="t")


def make_graph():
    retriever = FakeRetriever({"How do I return 404?": [chunk(1, "raise HTTPException(status_code=404)")]})
    provider = lambda q, n: [{"title": "k8s", "url": "https://kubernetes.io/releases/", "snippet": "1.35"}]  # noqa: E731
    llm = RoutingLLM()
    return build_graph(Supervisor(llm, lambda: ["FastAPI"]), DocAgent(llm, retriever, max_rewrites=1),
                       ResearchAgent(llm, provider, timeout_s=2), checkpointer=InMemorySaver())


def ask(g, q, thread="t1"):
    return g.invoke({"messages": [HumanMessage(content=q)], "question": q}, {"configurable": {"thread_id": thread}})


def test_docs_path_with_citations():
    s = ask(make_graph(), "How do I return 404?")
    assert s["agent"] == "documentation" and s["sources"][0]["kind"] == "doc"
    assert s["steps"][:3] == ["supervisor", "doc_agent.retrieve", "doc_agent.grade"] and s["steps"][-1] == "finalize"


def test_missing_context_delegates_to_research():
    s = ask(make_graph(), "latest kubernetes version?")
    assert s["agent"] == "research" and "doc_agent.no_context" in s["steps"]
    assert any(x.startswith("research_agent.") for x in s["steps"]) and s["sources"][0]["kind"] == "web"


def test_direct_and_memory_per_thread():
    g = make_graph()
    ask(g, "Hola!", thread="m")
    s = ask(g, "Recomiéndame una receta", thread="m")
    assert s["agent"] == "direct" and s["steps"] == ["supervisor", "direct_answer", "finalize"]
    assert len(s["messages"]) == 4                     # 2 turnos persistidos en el mismo thread
    assert len(ask(g, "Hola", thread="otro")["messages"]) == 2

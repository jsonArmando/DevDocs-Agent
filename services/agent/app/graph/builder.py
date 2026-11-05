"""Factory del grafo: supervisor + subgrafos de documentación e investigación, con memoria por thread_id."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..reducers import steps_reducer
from ..telemetry import timed
from .doc_agent import DocAgent, DocState
from .research_agent import ResearchAgent, ResearchState
from .supervisor import Supervisor


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # memoria conversacional (persistida por el checkpointer)
    question: str                                          # pregunta autónoma del turno
    doc_ids: list[str] | None                              # filtro opcional por documentos
    route: str
    route_reason: str
    direct_kind: str | None
    agent: str | None
    answer: str
    sources: list[dict]
    contexts: list[str]
    doc_has_context: bool | None
    rewritten_queries: list[str]
    steps: Annotated[list[str], steps_reducer]


def build_doc_subgraph(agent: DocAgent):
    g = StateGraph(DocState)
    g.add_node("retrieve", agent.retrieve)
    g.add_node("grade", agent.grade)
    g.add_node("rewrite", agent.rewrite)
    g.add_node("generate", agent.generate)
    g.add_node("no_context", agent.no_context)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", agent.after_grade, ["generate", "rewrite", "no_context"])
    g.add_edge("rewrite", "retrieve")                     # ciclo controlado por max_rewrites
    g.add_edge("generate", END)
    g.add_edge("no_context", END)
    return g.compile()


def build_research_subgraph(agent: ResearchAgent):
    g = StateGraph(ResearchState)
    g.add_node("search", agent.search)
    g.add_node("synthesize", agent.synthesize)
    g.add_edge(START, "search")
    g.add_edge("search", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def build_graph(supervisor: Supervisor, doc_agent: DocAgent, research_agent: ResearchAgent, checkpointer: Any = None):
    doc_graph = build_doc_subgraph(doc_agent)
    research_graph = build_research_subgraph(research_agent)

    def run_doc_agent(state: AgentState, config) -> dict:  # noqa: ANN001
        out = doc_graph.invoke({"question": state["question"], "query": state["question"], "queries": [],
                                "doc_ids": state.get("doc_ids"), "rewrites": 0, "steps": []}, config)
        return {"agent": "documentation", "answer": out.get("answer", ""), "sources": out.get("sources", []),
                "contexts": out.get("contexts", []), "doc_has_context": out.get("has_context", False), "rewritten_queries": out.get("queries", []),
                "steps": [f"doc_agent.{x}" for x in out.get("steps", [])]}

    def run_research_agent(state: AgentState, config) -> dict:  # noqa: ANN001
        out = research_graph.invoke({"question": state["question"], "steps": []}, config)
        return {"agent": "research", "answer": out.get("answer", ""), "sources": out.get("sources", []),
                "contexts": out.get("contexts", []),
                "steps": [f"research_agent.{x}" for x in out.get("steps", [])]}

    @timed("finalize")
    def finalize(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=state.get("answer", ""), name=state.get("agent") or "assistant")],
                "steps": ["finalize"]}

    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor.route)
    g.add_node("doc_agent", run_doc_agent)
    g.add_node("research_agent", run_research_agent)
    g.add_node("direct_answer", supervisor.direct_answer)
    g.add_node("finalize", finalize)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", supervisor.next_node, ["doc_agent", "research_agent", "direct_answer"])
    g.add_conditional_edges("doc_agent", supervisor.after_docs, ["finalize", "research_agent"])
    g.add_edge("research_agent", "finalize")
    g.add_edge("direct_answer", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def mermaid(graph) -> str:  # noqa: ANN001
    """Diagrama del grafo, incluidos los subgrafos (xray)."""
    return graph.get_graph(xray=True).draw_mermaid()

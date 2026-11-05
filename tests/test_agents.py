"""Pruebas de los nodos del grafo con fakes (sin LLM, sin Qdrant, sin LangGraph)."""
import operator
import time

from app.citations import doc_sources, is_no_context, web_sources
from app.graph.doc_agent import DocAgent
from app.graph.research_agent import EXTERNAL_PREFIX, ResearchAgent
from app.graph.supervisor import Supervisor, history_text, is_greeting
from app.reducers import RESET, steps_reducer
from app.telemetry import Telemetry, current_telemetry, timed
from app.tools.web_search import WebSearchArgs, run_search
from fakes import FakeLLM, FakeRetriever, chunk


def run_doc_graph(agent: DocAgent, question: str) -> dict:
    """Simula la semántica del subgrafo de LangGraph (merge de estado + aristas) para probar el flujo."""
    s = {"question": question, "query": question, "queries": [], "rewrites": 0, "steps": []}

    def apply(update):
        for k, v in update.items():
            s[k] = operator.add(s.get(k, []), v) if k == "steps" else v

    apply(agent.retrieve(s)); apply(agent.grade(s))
    while True:
        nxt = agent.after_grade(s)
        if nxt == "rewrite":
            apply(agent.rewrite(s)); apply(agent.retrieve(s)); apply(agent.grade(s))
        else:
            apply(getattr(agent, nxt)(s))
            return s


# ------------------------------------------------------------------ subagente de documentación
def test_doc_agent_answers_with_citations_first_try():
    r = FakeRetriever({"404?": [chunk(1, "raise HTTPException(status_code=404)"), chunk(2, "other")]})
    llm = FakeLLM()
    s = run_doc_graph(DocAgent(llm, r, max_rewrites=2), "404?")
    assert s["steps"] == ["retrieve", "grade", "generate"]
    assert s["has_context"] and s["sources"][0]["title"] == "Doc 1" and llm.graded == 1


def test_doc_agent_rewrites_then_succeeds():
    r = FakeRetriever({"raise HTTPException 404": [chunk(1, "raise HTTPException(status_code=404)")]})
    s = run_doc_graph(DocAgent(FakeLLM(), r, max_rewrites=2), "how do I return not found?")
    assert s["steps"] == ["retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]
    assert s["queries"] == ["raise HTTPException 404"] and s["has_context"]


def test_doc_agent_cycle_is_bounded_and_declares_no_context():
    r = FakeRetriever({})
    s = run_doc_graph(DocAgent(FakeLLM(rewrite_to="still nothing"), r, max_rewrites=2), "latest k8s?")
    assert s["steps"].count("rewrite") == 2 and s["steps"][-1] == "no_context"
    assert s["has_context"] is False and len(r.calls) == 3


def test_generator_no_context_marks_missing_context():
    r = FakeRetriever({"q": [chunk(1, "HTTPException basics")]})
    s = run_doc_graph(DocAgent(FakeLLM(answer="NO_CONTEXT"), r), "q")
    assert s["has_context"] is False


# ------------------------------------------------------------------ citas
def test_invented_citations_are_removed():
    answer, sources = doc_sources("Use X [1] and Y [7].", [chunk(1, "x")])
    assert "[7]" not in answer and [s.ref for s in sources] == [1]
    assert is_no_context("  no_context") and not is_no_context("It depends [1]")


def test_web_sources_default_to_all_results_when_uncited():
    _, sources = web_sources("text without refs", [{"title": "a", "url": "https://a", "snippet": "s"}])
    assert sources[0].url == "https://a" and sources[0].kind == "web"


# ------------------------------------------------------------------ supervisor
def test_greeting_fast_path_skips_llm():
    sup = Supervisor(FakeLLM(), catalog=lambda: [])
    out = sup.route({"messages": [type("M", (), {"content": "Hola, ¿qué puedes hacer?", "type": "human"})()]})
    assert out["route"] == "direct" and out["direct_kind"] == "greeting" and out["steps"][0] == RESET
    assert is_greeting("thanks!") and not is_greeting("hola, explícame cómo configurar CORS en FastAPI con cookies")


def test_router_failure_degrades_to_docs():
    sup = Supervisor(FakeLLM(), catalog=lambda: [])
    out = sup.route({"messages": [type("M", (), {"content": "boom question", "type": "human"})()]})
    assert out["route"] == "docs" and "Fallback" in out["route_reason"]


def test_supervisor_edges_and_delegation():
    assert Supervisor.next_node({"route": "research"}) == "research_agent"
    assert Supervisor.next_node({"route": "direct"}) == "direct_answer"
    assert Supervisor.after_docs({"doc_has_context": False}) == "research_agent"
    assert Supervisor.after_docs({"doc_has_context": True}) == "finalize"


def test_history_text_excludes_current_question():
    msgs = [type("M", (), {"content": c, "type": t})() for c, t in
            [("what is Depends?", "human"), ("Depends injects [1]", "ai"), ("and with yield?", "human")]]
    h = history_text(msgs)
    assert "Depends injects" in h and "and with yield" not in h


def test_steps_reducer_resets_each_turn():
    assert steps_reducer(["old", "finalize"], [RESET, "supervisor"]) == ["supervisor"]
    assert steps_reducer(["supervisor"], ["finalize"]) == ["supervisor", "finalize"]


# ------------------------------------------------------------------ investigación y herramienta
def test_search_timeout_and_errors_are_classified():
    slow = lambda q, n: (time.sleep(0.5), [])[1]          # noqa: E731
    assert run_search(slow, WebSearchArgs(query="q x"), timeout_s=0.05).error == "timeout"
    assert run_search(lambda q, n: [], WebSearchArgs(query="q x"), 1).error == "empty"
    boom = lambda q, n: (_ for _ in ()).throw(ConnectionError("down"))   # noqa: E731
    assert run_search(boom, WebSearchArgs(query="q x"), 1).error.startswith("provider_error")


def test_research_agent_marks_external_source_and_links():
    calls = []

    def provider(q, n):
        calls.append(q)
        return [{"title": "Kubernetes releases", "url": "https://kubernetes.io/releases/", "snippet": "1.35"}]

    agent = ResearchAgent(FakeLLM(), provider, timeout_s=2)
    s = {"question": "latest kubernetes?"}
    s.update(agent.search(s)); s.update(agent.synthesize(s))
    assert s["answer"].startswith(EXTERNAL_PREFIX) and s["sources"][0]["url"].startswith("https://kubernetes.io")
    assert calls == ["kubernetes latest version"]                     # usó los argumentos del tool call tipado


def test_research_agent_reports_tool_failure():
    agent = ResearchAgent(FakeLLM(), lambda q, n: [], timeout_s=1)
    s = {"question": "something obscure"}
    s.update(agent.search(s)); s.update(agent.synthesize(s))
    assert "no pude obtener" in s["answer"] and s["sources"] == []


# ------------------------------------------------------------------ telemetría
def test_telemetry_cost_and_node_timing():
    tel = Telemetry({"fast": (1.0, 5.0), "smart": (3.0, 15.0)})
    token = current_telemetry.set(tel)
    try:
        timed("doc_agent.grade")(lambda: tel.record_llm("doc_agent.grade", "fast", 1000, 100))()
    finally:
        current_telemetry.reset(token)
    m = tel.metrics()
    assert m.llm_calls == 1 and m.nodes[0].node == "doc_agent.grade"
    assert abs(m.cost_usd - (1000 / 1e6 * 1 + 100 / 1e6 * 5)) < 1e-9
    assert Telemetry({"m": (3, 15)}, free=True).metrics().cost_usd == 0

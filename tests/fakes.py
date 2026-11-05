"""Dobles de prueba compartidos."""
from docs_core.models import RetrievedChunk

from app.schemas import ChunkGrade, GradeResult, RouteDecision
from app.tools.web_search import WebSearchArgs


def chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(id=str(i), text=text, score=1.0, doc_id=f"d{i}", source=f"s{i}.md",
                          title=f"Doc {i}", section=f"Sec {i}")


class FakeRetriever:
    def __init__(self, index: dict[str, list[RetrievedChunk]]):
        self.index, self.calls = index, []

    def retrieve(self, query, k=None, doc_ids=None, mode=None, rerank=None):
        self.calls.append(query)
        return self.index.get(query, [chunk(9, "unrelated text about cooking")])


class FakeLLM:
    def __init__(self, answer="Use HTTPException [1].", rewrite_to="raise HTTPException 404"):
        self.answer, self.rewrite_to, self.graded = answer, rewrite_to, 0

    def route(self, history, question, catalog):
        if "boom" in question:
            raise RuntimeError("router down")
        return RouteDecision(route="research", standalone_question=question.upper(), reason="test")

    def grade(self, question, chunks):
        self.graded += 1                                           # una llamada por lote, no por chunk
        return GradeResult(grades=[ChunkGrade(ref=i, relevant="HTTPException" in c.text)
                                   for i, c in enumerate(chunks, start=1)])

    def rewrite(self, question, previous):
        return self.rewrite_to

    def generate(self, question, chunks):
        return self.answer

    def plan_search(self, question):
        return WebSearchArgs(query="kubernetes latest version", max_results=3)

    def synthesize(self, question, results):
        return "Kubernetes 1.35 is the latest [1]."

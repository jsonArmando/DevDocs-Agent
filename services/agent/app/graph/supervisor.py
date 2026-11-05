"""Supervisor: clasifica la consulta y decide subagente; maneja la delegación por falta de contexto."""
from __future__ import annotations

import re
from typing import Callable, Literal

from .. import prompts
from ..ports import AgentLLM
from ..reducers import RESET
from ..schemas import RouteDecision
from ..telemetry import timed

# Fast path sin LLM: los saludos cortos no consumen tokens
_GREETING = re.compile(
    r"^\s*(hola|hi|hello|hey|buen[oa]s(\s+(d[ií]as|tardes|noches))?|gracias|thanks|thank you)\b[\s!.,¡¿?]*(.{0,40})$",
    re.IGNORECASE)


def is_greeting(text: str) -> bool:
    return bool(_GREETING.match(text)) and len(text) <= 60


def history_text(messages: list, max_turns: int = 6) -> str:
    lines = []
    for m in messages[-(max_turns * 2) - 1:-1]:                      # excluye el último (la pregunta actual)
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        lines.append(f"{role}: {str(getattr(m, 'content', ''))[:600]}")
    return "\n".join(lines)


class Supervisor:
    def __init__(self, llm: AgentLLM, catalog: Callable[[], list[str]]):
        self.llm, self.catalog = llm, catalog

    @timed("supervisor")
    def route(self, s: dict) -> dict:
        messages = s.get("messages", [])
        question = str(messages[-1].content) if messages else s.get("question", "")
        if is_greeting(question):
            d = RouteDecision(route="direct", direct_kind="greeting", standalone_question=question,
                              reason="Saludo detectado por regla (sin llamada al LLM)")
        else:
            try:
                d = self.llm.route(history_text(messages), question, self.catalog())
            except Exception as e:                                   # degradación: ante fallo del router, documentación
                d = RouteDecision(route="docs", standalone_question=question, reason=f"Fallback por error del router: {e}")
        return {"question": d.standalone_question or question, "route": d.route, "route_reason": d.reason,
                "direct_kind": d.direct_kind, "answer": "", "sources": [], "contexts": [], "agent": None,
                "doc_has_context": None, "rewritten_queries": [], "steps": [RESET, "supervisor"]}

    @staticmethod
    def next_node(s: dict) -> Literal["doc_agent", "research_agent", "direct_answer"]:
        return {"docs": "doc_agent", "research": "research_agent"}.get(s.get("route", "docs"), "direct_answer")

    @staticmethod
    def after_docs(s: dict) -> Literal["finalize", "research_agent"]:
        """Delegación: si documentación no tiene contexto, investiga afuera."""
        return "finalize" if s.get("doc_has_context") else "research_agent"

    @timed("direct_answer")
    def direct_answer(self, s: dict) -> dict:
        text = prompts.GREETING if s.get("direct_kind") == "greeting" else prompts.OUT_OF_SCOPE
        return {"answer": text, "agent": "direct", "sources": [], "steps": ["direct_answer"]}

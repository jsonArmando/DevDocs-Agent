"""LLM-as-judge propio: UNA llamada por respuesta calcula todas las métricas (ahorro de costo frente a Ragas,
que hace varias llamadas por métrica). Usa el modelo rápido."""
from __future__ import annotations

from pydantic import BaseModel, Field

JUDGE_PROMPT = """You are a strict evaluator of a developer-documentation assistant.

Question: {question}
Expected answer (reference): {expected}
Retrieved contexts (numbered):
{contexts}
Assistant answer:
{answer}

Evaluate:
1. faithfulness (0-1): fraction of factual claims in the answer that are supported by the contexts.
   If there are no contexts, use 1.0 only if the answer makes no technical claims (e.g. greeting/refusal), else 0.
2. answer_relevancy (1-5): how directly the answer addresses the question.
3. correctness (1-5): agreement with the expected answer / expected behaviour.
4. context_relevant: for EACH numbered context, true if it is useful to answer the question."""


class JudgeResult(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    context_relevant: list[bool]


class Judge:
    def __init__(self, provider: str, model: str, base_url: str | None = None):
        from langchain.chat_models import init_chat_model

        kw = {"base_url": base_url} if provider == "ollama" and base_url else {}
        self.llm = init_chat_model(model, model_provider=provider, temperature=0, **kw).with_structured_output(JudgeResult)

    def __call__(self, question: str, expected: str, answer: str, contexts: list[str]) -> JudgeResult:
        ctx = "\n\n".join(f"[{i}] {c[:1200]}" for i, c in enumerate(contexts, start=1)) or "(none)"
        return self.llm.invoke([("human", JUDGE_PROMPT.format(question=question, expected=expected,
                                                               contexts=ctx, answer=answer))])

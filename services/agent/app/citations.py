"""Construcción y validación de citas (lógica pura, sin LLM)."""
from __future__ import annotations

import re

from docs_core.models import RetrievedChunk

from .schemas import Source

_REF = re.compile(r"\[(\d{1,2})\]")
NO_CONTEXT = "NO_CONTEXT"


def format_fragments(chunks: list[RetrievedChunk], max_chars: int = 1400) -> str:
    return "\n\n".join(
        f"[{i}] {c.title} — {c.section}\n{c.text[:max_chars]}" for i, c in enumerate(chunks, start=1)
    )


def is_no_context(answer: str) -> bool:
    return answer.strip().upper().startswith(NO_CONTEXT)


def doc_sources(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Source]]:
    """Mantiene solo las citas que apuntan a fragmentos reales; elimina referencias inventadas."""
    valid = set(range(1, len(chunks) + 1))
    used = sorted({int(n) for n in _REF.findall(answer)} & valid)
    cleaned = _REF.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)
    sources = [
        Source(ref=n, kind="doc", title=chunks[n - 1].title, section=chunks[n - 1].section, page=chunks[n - 1].page,
               doc_id=chunks[n - 1].doc_id, snippet=chunks[n - 1].text[:300])
        for n in used
    ]
    return cleaned.strip(), sources


def web_sources(answer: str, results: list[dict]) -> tuple[str, list[Source]]:
    valid = set(range(1, len(results) + 1))
    used = sorted({int(n) for n in _REF.findall(answer)} & valid) or sorted(valid)   # sin citas: mostrar todos los enlaces usados
    sources = [Source(ref=n, kind="web", title=results[n - 1]["title"], url=results[n - 1]["url"],
                      snippet=results[n - 1]["snippet"][:300]) for n in used]
    return answer.strip(), sources

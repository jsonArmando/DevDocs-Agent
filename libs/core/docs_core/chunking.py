"""Chunking estructural: primero por encabezados (Markdown) o páginas (PDF), luego por tamaño con solape.

Decisiones (ver docs/adr/0003-chunking.md):
* La documentación técnica ya está organizada por encabezados: respetarlos produce chunks temáticamente
  coherentes y da una "sección" citable gratis.
* Tamaño máximo ~1 400 caracteres (~350 tokens) para quedar holgados bajo el límite de 512 tokens de
  bge-small y mantener precisión en la recuperación.
* Solape de ~200 caracteres (~15 %) solo cuando una sección se parte, para no cortar una idea a la mitad.
* Los bloques de código nunca se parten por dentro si caben enteros.
* Cada chunk se vectoriza con un prefijo de contexto (título > sección): mejora el recall a costo cero.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Section

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")


def split_markdown_sections(text: str, title: str) -> list[Section]:
    """Divide Markdown por encabezados conservando la ruta jerárquica (h1 > h2 > h3)."""
    path: list[str] = []
    sections: list[Section] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(text=body, section=" > ".join(path) or title))
        buffer.clear()

    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), _clean_heading(m.group(2))
            path[:] = path[: level - 1] + [heading]
            continue
        buffer.append(line)
    flush()
    return sections


def split_plain_sections(text: str, title: str) -> list[Section]:
    """TXT: una sola sección; el troceo por párrafos lo hace split_text."""
    return [Section(text=text.strip(), section=title)] if text.strip() else []


def split_text(text: str, max_chars: int, overlap: int, min_chars: int = 0) -> list[str]:
    """Parte un texto en piezas <= max_chars respetando, en orden: bloques de código, párrafos, oraciones."""
    if len(text) <= max_chars:
        return [text]
    units = _units(text)
    pieces: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:                       # unidad gigante: corte duro por oraciones/caracteres
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(unit, max_chars))
            continue
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail      # no empezar a mitad de palabra
            current = f"{tail}\n\n{unit}".strip() if len(tail) + len(unit) + 2 <= max_chars else unit
    if current:
        pieces.append(current)
    return _merge_small(pieces, min_chars, max_chars)


def chunk_sections(sections: Iterable[Section], max_chars: int, overlap: int, min_chars: int) -> list[tuple[Section, str]]:
    """Devuelve (sección, texto del chunk) listos para construir Chunk con metadatos."""
    out: list[tuple[Section, str]] = []
    for s in sections:
        for piece in split_text(s.text, max_chars, overlap, min_chars):
            if piece.strip():
                out.append((s, piece.strip()))
    return out


def embed_text(title: str, section: str, text: str) -> str:
    """Prefijo de contexto para el embedding (contextual chunking barato)."""
    return f"{title} | {section}\n{text}"


# ----------------------------------------------------------------- utilidades internas
def _clean_heading(h: str) -> str:
    h = re.sub(r"\{#[^}]*\}", "", h)            # anclas tipo {#id} de mkdocs
    h = re.sub(r"[`*_]", "", h)
    return h.strip()


def _units(text: str) -> list[str]:
    """Párrafos, manteniendo cada bloque de código como una unidad indivisible."""
    units, buf, in_fence = [], [], False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if buf:
                units.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        units.append("\n".join(buf).strip())
    return [u for u in units if u]


def _hard_split(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:                 # sin puntuación: cortar por caracteres
            out.append(s[:max_chars])
            s = s[max_chars:]
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def _merge_small(pieces: list[str], min_chars: int, max_chars: int) -> list[str]:
    """Une piezas diminutas con la anterior si cabe (evita chunks de una línea sin contexto)."""
    merged: list[str] = []
    for p in pieces:
        if merged and len(p) < min_chars and len(merged[-1]) + len(p) + 2 <= max_chars:
            merged[-1] = f"{merged[-1]}\n\n{p}"
        else:
            merged.append(p)
    return merged

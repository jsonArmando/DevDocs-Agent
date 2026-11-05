"""Carga de documentos PDF, Markdown y TXT a secciones (Adapter por tipo de archivo)."""
from __future__ import annotations

import io
import re
from pathlib import PurePath

from .chunking import split_markdown_sections, split_plain_sections
from .models import Section

SUPPORTED = {".pdf": "pdf", ".md": "md", ".markdown": "md", ".txt": "txt"}


class UnsupportedFileError(ValueError):
    pass


def file_type_of(filename: str) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedFileError(f"Tipo no soportado: {ext or 'sin extensión'}. Usa PDF, MD o TXT.")
    return SUPPORTED[ext]


def guess_title(filename: str, text: str, file_type: str) -> str:
    if file_type == "md":
        for line in text.splitlines():
            m = re.match(r"^#\s+(.+)", line)
            if m:
                return re.sub(r"\{#[^}]*\}", "", m.group(1)).strip()
    stem = PurePath(filename).stem
    return stem.replace("-", " ").replace("_", " ").strip().capitalize()


def load_sections(filename: str, data: bytes) -> tuple[str, str, list[Section]]:
    """Devuelve (tipo, título, secciones)."""
    ftype = file_type_of(filename)
    if ftype == "pdf":
        return ftype, *_load_pdf(filename, data)
    text = data.decode("utf-8", errors="replace")
    title = guess_title(filename, text, ftype)
    text = _strip_front_matter(text)
    sections = split_markdown_sections(text, title) if ftype == "md" else split_plain_sections(text, title)
    return ftype, title, sections


def _load_pdf(filename: str, data: bytes) -> tuple[str, list[Section]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    title = meta_title.strip() or guess_title(filename, "", "pdf")
    sections: list[Section] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue                      # páginas escaneadas sin texto: requieren OCR (fuera de alcance)
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        label = f"página {i}" + (f" · {first_line[:60]}" if first_line else "")
        sections.append(Section(text=text, section=label, page=i))
    return title, sections


def _strip_front_matter(text: str) -> str:
    return re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)

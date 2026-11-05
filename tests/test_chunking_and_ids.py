import pytest

from docs_core.chunking import chunk_sections, embed_text, split_markdown_sections, split_text
from docs_core.ids import chunk_id, decide_ingest_action
from docs_core.loaders import UnsupportedFileError, file_type_of, guess_title, load_sections

MD = """---
title: front matter
---
# Query Parameters

Intro text.

## Optional parameters

Use `None` as default.

```python
# this is a comment, not a heading
async def read_items(q: str | None = None):
    return {"q": q}
```

### Deep {#deep-anchor}

Deep text.
"""


def test_markdown_sections_keep_heading_path_and_ignore_code_comments():
    _, title, sections = load_sections("query-params.md", MD.encode())
    assert title == "Query Parameters"
    names = [s.section for s in sections]
    assert names == ["Query Parameters", "Query Parameters > Optional parameters",
                     "Query Parameters > Optional parameters > Deep"]
    assert "# this is a comment" in sections[1].text           # el comentario dentro del bloque no es encabezado


def test_split_respects_max_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    pieces = split_text(text, max_chars=500, overlap=80, min_chars=50)
    assert all(len(p) <= 500 for p in pieces)
    assert len(pieces) > 3
    tail = pieces[0][-60:].split()[-1]
    assert tail in pieces[1]                                   # hay solape entre piezas consecutivas


def test_code_block_is_not_split_when_it_fits():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(30)) + "\n```"
    text = "Intro paragraph. " * 20 + "\n\n" + code + "\n\nOutro."
    pieces = split_text(text, max_chars=400, overlap=0)
    assert len(pieces) > 1 and any(code in p for p in pieces)       # el bloque completo vive en un solo chunk


def test_tiny_trailing_piece_is_merged():
    pieces = split_text("A" * 300 + "\n\n" + "B" * 300 + "\n\nshort", max_chars=320, overlap=0, min_chars=50)
    assert pieces[-1].endswith("short") and len(pieces) == 2


def test_chunk_sections_and_embed_text():
    sections = split_markdown_sections("# T\n\nhello world", "T")
    out = chunk_sections(sections, 1000, 100, 10)
    assert out[0][1] == "hello world"
    assert embed_text("T", "T > S", "x").startswith("T | T > S")


def test_ids_are_deterministic():
    assert chunk_id("abc", 1) == chunk_id("abc", 1) != chunk_id("abc", 2)


@pytest.mark.parametrize("existing,exists,expected", [
    (None, False, "created"),
    ("old", False, "updated"),
    ("h", True, "duplicate"),
    (None, True, "duplicate"),                                  # mismo contenido con otro nombre
])
def test_decide_ingest_action(existing, exists, expected):
    assert decide_ingest_action(existing, exists, "h") == expected


def test_unsupported_and_title():
    with pytest.raises(UnsupportedFileError):
        file_type_of("image.png")
    assert guess_title("docker-cheat_sheet.pdf", "", "pdf") == "Docker cheat sheet"


def _tiny_pdf(text: str) -> bytes:
    """PDF mínimo válido con una línea de texto (para probar el loader sin archivos binarios en el repo)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def test_pdf_loader_extracts_pages():
    ftype, title, sections = load_sections("docker-cheatsheet.pdf", _tiny_pdf("docker ps lists running containers"))
    assert ftype == "pdf" and sections[0].page == 1
    assert "docker ps" in sections[0].text and sections[0].section.startswith("página 1")

"""Descarga el dataset público: tutorial de FastAPI, conceptos de Pydantic (Markdown) y el cheat sheet de Docker (PDF).

Solo librería estándar. Uso: python scripts/fetch_dataset.py  -> data/raw/
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
FASTAPI = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/"
PYDANTIC = "https://raw.githubusercontent.com/pydantic/pydantic/main/docs/"

FASTAPI_FILES = [f"tutorial/{x}.md" for x in [
    "first-steps", "path-params", "query-params", "body", "query-params-str-validations",
    "path-params-numeric-validations", "body-multiple-params", "body-fields", "body-nested-models",
    "schema-extra-example", "extra-data-types", "cookie-params", "header-params", "response-model",
    "extra-models", "response-status-code", "request-forms", "request-files", "handling-errors",
    "path-operation-configuration", "encoder", "body-updates", "middleware", "cors", "sql-databases",
    "bigger-applications", "background-tasks", "metadata", "static-files", "testing", "debugging",
    "dependencies/index", "dependencies/classes-as-dependencies", "dependencies/sub-dependencies",
    "dependencies/dependencies-in-path-operation-decorators", "dependencies/global-dependencies",
    "dependencies/dependencies-with-yield", "security/index", "security/first-steps", "security/oauth2-jwt",
]]
PYDANTIC_FILES = [f"concepts/{x}.md" for x in [
    "models", "fields", "validators", "serialization", "types", "json_schema", "config",
    "pydantic_settings", "dataclasses", "alias", "unions", "strict_mode", "performance",
]]
PDFS = {"docker_cheatsheet.pdf": "https://docs.docker.com/get-started/docker_cheatsheet.pdf"}


def fetch(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "devdocs-agent-dataset/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:                       # un archivo que movieron no debe detener todo
        print(f"  ! {url}: {e}", file=sys.stderr)
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for base, prefix, files in ((FASTAPI, "fastapi", FASTAPI_FILES), (PYDANTIC, "pydantic", PYDANTIC_FILES)):
        for f in files:
            name = f"{prefix}__{f.replace('/', '__')}"          # nombres únicos y legibles en las citas
            ok += fetch(base + f, OUT / name)
    for name, url in PDFS.items():
        ok += fetch(url, OUT / name)
    print(f"{ok} archivos en {OUT}")
    if not any(OUT.glob("*.pdf")):
        print("Aviso: no se descargó ningún PDF; agrega uno manualmente a data/raw (el enunciado exige al menos uno).")


if __name__ == "__main__":
    main()

"""Ingesta todos los archivos de data/raw en lotes. Re-ejecutarlo no duplica nada (ingesta incremental)."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import httpx

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = os.getenv("INGESTION_URL", "http://localhost:8001")


def main(batch: int = 8) -> None:
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".pdf", ".md", ".txt"})
    status: Counter = Counter()
    for i in range(0, len(files), batch):
        part = files[i:i + batch]
        r = httpx.post(f"{URL}/documents", timeout=900,
                       files=[("files", (p.name, p.read_bytes())) for p in part])
        r.raise_for_status()
        for res in r.json():
            status[res["status"]] += 1
            print(f"{res['status']:9} {res['chunks']:4} chunks  {res['source']}  {res.get('detail') or ''}")
    print(dict(status))


if __name__ == "__main__":
    main()

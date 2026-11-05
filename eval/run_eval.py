"""Evalúa RAG lineal vs grafo agéntico sobre las mismas preguntas y una ablación de recuperación.

Uso:  python eval/run_eval.py            (con los servicios arriba; ver README)
Salida: eval/results/report.md y eval/results/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from judge import Judge  # noqa: E402
from metrics import aggregate, average_precision, hit_and_rr, markdown_report  # noqa: E402

HERE = Path(__file__).parent
AGENT = os.getenv("AGENT_URL", "http://localhost:8002")
INGESTION = os.getenv("INGESTION_URL", "http://localhost:8001")


def ask(endpoint: str, question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = httpx.post(f"{AGENT}{endpoint}", json={"question": question}, timeout=300)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t0) * 1000


def retrieval_ablation(items: list[dict]) -> dict:
    """hit@5 y MRR@5 por modo, usando la fuente esperada: determinista y sin costo de LLM."""
    modes = {"dense": {"mode": "dense", "rerank": False}, "hybrid": {"mode": "hybrid", "rerank": False},
             "hybrid+rerank": {"mode": "hybrid", "rerank": True}}
    out = {}
    for name, params in modes.items():
        hits, rrs = [], []
        for it in items:
            r = httpx.post(f"{INGESTION}/search", json={"query": it["question"], "k": 5, **params}, timeout=120)
            r.raise_for_status()
            h, rr = hit_and_rr([x["source"] for x in r.json()], it["expected_source"])
            hits.append(h)
            rrs.append(rr)
        out[name] = {"hit@5": round(sum(hits) / len(hits), 3), "mrr@5": round(sum(rrs) / len(rrs), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset.jsonl"))
    ap.add_argument("--skip-judge", action="store_true", help="Solo enrutamiento, latencia y costo (sin juez)")
    args = ap.parse_args()

    items = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not httpx.get(f"{INGESTION}/documents", timeout=30).json():
        sys.exit("No hay documentos indexados. Ejecuta primero: make ingest")

    judge = None if args.skip_judge else Judge(os.getenv("LLM_PROVIDER", "anthropic"),
                                               os.getenv("LLM_MODEL_FAST", "claude-haiku-5-5"),
                                               os.getenv("OLLAMA_BASE_URL"))
    rows = []
    for it in items:
        for system, endpoint in (("linear_rag", "/chat/linear"), ("agentic_graph", "/chat")):
            resp, wall_ms = ask(endpoint, it["question"])
            m = resp.get("metrics") or {}
            row = {"id": it["id"], "system": system, "category": it["category"], "question": it["question"],
                   "agent": resp["agent"] if system == "agentic_graph" else "documentation",
                   "expected_agent": it["expected_agent"], "answer": resp["answer"], "steps": resp["steps"],
                   "latency_ms": m.get("latency_ms", wall_ms), "cost_usd": m.get("cost_usd", 0.0),
                   "tokens": m.get("tokens_in", 0) + m.get("tokens_out", 0)}
            if judge:
                j = judge(it["question"], it["expected_answer"], resp["answer"], resp.get("contexts", []))
                row.update(faithfulness=round(j.faithfulness, 3), relevancy=round(j.answer_relevancy / 5, 3),
                           correctness=round(j.correctness / 5, 3),
                           context_precision=average_precision(j.context_relevant[:len(resp.get("contexts", []))]))
            rows.append(row)
            print(f"{it['id']} {system:14} agent={row['agent']:13} {row['latency_ms']:7.0f} ms "
                  f"correctness={row.get('correctness')}")

    ablation = retrieval_ablation([it for it in items if it.get("expected_source")])
    agg = aggregate(rows)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"aggregate": agg, "ablation": ablation, "rows": rows},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(markdown_report(agg, rows, ablation), encoding="utf-8")
    print(json.dumps({"aggregate": agg, "ablation": ablation}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

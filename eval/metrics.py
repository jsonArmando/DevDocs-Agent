"""Agregación de métricas (pura, testeable). El juez LLM está en judge.py."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean


def average_precision(relevant: list[bool]) -> float | None:
    """Context precision al estilo Ragas: premia que los fragmentos relevantes estén arriba."""
    if not relevant:
        return None
    hits, score = 0, 0.0
    for k, rel in enumerate(relevant, start=1):
        if rel:
            hits += 1
            score += hits / k
    return score / hits if hits else 0.0


def hit_and_rr(sources: list[str], expected: str, k: int = 5) -> tuple[int, float]:
    for rank, s in enumerate(sources[:k], start=1):
        if expected in s:
            return 1, 1.0 / rank
    return 0, 0.0


def _avg(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 3) if xs else None


def aggregate(rows: list[dict]) -> dict:
    """rows: {system, category, agent, expected_agent, faithfulness, relevancy, correctness, context_precision, latency_ms, cost_usd, tokens}"""
    out: dict = {}
    by_system = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
    for system, rs in by_system.items():
        cats = defaultdict(list)
        for r in rs:
            cats[r["category"]].append(r)
        out[system] = {
            "overall": _summary(rs),
            "by_category": {c: _summary(v) for c, v in sorted(cats.items())},
            "router_accuracy": round(mean(r["agent"] == r["expected_agent"] for r in rs), 3),
        }
    return out


def _summary(rs: list[dict]) -> dict:
    return {
        "n": len(rs),
        "faithfulness": _avg([r.get("faithfulness") for r in rs]),
        "answer_relevancy": _avg([r.get("relevancy") for r in rs]),
        "correctness": _avg([r.get("correctness") for r in rs]),
        "context_precision": _avg([r.get("context_precision") for r in rs]),
        "latency_ms_p50": _median([r["latency_ms"] for r in rs]),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rs), 5),
        "tokens_avg": round(mean(r["tokens"] for r in rs)) if rs else 0,
    }


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    m = len(xs) // 2
    return round(xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2, 1)


def markdown_report(agg: dict, rows: list[dict], ablation: dict | None) -> str:
    cols = ["faithfulness", "answer_relevancy", "correctness", "context_precision", "latency_ms_p50", "cost_usd_total", "tokens_avg"]
    lines = ["# Resultados de evaluación", "", "## Global", "",
             "| Sistema | " + " | ".join(cols) + " | router_accuracy |", "|" + "---|" * (len(cols) + 2)]
    for system, d in agg.items():
        lines.append(f"| {system} | " + " | ".join(str(d['overall'][c]) for c in cols) + f" | {d['router_accuracy']} |")
    lines += ["", "## Por categoría", ""]
    for system, d in agg.items():
        lines += [f"### {system}", "", "| Categoría | n | " + " | ".join(cols[:4]) + " |", "|" + "---|" * 6]
        for cat, s in d["by_category"].items():
            lines.append(f"| {cat} | {s['n']} | " + " | ".join(str(s[c]) for c in cols[:4]) + " |")
        lines.append("")
    if ablation:
        lines += ["## Ablación de recuperación (reto B)", "", "| Modo | hit@5 | MRR@5 |", "|---|---|---|"]
        lines += [f"| {m} | {v['hit@5']} | {v['mrr@5']} |" for m, v in ablation.items()]
        lines.append("")
    lines += ["## Detalle por pregunta", "", "| id | sistema | agente | esperado | correctness | faithfulness | pasos |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['system']} | {r['agent']} | {r['expected_agent']} | {r.get('correctness')} | "
                     f"{r.get('faithfulness')} | {' → '.join(r.get('steps', []))} |")
    return "\n".join(lines) + "\n"

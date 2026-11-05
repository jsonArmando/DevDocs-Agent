import json
from pathlib import Path

from metrics import aggregate, average_precision, hit_and_rr, markdown_report


def test_average_precision_rewards_relevant_on_top():
    assert average_precision([True, False, False]) == 1.0
    assert round(average_precision([False, False, True]), 3) == 0.333
    assert average_precision([False, False]) == 0.0 and average_precision([]) is None


def test_hit_and_reciprocal_rank():
    assert hit_and_rr(["a.md", "fastapi__tutorial__cors.md"], "tutorial__cors.md") == (1, 0.5)
    assert hit_and_rr(["a.md"], "x.md") == (0, 0.0)


def test_aggregate_and_report():
    base = {"latency_ms": 100, "cost_usd": 0.001, "tokens": 500, "steps": ["a"]}
    rows = [
        {**base, "id": "q1", "system": "agentic_graph", "category": "covered", "agent": "documentation",
         "expected_agent": "documentation", "faithfulness": 1.0, "relevancy": 1.0, "correctness": 0.8, "context_precision": 1.0},
        {**base, "id": "q2", "system": "agentic_graph", "category": "out_of_scope", "agent": "documentation",
         "expected_agent": "direct", "faithfulness": None, "relevancy": 0.4, "correctness": 0.2, "context_precision": None},
    ]
    agg = aggregate(rows)
    assert agg["agentic_graph"]["router_accuracy"] == 0.5
    assert agg["agentic_graph"]["overall"]["faithfulness"] == 1.0
    assert "| agentic_graph |" in markdown_report(agg, rows, {"dense": {"hit@5": 0.5, "mrr@5": 0.4}})


def test_dataset_is_well_formed():
    items = [json.loads(x) for x in (Path(__file__).parents[1] / "eval" / "dataset.jsonl").read_text().splitlines() if x]
    assert len(items) >= 10
    cats = {i["category"] for i in items}
    assert {"covered", "not_covered", "out_of_scope"} <= cats
    assert {i["expected_agent"] for i in items} == {"documentation", "research", "direct"}

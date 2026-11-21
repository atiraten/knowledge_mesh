"""Offline evaluation of the search agent (``knowledge-mesh eval --agent TESTSET.jsonl``).

Test set: JSON Lines, one case per line; only ``question`` is required::

    {"id": "clv", "question": "What is CLV and which tables hold the data?",
     "expected_intents": 2, "expected_datasets": ["orders", "customers"],
     "relevant_document_ids": ["doc-..."], "reference_answer": "..."}

Each case runs the full agent search (planner, task executor loops, synthesis) with the LLM judge
run synchronously, and is scored on: outcome, rule compliance, citation coverage, judge scores
(faithfulness, unsupported claims, relevance, completeness), document recall on the gathered
evidence, dataset recall, intent count, latency, tokens and cost. A case passes when it is answered,
compliant and faithful enough and finds the expected documents/datasets; the run passes when the
pass rate reaches ``min_pass_rate``. With ``ragas=True`` (``pip install -e ".[eval]"``), Ragas
faithfulness and response relevancy are added, judged by the configured Gemini/OpenAI model with the
local embedding model (see ``ragas_support``). All searches share ``session.id = eval-<run>``, so
Phoenix groups them; results go to ``data/evaluations/agent-eval-<timestamp>.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..common.ids import new_id
from ..observability import get_metrics, span
from ..observability.audit import RequestContext
from .ragas_support import evidence_contexts, run_ragas
from .retrieval_eval import _run_metadata


def load_agent_testset(path: str | Path) -> list[dict[str, Any]]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if not str(item.get("question") or "").strip():
            raise ValueError(f"line {n}: missing 'question'")
        item.setdefault("id", f"q{n}")
        items.append(item)
    return items


def _dataset_names(result: dict[str, Any]) -> set[str]:
    names = set()
    for e in result.get("evidence", []):
        if e.get("type") == "dataset":
            names.add(str(e.get("label") or "").lower())
            if e.get("catalog_ref"):
                names.add(str(e["catalog_ref"]).rsplit("/", 1)[-1].lower())
    return names


def score_case(item: dict[str, Any], result: dict[str, Any], min_compliance: float,
               min_faithfulness: float) -> dict[str, Any]:
    evaluation = result.get("evaluation") or {}
    judge = (evaluation.get("judge") or {}).get("scores") or {}
    docs = {e.get("document_id") for e in result.get("evidence", []) if e.get("document_id")}
    relevant = set(item.get("relevant_document_ids") or [])
    doc_recall = len(docs & relevant) / len(relevant) if relevant else None
    expected = {d.lower() for d in item.get("expected_datasets") or []}
    dataset_recall = len(expected & _dataset_names(result)) / len(expected) if expected else None
    expected_intents = item.get("expected_intents")
    if isinstance(expected_intents, list):
        expected_intents = len(expected_intents)
    intents = len((result.get("plan") or {}).get("intents") or [])
    reasons = []
    if result.get("outcome") != "answered":
        reasons.append("no answer")
    if (evaluation.get("compliance") or 0) < min_compliance:
        reasons.append(f"compliance {evaluation.get('compliance')} < {min_compliance}")
    if "faithfulness" in judge and judge["faithfulness"] < min_faithfulness:
        reasons.append(f"faithfulness {judge['faithfulness']} < {min_faithfulness}")
    if doc_recall is not None and doc_recall == 0:
        reasons.append("none of the relevant documents was found")
    if dataset_recall is not None and dataset_recall < 1:
        reasons.append(f"dataset recall {dataset_recall:.2f}")
    usage = result.get("usage") or {}
    return {
        "id": item["id"], "question": item["question"], "trace_id": result.get("trace_id"),
        "outcome": result.get("outcome"), "passed": not reasons, "reasons": reasons,
        "compliance": evaluation.get("compliance"), "citation_coverage": evaluation.get("citation_coverage"),
        "violations": evaluation.get("violations", []), "judge": judge,
        "unsupported_claims": (evaluation.get("judge") or {}).get("unsupported_claims", []),
        "document_recall": doc_recall, "dataset_recall": dataset_recall,
        "intents": intents, "expected_intents": expected_intents,
        "latency_ms": usage.get("latency_ms"), "tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        "cost_usd": usage.get("cost_usd"), "answer": result.get("answer"),
    }


def evaluate_agent(mesh, testset: list[dict[str, Any]], ragas: bool = False, min_pass_rate: float = 0.8,
                   min_compliance: float = 0.8, min_faithfulness: float = 0.7,
                   out_dir: str | Path | None = None) -> dict[str, Any]:
    run_id = new_id("eval")
    context = RequestContext(caller="eval", session_id=f"eval-{run_id}")
    cases, ragas_rows = [], []
    started = time.perf_counter()
    with span("evaluation.agent", "CHAIN", {"evaluation.questions": len(testset), "session.id": context.session_id}) as sp:
        for item in testset:
            result = mesh.search(item["question"], context=context, judge="sync")
            eval_context = result.pop("_eval_context", None) or {}
            if result.get("mode") != "agent":
                cases.append({"id": item["id"], "question": item["question"], "passed": False,
                              "reasons": [f"agent not used: {(result.get('agent') or {}).get('reason')}"]})
                continue
            cases.append(score_case(item, result, min_compliance, min_faithfulness))
            ragas_rows.append({"user_input": item["question"], "response": result.get("answer") or "",
                               "retrieved_contexts": evidence_contexts(eval_context.get("evidence")),
                               "reference": item.get("reference_answer") or ""})
        n = len(cases) or 1
        passed = sum(1 for c in cases if c["passed"])

        def avg(key: str, sub: str | None = None) -> float | None:
            vals = [(c.get(key) or {}).get(sub) if sub else c.get(key) for c in cases]
            vals = [v for v in vals if isinstance(v, (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else None

        summary = {
            "cases": len(cases), "passed": passed, "pass_rate": round(passed / n, 4),
            "min_pass_rate": min_pass_rate, "run_passed": passed / n >= min_pass_rate,
            "compliance": avg("compliance"), "citation_coverage": avg("citation_coverage"),
            "faithfulness": avg("judge", "faithfulness"), "answer_relevance": avg("judge", "answer_relevance"),
            "completeness": avg("judge", "completeness"), "context_relevance": avg("judge", "context_relevance"),
            "plan_adherence": avg("judge", "plan_adherence"),
            "hallucination_rate": round(sum(1 for c in cases if c.get("unsupported_claims")) / n, 4),
            "document_recall": avg("document_recall"), "dataset_recall": avg("dataset_recall"),
            "latency_ms": avg("latency_ms"), "tokens": sum(c.get("tokens") or 0 for c in cases),
            "cost_usd": round(sum(c.get("cost_usd") or 0 for c in cases), 6),
        }
        for name, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sp.set_attribute(f"evaluation.{name}", value)
    result: dict[str, Any] = {"run": {**_run_metadata(mesh, mesh.config.retrieval.top_k, len(testset)),
                                      "run_id": run_id, "session_id": context.session_id, "suite": "agent"},
                              "summary": summary, "elapsed_seconds": round(time.perf_counter() - started, 3),
                              "cases": cases}
    if ragas:
        result["ragas"] = run_ragas(mesh, ragas_rows)
    gauge = get_metrics().meter.create_gauge("mesh_eval_score", description="Latest offline evaluation scores")
    for name, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            gauge.set(value, {"metric": name, "suite": "agent"})
    out = Path(out_dir) if out_dir else mesh.config.storage.metadata_db.parent / "evaluations"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"agent-eval-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    result["output_path"] = str(path)
    return result

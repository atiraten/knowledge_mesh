"""Offline retrieval/RAG evaluation on a repeatable labeled test set.

Test set: JSON Lines, one question per line::

    {"id": "q1", "question": "...", "relevant_document_ids": ["doc-..."],
     "relevant_chunk_ids": ["chunk-..."], "reference_answer": "..."}

``relevant_document_ids`` and/or ``relevant_chunk_ids`` define relevance (chunk IDs take
precedence when present). Metrics: Recall@K (fraction of relevant items retrieved in the
top K, averaged), hit rate@K, MRR and nDCG@K — computed locally, no LLM needed.

With ``--ragas`` (requires the ``eval`` extra and answer mode), Ragas faithfulness and
answer relevancy are added using the configured answer model as the generator. Results are
written to ``data/evaluations/<timestamp>.json`` together with run metadata (config hash,
embedding model/revision, index versions, git commit) so scores link to versions.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from ..common.ids import utcnow
from ..observability import get_metrics, span


def load_testset(path: str | Path) -> list[dict[str, Any]]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        if "question" not in item:
            raise ValueError(f"line {n}: missing 'question'")
        if not item.get("relevant_document_ids") and not item.get("relevant_chunk_ids"):
            raise ValueError(f"line {n}: needs relevant_document_ids or relevant_chunk_ids")
        item.setdefault("id", f"q{n}")
        items.append(item)
    return items


def _metrics_for(retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
    top = retrieved[:k]
    hits = [1 if r in relevant else 0 for r in top]
    found = len(set(top) & relevant)
    recall = found / len(relevant) if relevant else 0.0
    rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {"recall": recall, "hit": 1.0 if found else 0.0, "rr": rr, "ndcg": dcg / ideal if ideal else 0.0}


def evaluate(mesh, testset: list[dict[str, Any]], k: int | None = None, ragas: bool = False,
             out_dir: str | Path | None = None) -> dict[str, Any]:
    k = k or mesh.config.retrieval.top_k
    per_question = []
    started = time.perf_counter()
    with span("evaluation.retrieval", "CHAIN", {"evaluation.questions": len(testset), "evaluation.k": k}) as sp:
        for item in testset:
            res = mesh.retrieve_documents([item["question"]], top_k=k, evidence_types=["documents"],
                                          expand_parents=False)
            use_chunks = bool(item.get("relevant_chunk_ids"))
            relevant = set(item.get("relevant_chunk_ids") or item.get("relevant_document_ids"))
            retrieved = [r["chunk_id"] if use_chunks else r["document_id"] for r in res["results"]]
            retrieved = list(dict.fromkeys(retrieved))
            scores = _metrics_for(retrieved, relevant, k)
            per_question.append({"id": item["id"], "question": item["question"], "retrieved": retrieved[:k],
                                 "relevant": sorted(relevant), **scores})
        n = len(per_question) or 1
        summary = {f"recall@{k}": sum(q["recall"] for q in per_question) / n,
                   f"hit_rate@{k}": sum(q["hit"] for q in per_question) / n,
                   "mrr": sum(q["rr"] for q in per_question) / n,
                   f"ndcg@{k}": sum(q["ndcg"] for q in per_question) / n}
        for name, value in summary.items():
            sp.set_attribute(f"evaluation.{name}", value)
    result: dict[str, Any] = {
        "run": _run_metadata(mesh, k, len(testset)),
        "summary": summary,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "questions": per_question,
    }
    if ragas:
        result["ragas"] = _ragas(mesh, testset, k)
    gauge = get_metrics().meter.create_gauge("mesh_eval_score", description="Latest offline evaluation scores")
    for name, value in summary.items():
        gauge.set(value, {"metric": name})
    out = Path(out_dir) if out_dir else mesh.config.storage.metadata_db.parent / "evaluations"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    result["output_path"] = str(path)
    return result


def _run_metadata(mesh, k: int, n: int) -> dict[str, Any]:
    cfg_json = mesh.config.model_dump_json(exclude={"base_dir"})
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                                cwd=mesh.config.base_dir).stdout.strip() or None
    except Exception:
        commit = None
    stats = mesh.doc_index.stats()
    return {"timestamp": utcnow(), "questions": n, "k": k, "git_commit": commit,
            "config_sha256": hashlib.sha256(cfg_json.encode()).hexdigest()[:16],
            "embedding_model": stats.get("embedding_model"), "embedding_revision": stats.get("embedding_revision"),
            "document_index_version": stats.get("version"), "chunk_sizes": mesh.config.chunking.chunk_sizes_tokens,
            "min_document_similarity": mesh.config.retrieval.min_document_similarity}


def _ragas(mesh, testset: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Ragas faithfulness/response relevancy of answer mode on the test set (evaluator: the configured model)."""
    if not mesh.answers.enabled:
        return {"status": "skipped", "reason": "answer_generation is not enabled"}
    from .ragas_support import run_ragas

    rows = []
    for item in testset:
        ans = mesh.answer(item["question"], top_k=k)
        rows.append({"user_input": item["question"], "response": ans["answer"],
                     "retrieved_contexts": [r["text"] for r in ans["evidence"]["results"]] or [""],
                     "reference": item.get("reference_answer") or ""})
    return run_ragas(mesh, rows)

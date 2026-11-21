"""Durable background jobs with paginated item manifests.

Long-running work (crawls, indexing, cataloging, key discovery, term generation) runs in a
thread pool. Each job records its status, counters, a bounded result summary, and one
manifest item per processed unit (page, document, dataset) with a status and reason.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from opentelemetry import context as otel_context

from .common.errors import MeshError, NotFound
from .common.ids import new_id, utcnow
from .observability import current_trace_id, get_metrics, sanitize_params, span
from .storage.db import Database, dumps, loads

log = logging.getLogger(__name__)

TERMINAL = {"succeeded", "partial", "failed"}


class JobContext:
    def __init__(self, store: "JobStore", job_id: str):
        self.store = store
        self.job_id = job_id
        self.counts: Counter[str] = Counter()
        self._seq = 0
        self._lock = threading.Lock()
        self.result: dict[str, Any] = {}
        self.stage: str | None = None  # label for multi-stage jobs (extract, keys, terms, publish)

    def add_item(self, ref: str | None, status: str, reason: str | None = None, **detail: Any) -> None:
        if self.stage and "stage" not in detail:
            detail["stage"] = self.stage
        with self._lock:
            self._seq += 1
            seq = self._seq
            self.counts[status] += 1
            counts = dict(self.counts)
        self.store.db.execute(
            "INSERT INTO job_items(job_id, seq, ref, status, reason, detail_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (self.job_id, seq, ref, status, reason, dumps(detail) if detail else None, utcnow()),
        )
        self.store.update_counts(self.job_id, counts)

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counts[name] += n
            counts = dict(self.counts)
        self.store.update_counts(self.job_id, counts)

    def set_result(self, **result: Any) -> None:
        self.result.update(result)


class JobStore:
    def __init__(self, db: Database):
        self.db = db

    def create(self, kind: str, params: dict[str, Any]) -> str:
        job_id = new_id("job")
        now = utcnow()
        self.db.execute(
            "INSERT INTO jobs(job_id, kind, status, params_json, counts_json, trace_id, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (job_id, kind, "queued", dumps(sanitize_params(params)), "{}", current_trace_id(), now, now),
        )
        return job_id

    def mark(self, job_id: str, status: str, **fields: Any) -> None:
        sets = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, utcnow()]
        for key, value in fields.items():
            sets.append(f"{key} = ?")
            values.append(value)
        values.append(job_id)
        self.db.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?", values)

    def update_counts(self, job_id: str, counts: dict[str, int]) -> None:
        self.db.execute(
            "UPDATE jobs SET counts_json = ?, updated_at = ? WHERE job_id = ?", (dumps(counts), utcnow(), job_id)
        )

    def get(self, job_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if row is None:
            raise NotFound(f"job not found: {job_id}")
        total_items = self.db.scalar("SELECT COUNT(*) FROM job_items WHERE job_id = ?", (job_id,))
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "params": loads(row["params_json"], {}),
            "counts": loads(row["counts_json"], {}),
            "result": loads(row["result_json"], {}),
            "error": loads(row["error_json"]),
            "trace_id": row["trace_id"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "item_count": total_items,
            "manifest_ref": f"/api/jobs/{job_id}/items",
        }

    def items(self, job_id: str, offset: int = 0, limit: int = 50, status: str | None = None) -> dict[str, Any]:
        self.get(job_id)
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        where, params = "job_id = ?", [job_id]
        if status:
            where += " AND status = ?"
            params.append(status)
        total = self.db.scalar(f"SELECT COUNT(*) FROM job_items WHERE {where}", params)
        rows = self.db.query(
            f"SELECT seq, ref, status, reason, detail_json FROM job_items WHERE {where} ORDER BY seq LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        items = [
            {"seq": r["seq"], "ref": r["ref"], "status": r["status"], "reason": r["reason"],
             "detail": loads(r["detail_json"], {})}
            for r in rows
        ]
        nxt = offset + limit if offset + limit < total else None
        return {"job_id": job_id, "total": total, "offset": offset, "limit": limit, "next_offset": nxt, "items": items}

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT job_id, kind, status, counts_json, created_at, finished_at FROM jobs ORDER BY created_at DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        )
        return [
            {"job_id": r["job_id"], "kind": r["kind"], "status": r["status"],
             "counts": loads(r["counts_json"], {}), "created_at": r["created_at"], "finished_at": r["finished_at"]}
            for r in rows
        ]

    def recover_interrupted(self) -> int:
        """Jobs left running by a crash or restart get a terminal state instead of hanging forever."""
        error = dumps({"code": "interrupted", "message": "job interrupted by application restart; resubmit it"})
        return self.db.execute(
            "UPDATE jobs SET status='failed', error_json=?, finished_at=?, updated_at=? "
            "WHERE status IN ('queued','running')",
            (error, utcnow(), utcnow()),
        )


JobFn = Callable[[JobContext], dict[str, Any] | None]


class JobRunner:
    def __init__(self, store: JobStore, workers: int = 2):
        self.store = store
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mesh-job")
        self._futures: dict[str, Future] = {}

    def submit(self, kind: str, params: dict[str, Any], fn: JobFn) -> str:
        job_id = self.store.create(kind, params)
        ctx = otel_context.get_current()
        self._futures[job_id] = self._pool.submit(self._run, job_id, kind, fn, ctx)
        return job_id

    def _run(self, job_id: str, kind: str, fn: JobFn, parent_ctx) -> None:
        token = otel_context.attach(parent_ctx)
        metrics = get_metrics()
        start = time.perf_counter()
        jc = JobContext(self.store, job_id)
        try:
            with span(f"job.{kind}", "CHAIN", {"job.id": job_id, "job.kind": kind}) as sp:
                self.store.mark(job_id, "running", started_at=utcnow())
                try:
                    result = fn(jc) or {}
                    jc.result.update(result)
                    failed = jc.counts.get("failed", 0)
                    ok = sum(v for k, v in jc.counts.items() if k != "failed")
                    status = "partial" if failed and ok else ("failed" if failed else "succeeded")
                    self.store.mark(job_id, status, result_json=dumps(jc.result), finished_at=utcnow())
                except MeshError as exc:
                    status = "failed"
                    self.store.mark(job_id, status, error_json=dumps(exc.to_dict()),
                                    result_json=dumps(jc.result), finished_at=utcnow())
                    sp.set_attribute("job.error", exc.code)
                except Exception as exc:  # noqa: BLE001 - recorded as a terminal job failure
                    status = "failed"
                    log.exception("job failed", extra={"job_id": job_id, "job_kind": kind})
                    self.store.mark(job_id, status, error_json=dumps({"code": "operation_failed",
                                                                      "message": f"{type(exc).__name__}: {exc}"[:500]}),
                                    result_json=dumps(jc.result), finished_at=utcnow())
                sp.set_attribute("job.status", status)
                metrics.jobs.add(1, {"kind": kind, "status": status})
                metrics.job_duration.record(time.perf_counter() - start, {"kind": kind})
                log.info("job finished", extra={"job_id": job_id, "job_kind": kind, "status": status,
                                                "counts": dict(jc.counts)})
        finally:
            otel_context.detach(token)

    def wait(self, job_id: str, timeout: float | None = 300) -> dict[str, Any]:
        fut = self._futures.get(job_id)
        if fut is not None:
            fut.result(timeout=timeout)
        return self.store.get(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

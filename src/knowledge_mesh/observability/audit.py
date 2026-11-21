"""Audit trail: who did what, on which resources, and whether it was allowed.

Rows go to the ``audit_events`` table and to the ``knowledge_mesh.audit`` logger (JSON logs, and
Loki when log export is on). Callers are identified by client address and a fingerprint of the API
key they used, never by the key itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

from ..common.ids import new_id, utcnow
from .telemetry import current_trace_id

log = logging.getLogger("knowledge_mesh.audit")


@dataclass(frozen=True)
class RequestContext:
    """Who is calling, from which UI session: set on traces (``user.id``, ``session.id``) and audit rows."""

    caller: str = "local"
    session_id: str | None = None


def caller_identity(client_host: str | None, api_key: str | None) -> str:
    """``key:<fingerprint>@<host>`` when an API key was presented, else ``local:<host>``."""
    host = client_host or "unknown"
    if api_key:
        return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:10]}@{host}"
    return f"local:{host}"


class AuditLog:
    def __init__(self, db):
        self.db = db

    def record(self, action: str, *, caller: str | None, decision: str = "allowed", resource_type: str | None = None,
               resource_ids: Iterable[str] = (), reason: str | None = None, trace_id: str | None = None) -> None:
        ids = list(dict.fromkeys(resource_ids))[:200]
        trace_id = trace_id or current_trace_id()
        try:
            self.db.execute(
                "INSERT INTO audit_events(event_id, ts, trace_id, caller, action, resource_type, resource_ids_json,"
                " decision, reason) VALUES (?,?,?,?,?,?,?,?,?)",
                (new_id("aud"), utcnow(), trace_id, caller, action, resource_type, json.dumps(ids), decision, reason))
        except Exception as exc:  # the audit trail must never break the request it describes
            log.error("audit write failed: %s", type(exc).__name__)
        log.info("audit", extra={"audit_action": action, "caller": caller, "decision": decision,
                                 "resource_type": resource_type, "resource_count": len(ids), "reason": reason})

    def recent(self, limit: int = 100, caller: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM audit_events"
        params: list[Any] = []
        if caller:
            sql += " WHERE caller = ?"
            params.append(caller)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        rows = self.db.query(sql, params)
        for r in rows:
            r["resource_ids"] = json.loads(r.pop("resource_ids_json") or "[]")
        return rows

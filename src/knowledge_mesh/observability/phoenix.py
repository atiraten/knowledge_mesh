"""Minimal Arize Phoenix REST client: span annotations (judge scores, rule checks, user feedback)
and links to a trace in the Phoenix UI. No Phoenix SDK is needed; failures are logged, never raised,
so observability can never break a search.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

log = logging.getLogger(__name__)


class PhoenixClient:
    def __init__(self, base_url: str, project: str, transport: httpx.BaseTransport | None = None,
                 timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.project = project
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)
        self._project_id: str | None = None
        self._lock = threading.Lock()

    def annotate(self, span_id: str | None, name: str, *, score: float | None = None, label: str | None = None,
                 explanation: str | None = None, kind: str = "LLM", metadata: dict[str, Any] | None = None) -> bool:
        """Attach one annotation (``kind`` = LLM | CODE | HUMAN) to a span. Returns False on any failure."""
        if not span_id:
            return False
        result = {k: v for k, v in (("score", score), ("label", label), ("explanation", explanation)) if v is not None}
        body = {"data": [{"span_id": span_id, "name": name, "annotator_kind": kind, "result": result,
                          "metadata": metadata or {}}]}
        try:
            resp = self._client.post("/v1/span_annotations", params={"sync": "false"}, json=body)
            if resp.status_code >= 400:
                log.warning("phoenix annotation rejected", extra={"status": resp.status_code, "annotation": name})
                return False
            return True
        except httpx.HTTPError as exc:
            log.warning("phoenix unreachable; annotation not sent", extra={"error": type(exc).__name__,
                                                                           "annotation": name})
            return False

    def _lookup_project_id(self) -> str | None:
        with self._lock:
            if self._project_id is None:
                try:
                    resp = self._client.get("/v1/projects", params={"limit": 100})
                    for p in resp.json().get("data", []) if resp.status_code < 400 else []:
                        if p.get("name") == self.project:
                            self._project_id = p.get("id")
                except (httpx.HTTPError, ValueError):
                    return None
            return self._project_id

    def trace_url(self, trace_id: str | None) -> str | None:
        """Link to the trace in the Phoenix UI (the project page if the project is not known yet)."""
        if not trace_id:
            return None
        project_id = self._lookup_project_id()
        if project_id:
            return f"{self.base_url}/projects/{project_id}/traces/{trace_id}"
        return f"{self.base_url}/projects"

    def close(self) -> None:
        self._client.close()

"""Traced, size-bounded tool calls shared by the MCP server and the search agent's tools."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from opentelemetry import context as otel_context
from opentelemetry.propagate import extract

from ..observability import current_trace_id, get_metrics, sanitize_params, span
from .errors import MeshError

log = logging.getLogger(__name__)
MAX_RESULT_CHARS = 60_000


def bounded(payload: dict[str, Any], max_chars: int = MAX_RESULT_CHARS) -> dict[str, Any]:
    """Trim the largest list fields until the JSON form fits ``max_chars``; lists the trimmed fields."""
    raw = json.dumps(payload, default=str)
    if len(raw) <= max_chars:
        return payload
    trimmed = dict(payload)
    for key in sorted(trimmed, key=lambda k: -len(json.dumps(trimmed[k], default=str))):
        if isinstance(trimmed[key], list) and trimmed[key]:
            keep = max(1, len(trimmed[key]) // 4)
            trimmed[key] = trimmed[key][:keep]
            trimmed.setdefault("truncated_fields", []).append(key)
        if len(json.dumps(trimmed, default=str)) <= max_chars:
            break
    return trimmed


def record_tool_call(source: str, tool: str, status: str, error_code: str | None = None,
                     seconds: float | None = None, result_chars: int | None = None) -> None:
    """Tool metrics shared by MCP tools and the agent's tools (``source`` = mcp | agent)."""
    metrics = get_metrics()
    metrics.tool_calls.add(1, {"source": source, "tool": tool, "status": status, "error_code": error_code or ""})
    if seconds is not None:
        metrics.tool_duration.record(seconds, {"source": source, "tool": tool})
    if result_chars is not None:
        metrics.tool_result_chars.record(result_chars, {"source": source, "tool": tool})


def call_tool(name: str, fn: Callable[[], dict[str, Any]], params: dict[str, Any],
              traceparent: str | None = None, span_prefix: str = "mcp",
              max_chars: int = MAX_RESULT_CHARS) -> dict[str, Any]:
    """Run ``fn`` in a TOOL span. Returns ``{"ok": true, ...}`` or ``{"ok": false, "error": {...}}``;
    unexpected exceptions are reported by type only (no stack traces leak to the caller).
    The span carries the tool name, sanitized parameters, status, error code, latency and result size."""
    source = "mcp" if span_prefix == "mcp" else "agent"
    token = otel_context.attach(extract({"traceparent": traceparent})) if traceparent else None
    start = time.perf_counter()
    try:
        with span(f"{span_prefix}.{name}", "TOOL", {"tool.name": name, "tool.source": source,
                                                    "tool.parameters": json.dumps(sanitize_params(params),
                                                                                  default=str)}) as sp:
            status, code, out = "ok", None, None
            try:
                result = fn()
                out = bounded({"ok": True, **result, "trace_id": current_trace_id()}, max_chars)
            except MeshError as exc:
                status, code = "error", exc.code
                out = {"ok": False, "error": exc.to_dict(), "trace_id": current_trace_id()}
            except Exception as exc:
                log.exception("tool failed", extra={"tool": name})
                status, code = "error", "operation_failed"
                out = {"ok": False, "error": {"code": "operation_failed", "message": f"{type(exc).__name__}"},
                       "trace_id": current_trace_id()}
            size = len(json.dumps(out, default=str))
            elapsed = time.perf_counter() - start
            sp.set_attribute("tool.status", status)
            sp.set_attribute("tool.result_chars", size)
            if code:
                sp.set_attribute("tool.error_code", code)
                sp.set_attribute("output.value", json.dumps(out.get("error"), default=str)[:2000])
            record_tool_call(source, name, status, code, elapsed, size)
            return out
    finally:
        if token is not None:
            otel_context.detach(token)

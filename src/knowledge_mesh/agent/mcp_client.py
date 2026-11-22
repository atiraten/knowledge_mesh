"""The search agent's tools: MCP tools only, called through the MCP protocol (``config.mcp``).

``McpToolClient`` keeps one MCP client session: in process (``mcp.mode: in_process``, an in-memory session to
this process's MCP server) or over streamable HTTP to ``mcp.url`` (the server mounted at ``/mcp`` in the app; the
API key header is sent when ``api.api_key_env`` is set). The session runs on its own event loop thread; calls are
synchronous for the harness threads and may run in parallel.

``McpToolBox`` is one search's view of those tools:

- tool list: the ``mcp.tools`` allowlist from ``list_tools`` (the knowledge-graph and SQL tools only when they are
  enabled), without the parameters the harness fills in itself (search scope, caller, traceparent);
- arguments: the search request's scope is added to every call; ``evidence_types``, ``top_k``, ``document_ids``
  and ``dataset_ids`` of ``retrieve_documents`` may only narrow it;
- ``generate_html``: a table block's ``result_ref`` (an execute_sql evidence ID such as R1) is filled with that
  result's columns and rows before the call; the returned fragment is kept for the page, not shown to the model;
- ``retrieve_documents`` and ``query_knowledge_graph`` are called with ``view: agent`` (their compact shape);
- results: every item with ``evidence_ref`` / ``evidence_kind`` is registered in the search's ``EvidenceLedger``
  (its ``evidence_data`` when present, else the item) and gets its evidence ID (D, G, K, T, B, R, H); the model
  sees the item without these fields. Items already shown in the task become ``seen_before`` stubs,
  long results are shortened to ``agent.max_observation_chars`` and text that looks like an instruction to the
  model is counted.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from opentelemetry.propagate import inject

from ..common.errors import InvalidArgument, MeshError
from ..common.guardrails import suspicious_instruction
from ..common.tooling import record_tool_call
from .ledger import EvidenceLedger
from .plan import AgentUnavailable, error_text, inline_json_schema

log = logging.getLogger(__name__)
MAX_QUERIES = 5  # queries per retrieve_documents call
CONNECT_TIMEOUT_SECONDS = 30
_COMPACT_KEYS = ("id", "title", "label", "name", "type", "catalog_ref", "score", "matched_queries")
EVIDENCE_FIELDS = ("evidence_ref", "evidence_kind", "evidence_detail", "evidence_data")
PREFIX = {"document_passage": "D", "document_text": "D", "glossary": "G", "graph": "K", "dataset": "T",
          "business_term": "B", "business_entity": "B", "business_definition": "B", "query_result": "R",
          "html_fragment": "H"}
KIND_LABELS = {"document_passage": "passages", "document_text": "document texts", "glossary": "glossary matches",
               "graph": "graph facts", "dataset": "tables", "business_term": "business terms",
               "business_entity": "business entities", "business_definition": "definitions",
               "query_result": "query results", "html_fragment": "HTML fragments"}
# Parameters the harness fills in (or leaves at their default); never shown to the model.
HIDDEN = {
    "retrieve_documents": {"source_types", "structured_source_ids", "expand_parents", "rerank", "view"},
    "query_knowledge_graph": {"structured_source_ids", "dataset_ids", "document_ids", "view"},
    "generate_sql": {"structured_source_ids", "dataset_ids"},
    "execute_sql": {"structured_source_ids", "dataset_ids", "caller"},
    "read_document": {"max_chars"},
}
ALWAYS_HIDDEN = {"traceparent"}
GRAPH_TOOLS = {"query_knowledge_graph"}
SQL_TOOLS = {"generate_sql", "execute_sql"}


@dataclass
class SearchScope:
    """Filters from the search request. They bound every MCP tool call the agent makes."""

    top_k: int | None = None
    evidence_types: list[str] | None = None
    document_ids: list[str] | None = None
    source_types: list[str] | None = None
    structured_source_ids: list[str] | None = None
    dataset_ids: list[str] | None = None
    expand_parents: bool | None = None
    rerank: bool | None = None
    output_format: str = "text"      # "html": the user asked for HTML output (generate_html)
    caller: str | None = None        # for the audit trail of execute_sql


@dataclass
class ToolResult:
    observation: dict[str, Any]
    status: str
    summary: str
    evidence_ids: list[str] = field(default_factory=list)
    compacted: int = 0          # listed items replaced by a seen_before stub
    truncated: bool = False     # the observation was shortened to fit max_observation_chars
    injection_flags: int = 0    # items whose text looks like an instruction to the model


# -- MCP client session -------------------------------------------------------------------------------
class McpToolClient:
    """One MCP client session for the harness (``mcp.mode``), opened on first use and reopened once after a failure."""

    def __init__(self, config, mesh: Callable[[], Any]):
        self.config = config
        self._mesh = mesh
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._owner: concurrent.futures.Future | None = None
        self._stop: asyncio.Event | None = None
        self._client = None
        self._tools: list[Any] | None = None

    def _target(self):
        if self.config.mcp.mode == "in_process":
            from ..mcp_server import build_server

            return build_server(self._mesh())
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        env = self.config.api.api_key_env
        key = os.environ.get(env) if env else None
        return streamable_http_client(self.config.mcp.url,
                                      http_client=create_mcp_http_client(headers={"X-API-Key": key} if key else None))

    async def _session(self, target, ready: concurrent.futures.Future) -> None:
        from mcp import Client

        self._stop = asyncio.Event()
        try:
            async with Client(target) as client:
                self._client = client
                ready.set_result(None)
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - reported to the waiting caller, or logged
            if not ready.done():
                ready.set_exception(exc)
            else:
                log.warning("MCP client session ended", extra={"error": error_text(exc)[:300]})
        finally:
            self._client = None

    def _connect(self):
        """The open client session (opens it when needed). Caller holds ``_lock``."""
        if self._client is not None:
            return self._client
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
            self._thread.start()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._owner = asyncio.run_coroutine_threadsafe(self._session(self._target(), ready), self._loop)
        ready.result(CONNECT_TIMEOUT_SECONDS)
        return self._client

    def _disconnect(self) -> None:
        """Close the session (the event loop thread stays). Caller holds ``_lock``."""
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._owner is not None:
            try:
                self._owner.result(5)
            except Exception:  # noqa: BLE001 - closing anyway
                pass
        self._client, self._owner, self._tools = None, None, None

    def _call(self, fn: Callable[[Any], Any]):
        """Run ``fn(client)`` on the session loop; on a transport failure reopen the session and retry once."""
        for attempt in (1, 2):
            with self._lock:
                try:
                    client = self._connect()
                except Exception as exc:
                    self._disconnect()
                    raise AgentUnavailable(f"MCP server unavailable ({self.config.mcp.mode}): "
                                           f"{error_text(exc)}") from exc
            try:
                return asyncio.run_coroutine_threadsafe(fn(client), self._loop).result()
            except Exception as exc:
                with self._lock:
                    if self._client is client:
                        self._disconnect()
                if attempt == 2:
                    raise AgentUnavailable(f"MCP call failed ({self.config.mcp.mode}): {error_text(exc)}") from exc
                log.warning("MCP call failed; reconnecting", extra={"error": error_text(exc)[:300]})

    def list_tools(self) -> list[Any]:
        if self._tools is None:
            self._tools = self._call(lambda c: c.list_tools()).tools
        return self._tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The tool's result as a dict, or ``{"ok": false, "error": ...}`` when the server reports an error."""
        res = self._call(lambda c: c.call_tool(name, arguments))
        if res.is_error:
            text = " ".join(getattr(x, "text", "") for x in res.content).strip()
            code = "invalid_argument" if "validation error" in text else "tool_error"
            return {"ok": False, "error": {"code": code, "message": text[:1000] or "tool error"}}
        # the text content is the tool's own result; structured content adds the output schema's defaults
        text = next((x.text for x in res.content if getattr(x, "text", None)), None)
        try:
            out = json.loads(text) if text else dict(res.structured_content or {})
        except ValueError:
            out = dict(res.structured_content or {})
        return {k: v for k, v in out.items() if v is not None}

    def close(self) -> None:
        with self._lock:
            self._disconnect()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread is not None:
                    self._thread.join(5)
                self._loop, self._thread = None, None


# -- one search's tools ---------------------------------------------------------------------------------
class McpToolBox:
    def __init__(self, client: McpToolClient, config, ledger: EvidenceLedger, scope: SearchScope | None):
        self.client = client
        self.config = config
        self.ledger = ledger
        self.scope = scope or SearchScope()
        self.max_chars = config.agent.max_observation_chars
        self._specs: list[dict[str, Any]] | None = None

    def _offered(self) -> set[str]:
        names = set(self.config.mcp.tools)
        if not self.config.knowledge_graph.enabled:
            names -= GRAPH_TOOLS
        if not self.config.sql_execution.enabled:
            names -= SQL_TOOLS
        return names

    def specs(self) -> list[dict[str, Any]]:
        if self._specs is None:
            offered = self._offered()
            specs = []
            for t in self.client.list_tools():
                if t.name not in offered:
                    continue
                schema = inline_json_schema(t.input_schema or {"type": "object", "properties": {}})
                hidden = ALWAYS_HIDDEN | HIDDEN.get(t.name, set())
                schema["properties"] = {k: v for k, v in (schema.get("properties") or {}).items() if k not in hidden}
                if "required" in schema:
                    schema["required"] = [r for r in schema["required"] if r not in hidden]
                specs.append({"name": t.name, "description": (t.description or "").strip(), "parameters": schema})
            self._specs = specs
        return self._specs

    def call(self, name: str | None, arguments: dict[str, Any], seen: set[str] | frozenset = frozenset()) -> ToolResult:
        """Run one MCP tool call. Listed items whose evidence ID is in ``seen`` (already shown in this task's
        context) are returned as ``seen_before`` without their text."""
        offered = [s["name"] for s in self.specs()]
        if name not in offered:
            record_tool_call("agent", "unknown", "error", "unknown_tool")
            msg = f"unknown tool {name!r}; available tools: {', '.join(offered)}"
            return ToolResult({"ok": False, "error": {"code": "unknown_tool", "message": msg}}, "error", msg)
        try:
            args = self._arguments(name, arguments if isinstance(arguments, dict) else {})
        except MeshError as exc:
            record_tool_call("agent", name, "error", exc.code)
            return ToolResult({"ok": False, "error": exc.to_dict()}, "error", f"error: {exc.message}")
        try:
            out = self.client.call_tool(name, args)
        except AgentUnavailable as exc:
            return ToolResult({"ok": False, "error": {"code": "mcp_unavailable", "message": exc.message}}, "error",
                              f"error: {exc.message}")
        out.pop("trace_id", None)
        if not out.get("ok", True):
            return ToolResult(out, "error", f"error: {(out.get('error') or {}).get('message')}")
        observation, ids = self._register(out)
        if name == "generate_html" and isinstance(observation.get("html"), str):
            observation["chars"] = len(observation.pop("html"))
            observation["note"] = "The fragment will be shown on the web page; it is not repeated here."
        ids = list(dict.fromkeys(ids))
        flags = sum(1 for text in _strings(observation) if len(text) > 40 and suspicious_instruction(text))
        compacted = 0
        if seen:
            observation, compacted = _compact_lists(observation, seen)
        fitted = fit_observation(observation, self.max_chars)
        return ToolResult(fitted, "ok", self._summary(name, out, ids), ids, compacted, "truncated" in fitted, flags)

    # -- arguments -------------------------------------------------------------------------------
    def _arguments(self, name: str, asked: dict[str, Any]) -> dict[str, Any]:
        s = self.scope
        args = {k: v for k, v in asked.items() if k not in ALWAYS_HIDDEN | HIDDEN.get(name, set())}
        if name == "retrieve_documents":
            top_k = args.get("top_k")
            args.update({
                "top_k": min(top_k, s.top_k) if top_k and s.top_k else (top_k or s.top_k),
                "evidence_types": _narrow(args.get("evidence_types"), s.evidence_types, "evidence types"),
                "document_ids": _narrow(args.get("document_ids"), s.document_ids, "documents"),
                "dataset_ids": _narrow(args.get("dataset_ids"), s.dataset_ids, "datasets"),
                "source_types": s.source_types, "structured_source_ids": s.structured_source_ids,
                "expand_parents": s.expand_parents, "rerank": s.rerank, "view": "agent"})
        elif name == "query_knowledge_graph":
            args.update({"structured_source_ids": s.structured_source_ids, "dataset_ids": s.dataset_ids,
                         "document_ids": s.document_ids, "view": "agent"})
        elif name == "generate_sql":
            args.update({"structured_source_ids": s.structured_source_ids, "dataset_ids": s.dataset_ids})
        elif name == "execute_sql":
            args.update({"structured_source_ids": s.structured_source_ids, "dataset_ids": s.dataset_ids,
                         "caller": s.caller or "agent"})
        elif name == "generate_html":
            args = self._resolve_result_refs(args)
        carrier: dict[str, str] = {}
        inject(carrier)  # continue this search's trace in the MCP tool spans
        if carrier.get("traceparent"):
            args["traceparent"] = carrier["traceparent"]
        return {k: v for k, v in args.items() if v is not None}

    def _resolve_result_refs(self, args: dict[str, Any]) -> dict[str, Any]:
        """Table blocks with ``result_ref`` get the columns and rows of that execute_sql result of this search."""
        blocks = []
        for b in args.get("blocks") or []:
            if isinstance(b, dict) and b.get("type") == "table" and b.get("result_ref"):
                item = self.ledger.item(str(b["result_ref"]))
                if item is None or item.kind != "query_result":
                    raise InvalidArgument(f"result_ref {b['result_ref']} is not an execute_sql result of this search")
                flagged = [c["name"] for c in item.data["columns"] if c.get("sensitive")]
                b = {**b, "result_ref": None, "columns": [c["name"] for c in item.data["columns"]],
                     "rows": item.data["rows"],
                     "sensitive_columns": list(dict.fromkeys(flagged + list(b.get("sensitive_columns") or [])))}
            blocks.append(b)
        return {**args, "blocks": blocks} if "blocks" in args else args

    # -- results ---------------------------------------------------------------------------------
    def _register(self, out: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Give every evidence item its ledger ID; returns the observation (IDs in, evidence fields out)."""
        ids: list[str] = []

        def walk(value: Any) -> Any:
            if isinstance(value, dict):
                kind = value.get("evidence_kind")
                shown = {k: v for k, v in value.items() if k not in EVIDENCE_FIELDS}
                if isinstance(value.get("evidence_ref"), str) and kind in PREFIX:
                    data = value["evidence_data"] if isinstance(value.get("evidence_data"), dict) else shown
                    ident = self.ledger.add(PREFIX[kind], value["evidence_ref"], kind, data,
                                            replace=bool(value.get("evidence_detail")))
                    ids.append(ident)
                    return {"id": ident, **{k: walk(v) for k, v in shown.items()}}
                return {k: walk(v) for k, v in shown.items()}
            if isinstance(value, list):
                return [walk(v) for v in value]
            return value

        return walk(out), ids

    def _summary(self, name: str, out: dict[str, Any], ids: list[str]) -> str:
        """One line per result, shown when older steps are shortened: item counts, rows, or the status."""
        status = out.get("status")
        if status == "error":
            return f"error: {(out.get('error') or {}).get('message')}"
        counts = Counter(it.kind for i in ids if (it := self.ledger.item(i)) is not None
                         and it.kind not in ("query_result", "html_fragment"))
        parts = [f"{n} {KIND_LABELS.get(k, k)}" for k, n in counts.items()]
        if isinstance(out.get("row_count"), int):
            parts.append(f"{out['row_count']} rows")
        head = f"{name} {status}" if isinstance(status, str) and status != "ok" else ""
        if not parts:
            return head or f"{name} ok"
        return (f"{head}: " if head else "") + ", ".join(parts)


def _compact_lists(value: Any, seen: set[str] | frozenset) -> tuple[Any, int]:
    """Listed items already shown in this task: their ID and labels only (anywhere in the result)."""
    count = 0

    def walk(v: Any, listed: bool) -> Any:
        nonlocal count
        if isinstance(v, dict):
            if listed and v.get("id") in seen:
                count += 1
                return {**{k: v[k] for k in _COMPACT_KEYS if v.get(k) is not None}, "seen_before": True}
            return {k: walk(x, False) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, True) for x in v]
        return v

    return walk(value, False), count


def fit_observation(obs: dict[str, Any], limit: int) -> dict[str, Any]:
    """Shrink an observation until its JSON fits ``limit``: shorten long strings first, then drop
    trailing list items (named in ``truncated``)."""
    size = lambda o: len(json.dumps(o, default=str))  # noqa: E731
    if size(obs) <= limit:
        return obs
    longest = max((len(s) for s in _strings(obs)), default=0)
    cap = longest
    while size(obs) > limit and cap > 200:
        cap = max(200, cap // 2)
        obs = _cut_strings(obs, cap)
    truncated: dict[str, int] = {}
    while size(obs) > limit:
        lists = [(k, v) for k, v in obs.items() if isinstance(v, list) and v]
        if not lists:
            break
        key, items = max(lists, key=lambda kv: size(kv[1]))
        truncated[key] = truncated.get(key, 0) + 1
        obs = {**obs, key: items[:-1]}
    if truncated:
        obs["truncated"] = {k: f"{n} item(s) omitted" for k, n in truncated.items()}
    return obs


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _cut_strings(value: Any, cap: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= cap else value[:cap] + "…"
    if isinstance(value, dict):
        return {k: _cut_strings(v, cap) for k, v in value.items()}
    if isinstance(value, list):
        return [_cut_strings(v, cap) for v in value]
    return value


def _narrow(asked: list[str] | None, allowed: list[str] | None, what: str) -> list[str] | None:
    if allowed is None:
        return asked
    if asked is None:
        return list(allowed)
    kept = [x for x in asked if x in allowed]
    if not kept:
        raise InvalidArgument(f"{what} are outside this search's scope", {"allowed": allowed})
    return kept

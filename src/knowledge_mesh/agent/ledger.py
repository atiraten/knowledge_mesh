"""Evidence gathered by the agent's tools during one search.

Every item a tool returns gets a stable ID shared by all intents of the search: D (document
passage or text), G (glossary match), K (knowledge-graph fact or path), T (table/dataset), B (business
term, entity or definition), R (rows returned by execute_sql), H (HTML fragment from generate_html).
The same chunk, dataset, graph node or concept always keeps its first ID. The ledger renders the
evidence blocks for synthesis and resolves citations. ``export``/``restore`` carry it across a
clarification round.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..retrieval.evidence import document_block, glossary_block, graph_block


@dataclass
class EvidenceItem:
    id: str
    kind: str
    data: dict[str, Any]
    citation: dict[str, Any] = field(default_factory=dict)


class EvidenceLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, EvidenceItem] = {}
        self._by_key: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def add(self, prefix: str, key: str, kind: str, data: dict[str, Any], replace: bool = False) -> str:
        """Register an item; returns its ID. ``replace`` upgrades a known item with richer data."""
        with self._lock:
            ident = self._by_key.get(key)
            if ident is not None:
                if replace:
                    self._items[ident].data = data
                    self._items[ident].kind = kind
                return ident
            self._counters[prefix] = self._counters.get(prefix, 0) + 1
            ident = f"{prefix}{self._counters[prefix]}"
            self._by_key[key] = ident
            self._items[ident] = EvidenceItem(ident, kind, data)
            return ident

    def item(self, ident: str) -> EvidenceItem | None:
        return self._items.get(ident)

    def __contains__(self, ident: str) -> bool:
        return ident in self._items

    def __len__(self) -> int:
        return len(self._items)

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def render(self, ids: list[str]) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Evidence blocks and citation catalog for the given IDs (shared parent text printed once)."""
        printed: dict[str, str] = {}
        blocks: list[str] = []
        catalog: dict[str, dict[str, Any]] = {}
        for ident in ids:
            item = self._items.get(ident)
            if item is None:
                continue
            block, entry = _render(item, printed)
            blocks.append(block)
            catalog[ident] = entry
        return blocks, catalog

    def catalog(self) -> dict[str, dict[str, Any]]:
        return self.render(self.ids())[1]

    def export(self) -> dict[str, Any]:
        with self._lock:
            return {"items": [{"id": i.id, "kind": i.kind, "data": i.data} for i in self._items.values()],
                    "by_key": dict(self._by_key), "counters": dict(self._counters)}

    @classmethod
    def restore(cls, state: dict[str, Any] | None) -> "EvidenceLedger":
        ledger = cls()
        for it in (state or {}).get("items", []):
            ledger._items[it["id"]] = EvidenceItem(it["id"], it["kind"], it["data"])
        ledger._by_key = dict((state or {}).get("by_key", {}))
        ledger._counters = dict((state or {}).get("counters", {}))
        return ledger

    def public(self) -> list[dict[str, Any]]:
        """The evidence list returned to API clients (no long texts)."""
        out = []
        for ident, entry in self.catalog().items():
            item = self._items[ident]
            view = {"id": ident, **entry}
            if item.kind in {"document_passage", "document_text"}:
                view["excerpt"] = (item.data.get("text") or "")[:300]
            out.append(view)
        return out


def _render(item: EvidenceItem, printed: dict[str, str]) -> tuple[str, dict[str, Any]]:
    d, ident = item.data, item.id
    if item.kind == "document_passage":
        return document_block(ident, d, printed)
    if item.kind == "glossary":
        return glossary_block(ident, d)
    if item.kind == "graph":
        return graph_block(ident, d)
    if item.kind == "document_text":
        loc = ", ".join(x for x in (d.get("title"), f"page {d['page']}" if d.get("page") else None,
                                    f"section {d['section']}" if d.get("section") else None,
                                    f"characters {d['offset']}-{d['offset'] + len(d['text'])}") if x)
        entry = {"type": "document_text", "ref": f"/api/documents/{d['document_id']}",
                 "document_id": d["document_id"], "page": d.get("page"), "section": d.get("section"),
                 "label": d.get("title") or d.get("source_uri")}
        return f"[{ident}] document text ({loc}):\n{d['text']}", entry
    if item.kind == "dataset":
        return _dataset_block(ident, d)
    if item.kind == "query_result":
        return _query_block(ident, d)
    if item.kind == "html_fragment":
        entry = {"type": "html_fragment", "ref": None, "label": d.get("title") or "HTML output"}
        return (f"[{ident}] HTML fragment generated for web display ({d.get('blocks_rendered')} block(s)"
                f"{', truncated' if d.get('truncated') else ''}); it shows content from the cited evidence."), entry
    if item.kind == "business_definition":
        links = ", ".join(t["label"] for t in d.get("terms", []))
        text = f"[{ident}] business definition '{d['name']}': {d['definition']}"
        if links:
            text += f" (defines: {links})"
        if d.get("document"):
            text += f" (defined in: {d['document'].get('title') or d['document']['document_id']}" \
                    + (f", {d['location']}" if d.get("location") else "") + ")"
        return text, {"type": "business_definition", "ref": d["ref"], "graph_ref": d.get("graph_ref"),
                      "label": d["name"]}
    # business_term / business_entity
    label = d.get("label") or d.get("slug")
    kind = item.kind.replace("_", " ")
    parts = [f"[{ident}] {kind} '{label}'"]
    if d.get("descriptions"):
        parts.append("description: " + "; ".join(d["descriptions"][:3]))
    if d.get("columns"):
        parts.append("bound to columns: " + ", ".join(
            f"{c['catalog_ref']}" + (f" ({c['description']})" if c.get("description") else "")
            for c in d["columns"][:20]))
    if d.get("datasets"):
        parts.append("represented by tables: " + ", ".join(t["catalog_ref"] for t in d["datasets"][:20]))
    if d.get("definitions"):
        parts.append("definitions: " + ", ".join(x["name"] for x in d["definitions"][:10]))
    return "; ".join(parts), {"type": item.kind, "ref": d["ref"], "graph_ref": d.get("graph_ref"), "label": label}


def _dataset_block(ident: str, d: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    head = f"[{ident}] table {d['catalog_ref']}"
    facts = [f"name {d['name']}"]
    if d.get("schema"):
        facts.append(f"schema {d['schema']}")
    if d.get("business_entity"):
        facts.append(f"business entity {d['business_entity']}")
    if d.get("row_count") is not None:
        facts.append(f"{d['row_count']} rows")
    lines = [f"{head} ({', '.join(facts)})"]
    cols = d.get("columns") or []
    if cols:
        lines.append("columns: " + "; ".join(
            f"{c['name']} {c.get('data_type') or ''}".strip()
            + (f" — term '{c['business_term']}'" if c.get("business_term") else "")
            for c in cols))
        if d.get("columns_total", len(cols)) > len(cols):
            lines.append(f"(first {len(cols)} of {d['columns_total']} columns)")
    for k in d.get("keys") or []:
        lines.append(_key_text(k))
    for k in d.get("referenced_by") or []:
        lines.append("referenced by " + _key_text(k))
    entry = {"type": "dataset", "ref": f"/api/datasets/{d['dataset_id']}", "catalog_ref": d["catalog_ref"],
             "label": d["name"]}
    return "\n".join(lines), entry


QUERY_BLOCK_ROWS = 50


def _query_block(ident: str, d: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Rows returned by execute_sql: record-level evidence."""
    cols = [c["name"] for c in d.get("columns", [])]
    rows = d.get("rows", [])
    more = d["row_count"] > len(rows[:QUERY_BLOCK_ROWS]) or d.get("truncated")
    lines = [f"[{ident}] query result from source {d['source_id']} ({d['row_count']} row(s)"
             + (", more rows exist" if d.get("truncated") else "") + f"); SQL: {d['sql']}",
             "columns: " + " | ".join(cols)]
    if not rows:
        lines.append("(the query returned no rows)")
    for r in rows[:QUERY_BLOCK_ROWS]:
        lines.append(" | ".join("" if v is None else str(v) for v in r))
    if more and len(rows) > QUERY_BLOCK_ROWS:
        lines.append(f"(first {QUERY_BLOCK_ROWS} rows shown)")
    entry = {"type": "query_result", "ref": None, "source_id": d["source_id"], "label": f"SQL on {d['source_id']}",
             "row_count": d["row_count"], "sql": d["sql"]}
    return "\n".join(lines), entry


def _key_text(k: dict[str, Any]) -> str:
    """``primary key (id) [declared]``, ``foreign key (cust) -> customers(id) [inferred]`` or, for a key of
    another table pointing here, ``foreign key orders(customer_id) -> customers(customer_id) [declared]``."""
    origin = "inferred" if k.get("origin") == "inferred" else "declared"
    cols = ", ".join(k.get("columns") or [])
    owner = f"{k['from_dataset']}({cols})" if k.get("from_dataset") else f"({cols})"
    text = f"{k['type']} key {owner}"
    if k.get("references"):
        text += f" -> {k['references']}"
    return f"{text} [{origin}]"

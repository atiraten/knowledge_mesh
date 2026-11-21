"""Read-only search tools offered to the agent's executor.

``query_knowledge_graph``, ``execute_sql`` and ``generate_html`` share their implementation with the
MCP server (``KnowledgeMesh.query_knowledge_graph`` / ``execute_sql`` / ``generate_html``);
``generate_sql`` (agent only) asks the agent model for a read-only query built from this task's catalog
and graph evidence and checks it without running it.

Each tool has a pydantic argument model (its JSON schema is what the model sees), runs in a
traced TOOL span, registers what it returns in the search's ``EvidenceLedger`` (so every item
carries an evidence ID) and returns a size-bounded observation. ``retrieve_context`` takes a list
of queries, searched in parallel and merged into one deduplicated result. Within one task, a
listed item that was already shown is returned as ``seen_before`` without its text. Raw profiled
values (most common values) are never passed to the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..common.errors import InvalidArgument, NotConfigured
from ..common.guardrails import suspicious_instruction
from ..common.html_render import HtmlRequest
from ..common.tooling import call_tool, record_tool_call
from ..graph.retriever import RECORD_NOTE, KGQueryRequest
from .ledger import EvidenceLedger
from .plan import inline_schema, parse_json_object, prompt_version

READ_CHARS = 4000
MAX_QUERIES = 5  # queries per retrieve_context call
_COMPACT_KEYS = ("id", "title", "label", "name", "type", "catalog_ref", "score", "matched_queries")


@dataclass
class SearchScope:
    """Filters from the search request. They bound every ``retrieve_context`` call the agent makes."""

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


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrieveArgs(_Args):
    queries: list[str] = Field(
        min_length=1, max_length=MAX_QUERIES,
        description="Search queries for the current task, searched in parallel and merged into one result. "
                    "Start with the task's retrieval queries; refine within the intent to retry.")
    evidence_types: list[Literal["documents", "glossary", "graph"]] | None = Field(
        default=None, description="Indexes to search (default: all three).")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Results per index (default 5).")
    document_ids: list[str] | None = Field(default=None, description="Only passages of these documents.")
    dataset_ids: list[str] | None = Field(default=None, description="Only glossary entries of these datasets.")

    @model_validator(mode="before")
    @classmethod
    def _single_query(cls, data: Any) -> Any:
        if isinstance(data, dict) and "queries" not in data and isinstance(data.get("query"), str):
            data = {**{k: v for k, v in data.items() if k != "query"}, "queries": [data["query"]]}
        return data


class ReadDocumentArgs(_Args):
    document_id: str = Field(min_length=1)
    chunk_id: str | None = Field(default=None, description="A retrieved chunk: returns it with its parent context.")
    page: int | None = Field(default=None, ge=0)
    section: str | None = Field(default=None, max_length=200, description="Text contained in a section heading.")
    offset: int = Field(default=0, ge=0, description="Character offset; use next_offset to continue reading.")


class FindDatasetsArgs(_Args):
    name_contains: str | None = Field(default=None, max_length=200,
                                      description="Text in the table name, schema or business entity label.")
    column_contains: str | None = Field(default=None, max_length=200,
                                        description="Text in a column name or the column's business term.")
    source_id: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=25, ge=1, le=50)


class DatasetArgs(_Args):
    dataset_id: str = Field(min_length=1)


class SlugArgs(_Args):
    slug: str = Field(min_length=1, max_length=200, description="Slug from a glossary result or a ref.")


class DefinitionArgs(_Args):
    definition_id: str = Field(min_length=1)


class GenerateSqlArgs(_Args):
    source_id: str = Field(min_length=1, max_length=64, description="Cataloged database source to query.")
    intent: str = Field(min_length=1, max_length=2000,
                        description="What the rows must answer, with every filter, comparison and date range.")
    evidence_ids: list[str] = Field(default_factory=list, max_length=30,
                                    description="IDs of this task's evidence describing the tables, columns, joins "
                                                "and filters to use (T, K, G and B items).")
    dataset_ids: list[str] = Field(default_factory=list, max_length=10,
                                   description="Datasets to include in the schema context (from find_datasets).")


class ExecuteSqlArgs(_Args):
    source_id: str = Field(min_length=1, max_length=64, description="Cataloged database source (source_id).")
    sql: str = Field(min_length=1, max_length=20000, description="One read-only SELECT (from generate_sql).")
    max_rows: int | None = Field(default=None, ge=1, le=10000, description="Rows to return (default 200).")


@dataclass
class AgentTool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[Any], dict[str, Any]]
    summarize: Callable[[dict[str, Any]], str]

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": inline_schema(self.args_model)}


@dataclass
class ToolResult:
    observation: dict[str, Any]
    status: str
    summary: str
    evidence_ids: list[str] = field(default_factory=list)
    compacted: int = 0          # listed items replaced by a seen_before stub
    truncated: bool = False     # the observation was shortened to fit max_observation_chars
    injection_flags: int = 0    # items whose text looks like an instruction to the model


class ToolBox:
    def __init__(self, tools: list[AgentTool], max_observation_chars: int):
        self.tools = {t.name: t for t in tools}
        self.max_chars = max_observation_chars

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str | None, arguments: dict[str, Any], seen: set[str] | frozenset = frozenset()) -> ToolResult:
        """Run one tool call. Listed items whose evidence ID is in ``seen`` (already shown in this task's
        context) are returned as ``seen_before`` without their text."""
        tool = self.tools.get(name or "")
        if tool is None:
            record_tool_call("agent", "unknown", "error", "unknown_tool")
            msg = f"unknown tool {name!r}; available tools: {', '.join(self.tools)}"
            return ToolResult({"ok": False, "error": {"code": "unknown_tool", "message": msg}}, "error", msg)
        try:
            args = tool.args_model.model_validate(arguments if isinstance(arguments, dict) else {})
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()[:5])
            record_tool_call("agent", tool.name, "error", "invalid_argument")
            return ToolResult({"ok": False, "error": {"code": "invalid_argument", "message": msg}}, "error",
                              f"invalid arguments: {msg}")
        out = call_tool(tool.name, lambda: tool.run(args), args.model_dump(exclude_none=True),
                        span_prefix="agent.tool", max_chars=10**9)
        out.pop("trace_id", None)
        if not out.get("ok"):
            return ToolResult(out, "error", f"error: {out['error'].get('message')}")
        evidence_ids = out.pop("evidence_ids", [])
        summary = tool.summarize(out)
        flags = sum(1 for text in _strings(out) if len(text) > 40 and suspicious_instruction(text))
        compacted = 0
        if seen:
            for key, val in list(out.items()):
                if isinstance(val, list):
                    items = [_compact(v, seen) for v in val]
                    compacted += sum(1 for old, new in zip(val, items) if new is not old)
                    out[key] = items
        fitted = fit_observation(out, self.max_chars)
        return ToolResult(fitted, "ok", summary, evidence_ids, compacted, "truncated" in fitted, flags)


def _compact(item: Any, seen: set[str] | frozenset) -> Any:
    """An item already shown in this task: its ID and labels only."""
    if isinstance(item, dict) and item.get("id") in seen:
        return {**{k: item[k] for k in _COMPACT_KEYS if item.get(k) is not None}, "seen_before": True}
    return item


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


def _slug(ref: str | None) -> str | None:
    return ref.rsplit("/", 1)[-1] if ref else None


def build_search_tools(mesh, ledger: EvidenceLedger, scope: SearchScope | None, max_observation_chars: int,
                       llm: Callable[[], Any] | None = None) -> ToolBox:
    scope = scope or SearchScope()

    def graph_items(items: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
        out = []
        for k in items[:limit]:
            ident = ledger.add("K", "graph:" + k["graph_ref"], "graph", k)
            view = {"id": ident, "type": k["type"], "text": k["text"], "inferred": bool(k.get("inferred"))}
            if k["type"] == "business_definition":
                view["definition_id"] = (k.get("provenance") or {}).get("definition_id")
            out.append(view)
        return out

    # -- retrieve_context ----------------------------------------------------------------------
    def retrieve(a: RetrieveArgs) -> dict[str, Any]:
        res = mesh.retrieve_many(
            a.queries, a.top_k or scope.top_k, _narrow(a.evidence_types, scope.evidence_types, "evidence types"),
            _narrow(a.document_ids, scope.document_ids, "documents"), scope.source_types,
            scope.structured_source_ids, _narrow(a.dataset_ids, scope.dataset_ids, "datasets"),
            scope.expand_parents, scope.rerank)
        docs, glossary = [], []
        for d in res["results"]:
            ident = ledger.add("D", "chunk:" + d["chunk_id"], "document_passage", d)
            docs.append({"id": ident, "title": d.get("title") or d.get("source_uri"), "section": d.get("section"),
                         "page": d.get("page"), "score": d["score"], "matched_queries": d["matched_queries"],
                         "document_id": d["document_id"], "chunk_id": d["chunk_id"], "text": d["text"]})
        for g in res["glossary_results"]:
            ident = ledger.add("G", "glossary:" + (g.get("catalog_ref") or g["graph_ref"]), "glossary", g)
            glossary.append({"id": ident, "type": g["type"], "label": g["label"], "description": g.get("description"),
                             "slug": _slug(g.get("ref")), "dataset_id": g.get("dataset_id"), "dataset": g.get("dataset"),
                             "column": g.get("column"), "catalog_ref": g.get("catalog_ref"), "score": g["score"],
                             "matched_queries": g["matched_queries"]})
        graph = graph_items(res["graph_evidence"], len(res["graph_evidence"]))
        for view, k in zip(graph, res["graph_evidence"]):
            view["matched_queries"] = k["matched_queries"]
        out: dict[str, Any] = {"queries": res["queries"], "documents": docs, "glossary": glossary, "graph": graph,
                               "evidence_ids": [x["id"] for x in docs + glossary + graph]}
        if res.get("reranking"):
            out["reranking"] = res["reranking"]
        if res.get("message"):
            out["message"] = res["message"]
        if docs:
            out["hint"] = "read_document(document_id, chunk_id) returns a passage with its surrounding context."
        return out

    # -- read_document -------------------------------------------------------------------------
    def read_document(a: ReadDocumentArgs) -> dict[str, Any]:
        r = mesh.read_document(a.document_id, a.page, a.section, a.chunk_id, a.offset, READ_CHARS)
        meta = r["metadata"]
        if "chunk" in r:
            c = r["chunk"]
            passage = {"document_id": c["document_id"], "chunk_id": c["chunk_id"], "title": c.get("title"),
                       "source_uri": c.get("source_uri"), "page": c.get("page"), "section": c.get("section"),
                       "text": c["text"], "citation": {"chunk_ref": f"/api/chunks/{c['chunk_id']}"},
                       "parents": [{"node_id": p["node_id"], "level": p["level"], "expanded": True, "text": p["text"]}
                                   for p in c.get("parents") or []]}
            ident = ledger.add("D", "chunk:" + c["chunk_id"], "document_passage", passage, replace=True)
            return {"id": ident, "document_id": c["document_id"], "chunk_id": c["chunk_id"],
                    "title": c.get("title") or c.get("source_uri"), "page": c.get("page"), "section": c.get("section"),
                    "text": c["text"], "parents": [{"level": p["level"], "text": p["text"]} for p in passage["parents"]],
                    "evidence_ids": [ident]}
        data = {"document_id": a.document_id, "title": meta.get("title"), "source_uri": meta.get("source_uri"),
                "page": r.get("page"), "section": r.get("section"), "offset": r["offset"], "text": r["text"]}
        ident = ledger.add("D", f"text:{a.document_id}:{a.page}:{a.section}:{a.offset}", "document_text", data)
        return {"id": ident, "document_id": a.document_id, "title": meta.get("title") or meta.get("source_uri"),
                "text": r["text"], "offset": r["offset"], "next_offset": r["next_offset"],
                "total_chars": r["total_chars"], "available_pages": r["available_pages"][:30],
                "available_sections": r["available_sections"][:40], "evidence_ids": [ident]}

    # -- catalog -------------------------------------------------------------------------------
    def find_datasets(a: FindDatasetsArgs) -> dict[str, Any]:
        res = mesh.catalog.find_datasets(a.name_contains, a.column_contains, a.source_id, a.limit)
        items = []
        for d in res["datasets"]:
            data = {"dataset_id": d["dataset_id"], "name": d["name"], "schema": d["schema_name"],
                    "source_id": d["source_id"], "catalog_ref": d["catalog_ref"], "row_count": d["row_count"],
                    "business_entity": d["entity_label"]}
            ident = ledger.add("T", "dataset:" + d["dataset_id"], "dataset", data)
            items.append({"id": ident, **data, "column_count": d["column_count"]})
        return {"total": res["total"], "datasets": items, "evidence_ids": [x["id"] for x in items],
                "hint": "get_dataset(dataset_id) describes columns, business terms and keys."}

    def dataset_name(dataset_id: str) -> str:
        row = mesh.db.query_one("SELECT name FROM datasets WHERE dataset_id = ?", (dataset_id,))
        return row["name"] if row else dataset_id

    def get_dataset(a: DatasetArgs) -> dict[str, Any]:
        full = mesh.get_dataset(a.dataset_id)
        columns = []
        for c in full["columns"][:60]:
            term = c.get("business_term") or {}
            col = {"name": c["name"], "data_type": c.get("data_type"), "nullable": c.get("nullable"),
                   "business_term": term.get("label"), "term_slug": term.get("slug"),
                   "term_description": term.get("description")}
            p = c.get("profile")
            if p:  # summary only: raw most-common values stay out of the model context
                col["profile"] = {"distinct": p["distinct_count"], "nulls": p["null_count"], "empty": p["empty_count"],
                                  "content_type": p["content_type"], "formats": p["format_patterns"][:2]}
            columns.append({k: v for k, v in col.items() if v is not None})
        keys = []
        for k in full["keys"]:
            key = {"type": k["key_type"], "origin": k["origin"], "columns": k["columns"]}
            if k["key_type"] == "foreign":
                key["references"] = f"{k.get('referenced_dataset')}({', '.join(k.get('referenced_columns') or [])})"
                key["referenced_dataset_id"] = k.get("referenced_dataset_id")
                if k.get("confidence"):  # inferred FKs: high / medium / low and why
                    key["confidence"] = k["confidence"]["level"]
                    key["confidence_reasons"] = k["confidence"]["reasons"][:4]
            keys.append(key)
        referenced_by = [{"type": "foreign", "origin": k["origin"], "from_dataset": dataset_name(k["dataset_id"]),
                          "from_dataset_id": k["dataset_id"], "columns": k["columns"],
                          "references": f"{full['name']}({', '.join(k.get('referenced_columns') or [])})",
                          **({"confidence": k["confidence"]["level"]} if k.get("confidence") else {})}
                         for k in full["referenced_by"]]
        entity = full.get("business_entity") or {}
        data = {"dataset_id": full["dataset_id"], "name": full["name"], "schema": full["schema"],
                "source_id": full["source_id"], "catalog_ref": full["catalog_ref"], "row_count": full["row_count"],
                "business_entity": entity.get("label"), "entity_slug": entity.get("slug"),
                "columns": columns, "columns_total": len(full["columns"]), "keys": keys,
                "referenced_by": referenced_by,
                "definitions": [{"name": d["name"], "definition_id": d["definition_id"]} for d in full["definitions"]]}
        ident = ledger.add("T", "dataset:" + full["dataset_id"], "dataset", data, replace=True)
        return {"id": ident, **data, "evidence_ids": [ident]}

    # -- business concepts ---------------------------------------------------------------------
    def get_business_term(a: SlugArgs) -> dict[str, Any]:
        t = mesh.get_business_term(a.slug)
        descriptions = t["graph"].get("descriptions") or sorted({c["term_description"] for c in t["columns"]
                                                                 if c.get("term_description")})
        data = {"slug": t["slug"], "label": t["label"], "ref": f"/api/business-terms/{t['slug']}",
                "graph_ref": t["graph_ref"], "descriptions": descriptions[:5],
                "columns": [{"catalog_ref": c["catalog_ref"], "dataset_id": c["dataset_id"], "dataset": c["dataset"],
                             "column": c["name"], "description": c.get("term_description")} for c in t["columns"][:30]],
                "definitions": [{"name": d["name"], "definition_id": d["definition_id"]} for d in t["definitions"]]}
        ident = ledger.add("B", "term:" + t["slug"], "business_term", data, replace=True)
        related = graph_items(t["graph"].get("evidence") or [])
        return {"id": ident, **data, "related": related, "evidence_ids": [ident] + [x["id"] for x in related]}

    def get_business_entity(a: SlugArgs) -> dict[str, Any]:
        e = mesh.get_business_entity(a.slug)
        data = {"slug": e["slug"], "label": e["label"], "ref": f"/api/business-entities/{e['slug']}",
                "graph_ref": e["graph_ref"], "descriptions": (e["graph"].get("descriptions") or [])[:5],
                "datasets": [{"dataset_id": d["dataset_id"], "name": d["name"], "catalog_ref": d["catalog_ref"],
                              "source_id": d["source_id"]} for d in e["datasets"][:30]],
                "definitions": [{"name": d["name"], "definition_id": d["definition_id"]} for d in e["definitions"]],
                "document_mentions": len(e["document_mentions"])}
        ident = ledger.add("B", "entity:" + e["slug"], "business_entity", data, replace=True)
        related = graph_items(e["graph"].get("evidence") or [])
        return {"id": ident, **data, "related": related, "evidence_ids": [ident] + [x["id"] for x in related]}

    def get_business_definition(a: DefinitionArgs) -> dict[str, Any]:
        d = mesh.definitions.get(a.definition_id)
        ident = ledger.add("B", "definition:" + d["definition_id"], "business_definition", d, replace=True)
        return {"id": ident, "definition_id": d["definition_id"], "name": d["name"], "definition": d["definition"],
                "author": d.get("author"), "entities": [x["label"] for x in d["entities"]],
                "terms": [x["label"] for x in d["terms"]], "evidence_ids": [ident]}

    # -- knowledge graph -----------------------------------------------------------------------
    def query_knowledge_graph(a: KGQueryRequest) -> dict[str, Any]:
        res = mesh.query_knowledge_graph(a, scope.structured_source_ids, scope.dataset_ids, scope.document_ids)
        if res["status"] == "error":
            return {"status": "error", "intent_id": res.get("intent_id"), "error": res.get("error"),
                    "warnings": res.get("warnings", []), "evidence_ids": []}
        ids: list[str] = []
        joins: dict[str, str] = {}  # local join id -> evidence id
        out_joins = []
        for j in res["physical_mappings"]["joins"]:
            pairs = ", ".join(f"{p['from_column']}->{p['to_column']}" for p in j["column_pairs"])
            text = (f"join {j['join_predicate']} ({j['from_table']} -> {j['to_table']}; column pairs in order: "
                    f"{pairs}) implements relationship '{j['relationship']}' [{j['method']} mapping, "
                    f"{j['origin']} foreign key, mapping status {j['status']}"
                    + (f", confidence {j['confidence']}" if j.get("confidence") else "") + "]")
            ident = ledger.add("K", f"graph:join:{j['relationship_iri']}|{j['foreign_key_iri']}", "graph", {
                "type": "join_mapping", "label": j["join_predicate"], "text": text,
                "graph_ref": j["foreign_key_iri"], "source_ref": None, "inferred": j["status"] != "verified",
                "provenance": {"relationship": j["relationship_iri"], "foreign_key": j["foreign_key_iri"]}})
            joins[j["join_id"]] = ident
            ids.append(ident)
            out_joins.append({"id": ident, "predicate": j["join_predicate"], "column_pairs": pairs,
                              "from_table": j["from_table"], "to_table": j["to_table"],
                              "relationship": j["relationship"], "method": j["method"], "origin": j["origin"],
                              "status": j["status"], "confidence": j.get("confidence")})
        rels = {r["iri"]: r for r in res.get("relationships", [])}
        out_paths = []
        for pth in res["paths"]:
            key = "graph:path:" + "|".join(f"{e['relationship_iri']}>{e['direction']}" for e in pth["edges"])
            text = (f"path {pth['label']} ({pth['hops']} hop(s); relevance score {pth['relevance_score']}, "
                    f"uncalibrated; status {pth['validation_status']}; "
                    + ("executable: every hop has a join mapping" if pth["executable"] else
                       "NOT executable: missing join mapping for " + ", ".join(
                           rels.get(m, {}).get("name", m) for m in pth["missing_mappings"]))
                    + "). Schema-level connectivity only, not a record-level fact.")
            ident = ledger.add("K", key, "graph", {"type": "graph_path", "label": pth["label"], "text": text,
                                                  "graph_ref": pth["nodes"][0], "source_ref": None,
                                                  "inferred": pth["validation_status"] != "verified"})
            ids.append(ident)
            out_paths.append({"id": ident, "path": pth["label"], "score": pth["relevance_score"],
                              "executable": pth["executable"], "status": pth["validation_status"],
                              "joins": [joins[r] for e in pth["edges"] for r in e["join_refs"] if r in joins],
                              "relationships": [{"name": e["predicate"], "kind": e["kind"], "status": e["status"],
                                                 "definition": rels.get(e["relationship_iri"], {}).get("definition")}
                                                for e in pth["edges"]]})
        tables = {t["iri"]: t for t in res["physical_mappings"]["tables"]}
        out_entities = []
        for e in res["business_entities"]:
            slug = e["iri"].rsplit(":", 1)[-1]
            data = {"slug": slug, "label": e["name"], "ref": f"/api/business-entities/{slug}", "graph_ref": e["iri"],
                    "descriptions": [e["definition"]] if e.get("definition") else [],
                    "datasets": [{"catalog_ref": tables[t["iri"]]["catalog_ref"], "dataset_id": tables[t["iri"]]["dataset_id"],
                                  "name": t["name"]} for t in e.get("tables", []) if t["iri"] in tables]}
            ident = ledger.add("B", "entity:" + slug, "business_entity", data)
            ids.append(ident)
            out_entities.append({"id": ident, "name": e["name"], "iri": e["iri"], "definition": e.get("definition"),
                                 "alt_names": e.get("alt_names") or None, "match": e.get("match_type"),
                                 "tables": [f"{t['name']} ({tables[t['iri']]['dataset_id']}, {t['mapping_status']})"
                                            for t in e.get("tables", []) if t["iri"] in tables]})
        out_terms = []
        for t in res["business_terms"]:
            slug = t["iri"].rsplit(":", 1)[-1]
            data = {"slug": slug, "label": t["name"], "ref": f"/api/business-terms/{slug}", "graph_ref": t["iri"],
                    "descriptions": [t["definition"]] if t.get("definition") else [],
                    "columns": [{"catalog_ref": c.get("catalog_ref"), "dataset_id": c.get("dataset_id"), "description": None}
                                for c in t.get("columns", []) if c.get("catalog_ref")]}
            ident = ledger.add("B", "term:" + slug, "business_term", data)
            ids.append(ident)
            out_terms.append({"id": ident, "name": t["name"], "iri": t["iri"], "definition": t.get("definition"),
                              "synonyms": t.get("synonyms") or None, "match": t.get("match_type"),
                              "columns": [f"{c['table']}.{c['name']} ({c.get('data_type') or '?'}, {c['mapping_status']})"
                                          for c in t.get("columns", [])],
                              "related_terms": [f"{r['predicate']} {r['term']} ({r['status']})" if r["direction"] == "outgoing"
                                                else f"{r['term']} {r['predicate']} this ({r['status']})"
                                                for r in t.get("related_terms", [])] or None})
        constraints = []
        for c in res["constraints"]:
            if c["type"] == "term_relation":
                text = (f"{c['subject']} {c['predicate']} {c['object']}"
                        + (f" — {c['definition']}" if c.get("definition") else "")
                        + f" (descriptive; not validated against records; status {c['status']})")
                ident = ledger.add("K", "graph:" + c["iri"], "graph", {
                    "type": "term_relation", "label": f"{c['subject']} {c['predicate']} {c['object']}", "text": text,
                    "graph_ref": c["iri"], "source_ref": None, "inferred": c["status"] != "verified"})
                ids.append(ident)
                constraints.append({"id": ident, "relation": text})
            else:
                constraints.append({"key": f"{c['type'].replace('_', ' ')} {c['table']}({', '.join(c['columns'])})",
                                    "origin": c["origin"], "status": c["status"]})
        out = {"status": res["status"], "intent_id": res.get("intent_id"), "matched_queries": res["matched_queries"],
               "entities": out_entities, "terms": out_terms, "paths": out_paths, "joins": out_joins,
               "constraints": constraints, "warnings": res["warnings"], "note": RECORD_NOTE,
               "graph_version": res["graph"]["version"], "evidence_ids": ids}
        if res.get("ambiguities"):
            out["ambiguities"] = res["ambiguities"]
        if res.get("truncated"):
            out["truncated"] = res["truncated"]
        if res["status"] == "no_match":
            out["hint"] = ("Nothing in the knowledge graph matches these queries. Try other terms from the user's "
                           "request, retrieve_context, or find_datasets.")
        return out

    # -- SQL -------------------------------------------------------------------------------------
    def schema_context(a: GenerateSqlArgs) -> tuple[list[str], list[str]]:
        """Tables/columns/keys and join/graph facts from the cited evidence and the listed datasets."""
        tables: list[str] = []
        facts: list[str] = []
        seen_ds: set[str] = set()

        def add_dataset(d: dict[str, Any]) -> None:
            if d["dataset_id"] in seen_ds:
                return
            seen_ds.add(d["dataset_id"])
            if d.get("source_id") and d["source_id"] != a.source_id:
                return
            name = ".".join(x for x in (d.get("schema"), d["name"]) if x)
            cols = "; ".join(f"{c['name']} {c.get('data_type') or ''}".strip()
                             + (f" (term: {c['business_term']})" if c.get("business_term") else "")
                             for c in d.get("columns", []))
            keys = "; ".join(f"{k['type']} key ({', '.join(k['columns'])})"
                             + (f" -> {k['references']}" if k.get("references") else "") + f" [{k['origin']}]"
                             for k in d.get("keys", []))
            tables.append(f"table {name} (dataset {d['dataset_id']}): columns: {cols}" + (f"; keys: {keys}" if keys else ""))

        for ident in a.evidence_ids:
            item = ledger.item(ident)
            if item is None:
                raise InvalidArgument(f"unknown evidence id {ident}")
            d = item.data
            if item.kind == "dataset":
                if "columns" not in d:
                    d = _dataset_data(d["dataset_id"])
                add_dataset(d)
            elif item.kind == "graph":
                facts.append(f"[{ident}] {d.get('text')}")
            elif item.kind == "glossary" and d.get("dataset_id"):
                facts.append(f"[{ident}] {d.get('type')} '{d.get('label')}' is {d.get('dataset')}"
                             + (f".{d['column']}" if d.get("column") else ""))
                add_dataset(_dataset_data(d["dataset_id"]))
            elif item.kind in ("business_term", "business_entity"):
                facts.append(f"[{ident}] {item.kind.replace('_', ' ')} '{d.get('label')}'"
                             + (f": {'; '.join(d['descriptions'][:2])}" if d.get("descriptions") else ""))
                for ds in d.get("datasets") or []:
                    if ds.get("dataset_id"):
                        add_dataset(_dataset_data(ds["dataset_id"]))
                for c in d.get("columns") or []:
                    if c.get("dataset_id"):
                        add_dataset(_dataset_data(c["dataset_id"]))
        for ds in a.dataset_ids:
            add_dataset(_dataset_data(ds))
        return tables, facts

    def _dataset_data(dataset_id: str) -> dict[str, Any]:
        full = mesh.get_dataset(dataset_id)
        return {"dataset_id": full["dataset_id"], "name": full["name"], "schema": full["schema"],
                "source_id": full["source_id"],
                "columns": [{"name": c["name"], "data_type": c.get("data_type"),
                             "business_term": (c.get("business_term") or {}).get("label")} for c in full["columns"][:80]],
                "keys": [{"type": k["key_type"], "origin": k["origin"], "columns": k["columns"],
                          "references": f"{k.get('referenced_dataset')}({', '.join(k.get('referenced_columns') or [])})"
                          if k["key_type"] == "foreign" else None} for k in full["keys"]
                         if not (k["key_type"] == "unique" and k["origin"] == "inferred"
                                 and not k.get("primary_key_candidate"))]}

    def generate_sql(a: GenerateSqlArgs) -> dict[str, Any]:
        from ..structured.sql_exec import SqlRejected

        if llm is None:
            raise NotConfigured("no agent model is available for SQL generation")
        path = mesh.config.agent.sql_generation_prompt_path
        try:
            system = open(path, encoding="utf-8").read().strip()
        except OSError as exc:
            raise NotConfigured(f"SQL generation prompt not readable: {path}") from exc
        try:
            dialect = mesh.sql.dialect(a.source_id)
        except SqlRejected as exc:
            return {"status": "error", "error": {"code": exc.code, "message": exc.message}}
        tables, facts = schema_context(a)
        if not tables:
            raise InvalidArgument("no table of this source is described by the given evidence_ids or dataset_ids; "
                                  "get_dataset or query_knowledge_graph first")
        prompt = (f"Target source: {a.source_id}\nSQL dialect: {dialect}\n\nIntent:\n{a.intent}\n\n"
                  "Tables (only these exist; use their exact names):\n" + "\n".join(tables)
                  + "\n\nJoins and business facts from the knowledge graph and catalog:\n"
                  + ("\n".join(facts) or "(none)") + "\n\nReturn the JSON object.")
        resp = llm().generate(prompt, system=system, purpose="sql_generation", json_output=True,
                              prompt_version=prompt_version(system))
        try:
            reply = parse_json_object(resp.text)
        except ValueError as exc:
            return {"status": "error", "error": {"code": "invalid_reply", "message": str(exc)[:200]}}
        sql = str(reply.get("sql") or "").strip().rstrip(";").strip()
        if not sql:
            return {"status": "cannot_answer", "explanation": str(reply.get("explanation") or "")[:1000],
                    "hint": "The model found no way to answer this intent from the given tables."}
        try:
            checked = mesh.sql.validate(a.source_id, sql, scope.structured_source_ids, scope.dataset_ids)
        except SqlRejected as exc:
            return {"status": "invalid", "sql": sql, "dialect": dialect,
                    "validation": {"ok": False, "code": exc.code, "message": exc.message},
                    "hint": "Fix the problem (call generate_sql again with more evidence) before execute_sql."}
        return {"status": "ok", "sql": sql, "dialect": dialect, "source_id": a.source_id,
                "tables": [t["catalog_ref"] for t in checked["tables"]],
                "explanation": str(reply.get("explanation") or "")[:1000], "validation": {"ok": True},
                "hint": "Pass this sql and source_id to execute_sql."}

    def execute_sql(a: ExecuteSqlArgs) -> dict[str, Any]:
        from ..observability.audit import RequestContext

        res = mesh.execute_sql(a.source_id, a.sql, a.max_rows, scope.structured_source_ids, scope.dataset_ids,
                               RequestContext(caller=scope.caller or "agent"))
        if res["status"] != "ok":
            return {"status": "error", "source_id": a.source_id, "error": res["error"], "evidence_ids": []}
        data = {k: res[k] for k in ("source_id", "sql", "columns", "rows", "row_count", "truncated", "dialect")}
        data["tables"] = [t["catalog_ref"] for t in res["tables"]]
        ident = ledger.add("R", f"sql:{a.source_id}:{res['sql']}:{a.max_rows}", "query_result", data, replace=True)
        out = {"id": ident, "status": "ok", "source_id": a.source_id, "columns": [c["name"] for c in res["columns"]],
               "rows": res["rows"], "row_count": res["row_count"], "truncated": res["truncated"],
               "tables": data["tables"], "evidence_ids": [ident]}
        if not res["rows"]:
            out["note"] = "The query ran and returned no rows."
        return out

    # -- HTML ------------------------------------------------------------------------------------
    def generate_html(a: HtmlRequest) -> dict[str, Any]:
        req = a.model_copy(deep=True)
        for b in req.blocks:
            if b.type == "table" and b.result_ref:
                item = ledger.item(b.result_ref)
                if item is None or item.kind != "query_result":
                    raise InvalidArgument(f"result_ref {b.result_ref} is not an execute_sql result of this search")
                b.columns = [c["name"] for c in item.data["columns"]]
                b.rows = item.data["rows"]
                flagged = [c["name"] for c in item.data["columns"] if c.get("sensitive")]
                b.sensitive_columns = list(dict.fromkeys(flagged + b.sensitive_columns))  # never unmark
        res = mesh.generate_html(req)
        ident = ledger.add("H", f"html:{len(ledger)}", "html_fragment",
                           {"html": res["html"], "title": req.title, "blocks_rendered": res["blocks_rendered"],
                            "truncated": res["truncated"]})
        return {"id": ident, "status": res["status"], "blocks_rendered": res["blocks_rendered"],
                "truncated": res["truncated"], "chars": len(res["html"]),
                "note": "The fragment will be shown on the web page; it is not repeated here.", "evidence_ids": [ident]}

    tools = [
        AgentTool("retrieve_context",
                  "Semantic search over indexed document passages, the business glossary (business terms bound to "
                  "table columns, business entities bound to tables) and knowledge-graph facts (captured business "
                  "definitions, entity relationships, key constraints). Takes a list of queries, searched in "
                  "parallel and merged: each result appears once with the queries that matched it. Start here for "
                  "definitions, policies and to find which tables or columns hold a business concept.",
                  RetrieveArgs, retrieve,
                  lambda o: f"{len(o['queries'])} queries: {len(o['documents'])} passages, "
                            f"{len(o['glossary'])} glossary matches, {len(o['graph'])} graph facts"),
        AgentTool("read_document",
                  "Read more of a document: a retrieved chunk with its surrounding parent context (document_id + "
                  "chunk_id), one page, one section, or the next part from an offset.",
                  ReadDocumentArgs, read_document,
                  lambda o: f"{len(o.get('text') or '')} characters of {o.get('title') or o.get('document_id')}"),
        AgentTool("find_datasets",
                  "List cataloged tables/datasets whose name, schema or business entity contains name_contains, "
                  "and/or that have a column whose name or business term contains column_contains. Returns "
                  "dataset IDs for get_dataset.",
                  FindDatasetsArgs, find_datasets,
                  lambda o: f"{len(o['datasets'])} of {o['total']} datasets"),
        AgentTool("get_dataset",
                  "Describe one table: columns with data types, business terms and profile summary, primary/unique "
                  "keys and foreign keys (declared, or inferred by profiling), and the tables that reference it.",
                  DatasetArgs, get_dataset,
                  lambda o: f"{o['name']}: {o['columns_total']} columns, {len(o['keys'])} keys, "
                            f"referenced by {len(o['referenced_by'])}"),
        AgentTool("get_business_term",
                  "A business term by slug (from a glossary result): its description, the physical columns it is "
                  "bound to, and linked definitions and graph facts.",
                  SlugArgs, get_business_term,
                  lambda o: f"term {o['label']}: {len(o['columns'])} columns"),
        AgentTool("get_business_entity",
                  "A business entity by slug: the tables that represent it, its definitions, relationships and key "
                  "constraints.",
                  SlugArgs, get_business_entity,
                  lambda o: f"entity {o['label']}: {len(o['datasets'])} tables"),
        AgentTool("get_business_definition",
                  "A captured business definition by definition_id (from a graph fact or another result): its full "
                  "text, author and the entities/terms it describes.",
                  DefinitionArgs, get_business_definition,
                  lambda o: f"definition {o['name']}"),
    ]
    if mesh.config.knowledge_graph.enabled:
        tools.append(AgentTool(
            "query_knowledge_graph",
            "Input: the intent (intent_id, intent text, retrieval_queries; max_hops, top_k). Purpose: resolve "
            "business entities, business terms (with synonyms) and named relationships in the knowledge graph, find "
            "ranked multi-hop relationship paths between the entities (e.g. Customer -> Order -> Order Line -> "
            "Product) and expand them with the mapped tables, columns, ordered join column pairs, keys and term "
            "relations. Output: status (ok / no_match / error), entities, terms, paths, joins, constraints and "
            "warnings with evidence IDs. Results describe how data can be joined, never facts about records.",
            KGQueryRequest, query_knowledge_graph,
            lambda o: (f"{o['status']}: {len(o.get('entities', []))} entities, {len(o.get('terms', []))} terms, "
                       f"{len(o.get('paths', []))} paths, {len(o.get('joins', []))} joins")))
    if mesh.config.sql_execution.enabled:
        tools += [
            AgentTool(
                "generate_sql",
                "Input: source_id, the intent (with every filter, comparison and date range) and evidence_ids of "
                "this task's tables, joins and terms (or dataset_ids). Purpose: an LLM writes ONE read-only SELECT "
                "for that source's SQL dialect using only that evidence; the query is checked (read-only, cataloged "
                "tables) but not run. Output: status (ok / invalid / cannot_answer / error), sql, dialect, tables.",
                GenerateSqlArgs, generate_sql,
                lambda o: f"generate_sql {o.get('status')}"),
            AgentTool(
                "execute_sql",
                "Input: source_id (a cataloged database) and one read-only SELECT (from generate_sql); optional "
                "max_rows. Purpose: retrieve database records with the source's read-only connection. Output: "
                "status ok with columns, rows (empty list when nothing matches), row_count and truncated, plus an "
                "R evidence ID to cite for record-level facts; or status error with a safe message.",
                ExecuteSqlArgs, execute_sql,
                lambda o: (f"{o['row_count']} rows" if o.get("status") == "ok"
                           else f"error: {(o.get('error') or {}).get('message')}")),
        ]
    tools.append(AgentTool(
        "generate_html",
        "Input: title and content blocks (heading, paragraph, list, citations, table with columns/rows or "
        "result_ref = an execute_sql evidence ID such as R1). Purpose: build an HTML fragment for display on the web "
        "page when the user asks for HTML / web output; values are escaped. Output: an H evidence ID (the fragment "
        "itself is shown to the user, not returned here).",
        HtmlRequest, generate_html,
        lambda o: f"HTML fragment {o.get('id')} ({o.get('chars')} chars)"))
    return ToolBox(tools, max_observation_chars)

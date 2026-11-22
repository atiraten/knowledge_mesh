"""Evidence markers on MCP tool results.

Every item a caller can cite carries ``evidence_ref`` (a stable key: the same chunk, glossary entry, graph fact,
table or concept always has the same ref), ``evidence_kind`` (``document_passage``, ``document_text``,
``glossary``, ``graph``, ``dataset``, ``business_term``, ``business_entity``, ``business_definition``,
``query_result``, ``html_fragment``) and, for a richer version of an item another tool may already have
returned, ``evidence_detail: true``. The search agent harness numbers these items for citations (D, G, K, T, B,
R, H); other MCP clients can ignore the fields. Raw profiled values are never included.

``retrieve_documents`` and ``query_knowledge_graph`` take ``view``: ``full`` (default) returns the documented
output with the markers; ``agent`` (sent by the harness) returns the compact shape shown to the executor, with each
item's citation data in ``evidence_data``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .graph.retriever import RECORD_NOTE

READ_CHARS = 4000


def mark(item: dict[str, Any], ref: str, kind: str, detail: bool = False,
         data: dict[str, Any] | None = None) -> dict[str, Any]:
    """``data``: the item's citation data (``evidence_data``), when the item shown is a compact view of it."""
    out = {**item, "evidence_ref": ref, "evidence_kind": kind}
    if detail:
        out["evidence_detail"] = True
    if data is not None:
        out["evidence_data"] = data
    return out


def graph_fact(k: dict[str, Any]) -> dict[str, Any]:
    return mark(k, "graph:" + k["graph_ref"], "graph")


# -- retrieve_documents -------------------------------------------------------------------------------
def retrieval(res: dict[str, Any], view: str = "full") -> dict[str, Any]:
    """``full``: the retrieval result with evidence markers. ``agent``: the compact shape shown to the search
    agent's executor (no parent text), each item's full data in ``evidence_data``."""
    docs = [(d, "chunk:" + d["chunk_id"]) for d in res["results"]]
    glossary = [(g, "glossary:" + (g.get("catalog_ref") or g["graph_ref"])) for g in res["glossary_results"]]
    graph = [(k, "graph:" + k["graph_ref"]) for k in res["graph_evidence"]]
    if view != "agent":
        return {**res, "results": [mark(d, ref, "document_passage") for d, ref in docs],
                "glossary_results": [mark(g, ref, "glossary") for g, ref in glossary],
                "graph_evidence": [mark(k, ref, "graph") for k, ref in graph]}
    out: dict[str, Any] = {
        "queries": res["queries"],
        "documents": [mark({"title": d.get("title") or d.get("source_uri"), "section": d.get("section"),
                            "page": d.get("page"), "score": d["score"], "matched_queries": d["matched_queries"],
                            "document_id": d["document_id"], "chunk_id": d["chunk_id"], "text": d["text"]},
                           ref, "document_passage", data=d) for d, ref in docs],
        "glossary": [mark({"type": g["type"], "label": g["label"], "description": g.get("description"),
                           "slug": g["ref"].rsplit("/", 1)[-1] if g.get("ref") else None,
                           "dataset_id": g.get("dataset_id"), "dataset": g.get("dataset"), "column": g.get("column"),
                           "catalog_ref": g.get("catalog_ref"), "score": g["score"],
                           "matched_queries": g["matched_queries"]}, ref, "glossary", data=g) for g, ref in glossary],
        "graph": [mark({"type": k["type"], "text": k["text"], "inferred": bool(k.get("inferred")),
                        **({"definition_id": (k.get("provenance") or {}).get("definition_id")}
                           if k["type"] == "business_definition" else {}),
                        "matched_queries": k["matched_queries"]}, ref, "graph", data=k) for k, ref in graph]}
    if res.get("reranking"):
        out["reranking"] = [{"query": r["query"], "outcome": r.get("outcome"), "message": r.get("message")}
                            for r in res["reranking"]]
    if res.get("message"):
        out["message"] = res["message"]
    if docs:
        out["hint"] = "read_document(document_id, chunk_id) returns a passage with its surrounding context."
    return out


# -- read_document ------------------------------------------------------------------------------------
def read_document(r: dict[str, Any], document_id: str, page: int | None, section: str | None,
                  offset: int) -> dict[str, Any]:
    """The read_document result as before, marked; its citation data in ``evidence_data``."""
    meta = r["metadata"]
    if "chunk" in r:
        c = r["chunk"]
        passage = {"document_id": c["document_id"], "chunk_id": c["chunk_id"], "title": c.get("title"),
                   "source_uri": c.get("source_uri"), "page": c.get("page"), "section": c.get("section"),
                   "text": c["text"], "citation": {"chunk_ref": f"/api/chunks/{c['chunk_id']}"},
                   "parents": [{"node_id": p["node_id"], "level": p["level"], "expanded": True, "text": p["text"]}
                               for p in c.get("parents") or []]}
        return mark(r, "chunk:" + c["chunk_id"], "document_passage", detail=True, data=passage)
    data = {"document_id": document_id, "title": meta.get("title"), "source_uri": meta.get("source_uri"),
            "page": r.get("page"), "section": r.get("section"), "offset": r["offset"], "text": r["text"]}
    return mark(r, f"text:{document_id}:{page}:{section}:{offset}", "document_text", data=data)


# -- catalog ------------------------------------------------------------------------------------------
def find_datasets(res: dict[str, Any]) -> dict[str, Any]:
    items = []
    for d in res["datasets"]:
        data = {"dataset_id": d["dataset_id"], "name": d["name"], "schema": d["schema_name"],
                "source_id": d["source_id"], "catalog_ref": d["catalog_ref"], "row_count": d["row_count"],
                "business_entity": d["entity_label"], "column_count": d["column_count"]}
        items.append(mark(data, "dataset:" + d["dataset_id"], "dataset"))
    return {"total": res["total"], "datasets": items,
            "hint": "get_dataset(dataset_id) describes columns, business terms and keys."}


def get_dataset(full: dict[str, Any], dataset_name) -> dict[str, Any]:
    columns = []
    for c in full["columns"][:60]:
        term = c.get("business_term") or {}
        col = {"name": c["name"], "data_type": c.get("data_type"), "nullable": c.get("nullable"),
               "business_term": term.get("label"), "term_slug": term.get("slug"),
               "term_description": term.get("description")}
        p = c.get("profile")
        if p:  # summary only: raw most-common values stay out
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
            "definitions": [{"name": d["name"], "definition_id": d["definition_id"]}
                            for d in full.get("definitions") or []]}
    return mark(data, "dataset:" + full["dataset_id"], "dataset", detail=True)


# -- business concepts --------------------------------------------------------------------------------
def business_term(t: dict[str, Any]) -> dict[str, Any]:
    descriptions = t["graph"].get("descriptions") or sorted({c["term_description"] for c in t["columns"]
                                                             if c.get("term_description")})
    data = {"slug": t["slug"], "label": t["label"], "ref": f"/api/business-terms/{t['slug']}",
            "graph_ref": t["graph_ref"], "descriptions": descriptions[:5],
            "columns": [{"catalog_ref": c["catalog_ref"], "dataset_id": c["dataset_id"], "dataset": c["dataset"],
                         "column": c["name"], "description": c.get("term_description")} for c in t["columns"][:30]],
            "definitions": [{"name": d["name"], "definition_id": d["definition_id"]}
                            for d in t.get("definitions") or []]}
    return {**mark(data, "term:" + t["slug"], "business_term", detail=True),
            "related": [graph_fact(k) for k in (t["graph"].get("evidence") or [])[:10]]}


def business_entity(e: dict[str, Any]) -> dict[str, Any]:
    data = {"slug": e["slug"], "label": e["label"], "ref": f"/api/business-entities/{e['slug']}",
            "graph_ref": e["graph_ref"], "description": e.get("description"),
            "descriptions": (e["graph"].get("descriptions") or [])[:5],
            "datasets": [{"dataset_id": d["dataset_id"], "name": d["name"], "catalog_ref": d["catalog_ref"],
                          "source_id": d["source_id"]} for d in e["datasets"][:30]],
            "document_mentions": len(e["document_mentions"])}
    return {**mark(data, "entity:" + e["slug"], "business_entity", detail=True),
            "related": [graph_fact(k) for k in (e["graph"].get("evidence") or [])[:10]]}


def business_definition(d: dict[str, Any]) -> dict[str, Any]:
    return mark(d, "definition:" + d["definition_id"], "business_definition", detail=True)


# -- knowledge graph ----------------------------------------------------------------------------------
def knowledge_graph(res: dict[str, Any], view: str = "full") -> dict[str, Any]:
    """``full``: the documented output, with evidence markers on entities, terms, joins, paths and term relations.
    ``agent``: the compact shape shown to the search agent's executor, each item's citation data in
    ``evidence_data`` (the ledger renders joins, paths and relations from their ``text``)."""
    if res.get("status") == "error":
        if view == "agent":
            return {"status": "error", "intent_id": res.get("intent_id"), "error": res.get("error"),
                    "warnings": res.get("warnings", [])}
        return res
    agent = view == "agent"
    pm = res.get("physical_mappings") or {}
    tables = {t["iri"]: t for t in pm.get("tables", [])}
    rels = {r["iri"]: r for r in res.get("relationships", [])}
    joins = []
    for j in pm.get("joins", []):
        ref = f"graph:join:{j['relationship_iri']}|{j['foreign_key_iri']}"
        if not agent:
            joins.append(mark(j, ref, "graph"))
            continue
        pairs = ", ".join(f"{p['from_column']}->{p['to_column']}" for p in j["column_pairs"])
        text = (f"join {j['join_predicate']} ({j['from_table']} -> {j['to_table']}; column pairs in order: "
                f"{pairs}) implements relationship '{j['relationship']}' [{j['method']} mapping, "
                f"{j['origin']} foreign key, mapping status {j['status']}"
                + (f", confidence {j['confidence']}" if j.get("confidence") else "") + "]")
        joins.append(mark({"join_id": j["join_id"], "predicate": j["join_predicate"], "column_pairs": pairs,
                           "from_table": j["from_table"], "to_table": j["to_table"], "relationship": j["relationship"],
                           "method": j["method"], "origin": j["origin"], "status": j["status"],
                           "confidence": j.get("confidence")}, ref, "graph",
                          data={"type": "join_mapping", "label": j["join_predicate"], "text": text,
                                "graph_ref": j["foreign_key_iri"], "source_ref": None,
                                "inferred": j["status"] != "verified",
                                "provenance": {"relationship": j["relationship_iri"],
                                               "foreign_key": j["foreign_key_iri"]}}))
    paths = []
    for pth in res.get("paths", []):
        ref = "graph:path:" + "|".join(f"{e['relationship_iri']}>{e['direction']}" for e in pth["edges"])
        if not agent:
            paths.append(mark(pth, ref, "graph"))
            continue
        text = (f"path {pth['label']} ({pth['hops']} hop(s); relevance score {pth['relevance_score']}, "
                f"uncalibrated; status {pth['validation_status']}; "
                + ("executable: every hop has a join mapping" if pth["executable"] else
                   "NOT executable: missing join mapping for " + ", ".join(
                       rels.get(m, {}).get("name", m) for m in pth["missing_mappings"]))
                + "). Schema-level connectivity only, not a record-level fact.")
        paths.append(mark({"path": pth["label"], "score": pth["relevance_score"], "executable": pth["executable"],
                           "status": pth["validation_status"],
                           "joins": [r for e in pth["edges"] for r in e["join_refs"]],
                           "relationships": [{"name": e["predicate"], "kind": e["kind"], "status": e["status"],
                                              "definition": rels.get(e["relationship_iri"], {}).get("definition")}
                                             for e in pth["edges"]]}, ref, "graph",
                          data={"type": "graph_path", "label": pth["label"], "text": text, "graph_ref": pth["nodes"][0],
                                "source_ref": None, "inferred": pth["validation_status"] != "verified"}))
    entities = []
    for e in res.get("business_entities", []):
        slug = e["iri"].rsplit(":", 1)[-1]
        if not agent:
            entities.append(mark(e, "entity:" + slug, "business_entity"))
            continue
        mapped = [t for t in e.get("tables", []) if t["iri"] in tables]
        entities.append(mark({"name": e["name"], "iri": e["iri"], "definition": e.get("definition"),
                              "alt_names": e.get("alt_names") or None, "match": e.get("match_type"),
                              "tables": [f"{t['name']} ({tables[t['iri']]['dataset_id']}, {t['mapping_status']})"
                                         for t in mapped]}, "entity:" + slug, "business_entity",
                             data={"slug": slug, "label": e["name"], "ref": f"/api/business-entities/{slug}",
                                   "graph_ref": e["iri"],
                                   "descriptions": [e["definition"]] if e.get("definition") else [],
                                   "datasets": [{"catalog_ref": tables[t["iri"]]["catalog_ref"],
                                                 "dataset_id": tables[t["iri"]]["dataset_id"], "name": t["name"]}
                                                for t in mapped]}))
    terms = []
    for t in res.get("business_terms", []):
        slug = t["iri"].rsplit(":", 1)[-1]
        if not agent:
            terms.append(mark(t, "term:" + slug, "business_term"))
            continue
        terms.append(mark({"name": t["name"], "iri": t["iri"], "definition": t.get("definition"),
                           "synonyms": t.get("synonyms") or None, "match": t.get("match_type"),
                           "columns": [f"{c['table']}.{c['name']} ({c.get('data_type') or '?'}, {c['mapping_status']})"
                                       for c in t.get("columns", [])],
                           "related_terms": [f"{r['predicate']} {r['term']} ({r['status']})"
                                             if r["direction"] == "outgoing"
                                             else f"{r['term']} {r['predicate']} this ({r['status']})"
                                             for r in t.get("related_terms", [])] or None},
                          "term:" + slug, "business_term",
                          data={"slug": slug, "label": t["name"], "ref": f"/api/business-terms/{slug}",
                                "graph_ref": t["iri"], "descriptions": [t["definition"]] if t.get("definition") else [],
                                "columns": [{"catalog_ref": c.get("catalog_ref"), "dataset_id": c.get("dataset_id"),
                                             "description": None}
                                            for c in t.get("columns", []) if c.get("catalog_ref")]}))
    constraints = []
    for c in res.get("constraints", []):
        if c["type"] != "term_relation":
            constraints.append({"key": f"{c['type'].replace('_', ' ')} {c['table']}({', '.join(c['columns'])})",
                                "origin": c["origin"], "status": c["status"]} if agent else c)
        elif not agent:
            constraints.append(mark(c, "graph:" + c["iri"], "graph"))
        else:
            text = (f"{c['subject']} {c['predicate']} {c['object']}"
                    + (f" — {c['definition']}" if c.get("definition") else "")
                    + f" (descriptive; not validated against records; status {c['status']})")
            constraints.append(mark({"relation": text}, "graph:" + c["iri"], "graph",
                                    data={"type": "term_relation",
                                          "label": f"{c['subject']} {c['predicate']} {c['object']}", "text": text, "graph_ref": c["iri"], "source_ref": None,
                                          "inferred": c["status"] != "verified"}))
    if not agent:
        out = {**res, "business_entities": entities, "business_terms": terms, "paths": paths,
               "constraints": constraints, "note": RECORD_NOTE}
        if pm:
            out["physical_mappings"] = {**pm, "joins": joins}
        return out
    # ranked_paths, not paths: the published output schema describes full paths
    out = {"status": res["status"], "intent_id": res.get("intent_id"), "matched_queries": res["matched_queries"],
           "entities": entities, "terms": terms, "ranked_paths": paths, "joins": joins, "constraints": constraints,
           "warnings": res["warnings"], "note": RECORD_NOTE, "graph_version": res["graph"]["version"]}
    if res.get("ambiguities"):
        out["ambiguities"] = res["ambiguities"]
    if res.get("truncated"):
        out["truncated"] = res["truncated"]
    if res["status"] == "no_match":
        out["hint"] = ("Nothing in the knowledge graph matches these queries. Try other terms from the user's "
                       "request, retrieve_documents, or find_datasets.")
    return out


# -- SQL and HTML -------------------------------------------------------------------------------------
def query_result(res: dict[str, Any], max_rows: int | None) -> dict[str, Any]:
    if res.get("status") != "ok":
        return res
    out = mark(res, f"sql:{res['source_id']}:{res['sql']}:{max_rows}", "query_result", detail=True)
    if not res["rows"]:
        out["note"] = "The query ran and returned no rows."
    return out


def html_fragment(res: dict[str, Any], title: str | None) -> dict[str, Any]:
    ref = "html:" + hashlib.sha256(res["html"].encode("utf-8")).hexdigest()[:16]
    return mark({**res, "title": title}, ref, "html_fragment")

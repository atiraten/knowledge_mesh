"""Knowledge graph view of a catalog selection (a database = source, a schema, or a table) for the UI.

Returns cytoscape-ready ``nodes`` and ``edges``: the selected tables with their columns and keys (declared keys,
primary-key candidates and foreign keys; not every inferred unique column combination), the
business entities the tables represent, the business terms of the columns, the relationships between
those entities and the term relations between those terms, each with its status. Bounded by
``knowledge_graph.view_max_nodes``.
"""

from __future__ import annotations

from typing import Any

from ..common.errors import InvalidArgument, NotFound
from .retriever import build_snapshot
from .store import KnowledgeGraph


def graph_view(graph: KnowledgeGraph, max_nodes: int, source_id: str, schema: str | None = None,
               dataset_id: str | None = None) -> dict[str, Any]:
    if not source_id:
        raise InvalidArgument("source_id is required")
    snap = graph.cached("kg_snapshot", lambda: build_snapshot(graph))
    tables = {iri: t for iri, t in snap.tables.items() if t["source_id"] == source_id
              and (dataset_id is None or t["dataset_id"] == dataset_id)
              and (schema is None or (t["schema"] or "") == schema)}
    if not tables:
        raise NotFound("no knowledge graph data for this selection (catalog the source first)")
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    truncated = False

    def node(iri: str, label: str | None, kind: str, **extra) -> bool:
        nonlocal truncated
        if iri in nodes:
            return True
        if len(nodes) >= max_nodes:
            truncated = True
            return False
        nodes[iri] = {"id": iri, "label": label or iri.rsplit(":", 1)[-1], "type": kind, **extra}
        return True

    def edge(source: str, target: str, label: str, kind: str, key: str | None = None, **extra) -> None:
        if source in nodes and target in nodes:
            ident = key or f"{source}|{kind}|{target}"
            edges.setdefault(ident, {"id": ident, "source": source, "target": target, "label": label,
                                     "type": kind, **extra})

    # Priority order, so that a node budget cuts profiling detail first: tables and their entities, columns and
    # their terms, primary and foreign keys, then unique keys.
    for iri, t in sorted(tables.items(), key=lambda kv: kv[1]["name"] or ""):
        node(iri, t["name"], "table", dataset_id=t["dataset_id"], schema=t["schema"], ref=f"#dataset/{t['dataset_id']}")
    entity_iris: set[str] = set()
    for iri, t in tables.items():
        for e, status in t["entities"].items():
            ent = snap.entities.get(e)
            if ent and node(e, ent["name"], "entity", ref=f"#entity/{e.rsplit(':', 1)[-1]}"):
                entity_iris.add(e)
                edge(iri, e, "represents", "represents", status=status)
    term_iris: set[str] = set()
    for c_iri, c in sorted(snap.columns.items(), key=lambda kv: (kv[1]["table"] or "", kv[1]["name"] or "")):
        if c["table"] not in tables or not node(c_iri, c["name"], "column", data_type=c["data_type"]):
            continue
        edge(c["table"], c_iri, "has column", "has_column")
        for t, status in c["terms"].items():
            term = snap.terms.get(t)
            if term and node(t, term["name"], "term", ref=f"#term/{t.rsplit(':', 1)[-1]}",
                             synonyms=term["alt_names"]):
                term_iris.add(t)
                edge(c_iri, t, "business term", "has_term", status=status)
    rank = {"primary": 0, "foreign": 1, "unique": 2}
    for k_iri, k in sorted(snap.keys.items(), key=lambda kv: (rank[kv[1]["type"]], kv[0])):
        if k["table"] not in tables:
            continue
        if k["type"] == "unique" and k["origin"] != "declared" and not k["pk_candidate"]:
            continue  # inferred unique column combinations are profiling detail, not part of the model view
        cols = [snap.columns.get(c, {}).get("name") for c in k["columns"]]
        label = f"{k['type'].upper()} ({', '.join(c for c in cols if c)})"
        if not node(k_iri, label, f"{k['type']}_key", origin=k["origin"], status=k["status"],
                    confidence=k["confidence"]):
            continue
        edge(k["table"], k_iri, "key", "has_key")
        if k["type"] == "foreign" and k["references"]:
            ref = snap.tables.get(k["references"])
            if ref and node(k["references"], ref["name"], "table", dataset_id=ref["dataset_id"],
                            schema=ref["schema"], ref=f"#dataset/{ref['dataset_id']}"):
                pairs = ", ".join(f"{snap.columns.get(a, {}).get('name')}→{snap.columns.get(b, {}).get('name')}"
                                  for a, b in k["pairs"])
                edge(k_iri, k["references"], pairs, "references", status=k["status"], origin=k["origin"])
    for e in snap.edges.values():
        if e.source in entity_iris and e.target in entity_iris:
            edge(e.source, e.target, e.name, f"{e.kind}_relationship", key=e.iri, status=e.status,
                 definition=e.definition)
    for tr in snap.term_relations.values():
        if tr["subject_iri"] in term_iris and tr["object_iri"] in term_iris:
            edge(tr["subject_iri"], tr["object_iri"], tr["predicate"], "term_relation", key=tr["iri"],
                 status=tr["status"])
    return {"scope": {"source_id": source_id, "schema": schema, "dataset_id": dataset_id},
            "nodes": list(nodes.values()), "edges": list(edges.values()), "truncated": truncated,
            "graph": {"version": snap.version, "published_at": snap.published_at}}

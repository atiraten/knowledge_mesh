"""Business glossary vector index (separate persistent SimpleVectorStore).

One node per business term (per column) and per business entity (per dataset). Each node's
ref doc is its dataset, so a dataset refresh replaces exactly its glossary vectors and a
retired dataset removes them; other datasets are untouched. Unchanged content is not re-embedded.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from ..indexing.vector_index import PersistentVectorIndex
from ..storage.db import Database

_HIDDEN = ["record_type", "label", "slug", "dataset_id", "column_id", "source_id", "catalog_ref", "graph_ref",
           "dataset", "column"]


class GlossaryService:
    def __init__(self, db: Database, index: PersistentVectorIndex):
        self.db = db
        self.index = index

    def nodes_for(self, dataset_id: str) -> list[TextNode]:
        d = self.db.query_one("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        if d is None or not d["active"]:
            return []
        nodes: list[TextNode] = []
        source = {NodeRelationship.SOURCE: RelatedNodeInfo(node_id=dataset_id)}
        if d.get("entity_label"):
            nodes.append(TextNode(
                id_=f"glossary-entity-{dataset_id}",
                text=f"{d['entity_label']}. Business entity for dataset {d['name']}.",
                metadata={"record_type": "business_entity", "label": d["entity_label"], "slug": d["entity_slug"],
                          "dataset_id": dataset_id, "column_id": "", "source_id": d["source_id"],
                          "catalog_ref": d["catalog_ref"], "graph_ref": f"urn:mesh:entity:{d['entity_slug']}",
                          "dataset": d["name"], "column": ""},
                excluded_embed_metadata_keys=list(_HIDDEN), excluded_llm_metadata_keys=list(_HIDDEN),
                relationships=dict(source),
            ))
        for c in self.db.query("SELECT * FROM columns WHERE dataset_id = ? AND active = 1 AND term_status = 'generated' "
                               "ORDER BY ordinal", (dataset_id,)):
            text = f"{c['term_label']}."
            if c.get("term_description"):
                text += f" {c['term_description']}"
            text += f" (column {c['name']} of {d['name']})"
            nodes.append(TextNode(
                id_=f"glossary-term-{c['column_id']}", text=text,
                metadata={"record_type": "business_term", "label": c["term_label"], "slug": c["term_slug"],
                          "dataset_id": dataset_id, "column_id": c["column_id"], "source_id": d["source_id"],
                          "catalog_ref": c["catalog_ref"], "graph_ref": f"urn:mesh:term:{c['term_slug']}",
                          "dataset": d["name"], "column": c["name"]},
                excluded_embed_metadata_keys=list(_HIDDEN), excluded_llm_metadata_keys=list(_HIDDEN),
                relationships=dict(source),
            ))
        return nodes

    def upsert_dataset(self, dataset_id: str) -> dict[str, Any]:
        nodes = self.nodes_for(dataset_id)
        if not nodes:
            removed = self.index.delete_ref_doc(dataset_id)
            return {"dataset_id": dataset_id, "nodes": 0, "status": "removed" if removed else "empty"}
        existing = set(self.index.node_ids_for(dataset_id))
        if existing == {n.node_id for n in nodes}:
            stored = {i: self.index.get_node(i) for i in existing}
            if all(stored[n.node_id] is not None and stored[n.node_id].hash == n.hash for n in nodes):
                return {"dataset_id": dataset_id, "nodes": len(nodes), "status": "unchanged"}
        version = self.index.replace_ref_doc(dataset_id, nodes)
        return {"dataset_id": dataset_id, "nodes": len(nodes), "status": "indexed", "index_version": version}

    def sync_source(self, source_id: str) -> dict[str, int]:
        counts = {"indexed": 0, "unchanged": 0, "removed": 0, "empty": 0}
        for d in self.db.query("SELECT dataset_id FROM datasets WHERE source_id = ?", (source_id,)):
            counts[self.upsert_dataset(d["dataset_id"])["status"]] += 1
        return counts

    def search(self, query: str, top_k: int, record_types: list[str] | None = None,
               source_ids: list[str] | None = None, dataset_ids: list[str] | None = None):
        filters: dict[str, Any] = {}
        if record_types:
            filters["record_type"] = record_types
        if source_ids:
            filters["source_id"] = source_ids
        if dataset_ids:
            filters["dataset_id"] = dataset_ids
        return self.index.retrieve(query, top_k, filters)

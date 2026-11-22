"""Document indexing pipeline: chunk -> embed leaves -> store parents -> extract entities -> graph."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..common.errors import GraphValidationError, MeshError
from ..graph.builders import build_document_context
from ..graph.ontology import context_uri
from ..graph.store import KnowledgeGraph
from ..indexing.vector_index import PersistentVectorIndex
from ..observability import get_metrics, span
from ..storage.db import Database
from ..structured.glossary_store import GlossaryStore
from .chunking import HierarchicalChunker
from .documents import DocumentService
from .entities import EntityExtractor, Mention, Relationship

log = logging.getLogger(__name__)


class IndexingService:
    def __init__(self, db: Database, documents: DocumentService, index: PersistentVectorIndex,
                 chunk_settings, extractor: EntityExtractor, graph: KnowledgeGraph):
        self.db = db
        self.documents = documents
        self.index = index
        self.chunk_settings = chunk_settings
        self.extractor = extractor
        self.graph = graph

    def index_document(self, document_id: str, chunk_sizes: list[int] | None = None,
                       force: bool = False) -> dict[str, Any]:
        metrics = get_metrics()
        row = self.documents.get_row(document_id)
        if not row["active"] or row["status"] in {"deleted"}:
            return {"document_id": document_id, "status": "skipped", "reason": "document deleted"}
        if row["status"] in {"skipped", "failed"}:
            self.retire_index(document_id)
            return {"document_id": document_id, "status": "skipped", "reason": row["status_reason"] or row["status"]}
        if (row["status"] == "indexed" and row["indexed_content_hash"] == row["content_hash"]
                and not force and chunk_sizes is None):
            return {"document_id": document_id, "status": "skipped", "reason": "already indexed (unchanged)",
                    "index_version": row["index_version"]}
        sizes = chunk_sizes or self.chunk_settings.chunk_sizes_tokens
        timings: dict[str, float] = {}
        with span("document.index", "CHAIN", {"document.id": document_id, "chunking.sizes": sizes}) as sp:
            t0 = time.perf_counter()
            extracted = self.documents.load_extracted(document_id)
            chunker = HierarchicalChunker(sizes, self.chunk_settings.chunk_overlap_tokens)
            result = chunker.chunk(document_id, row["content_hash"], extracted, {
                "source_uri": row["source_uri"], "source_type": row["source_type"],
                "content_hash": row["content_hash"],
            })
            timings["chunking"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            with span("document.embed_and_store", "EMBEDDING",
                      {"embedding.model_name": self.index.meta.get("embedding_model") or "",
                       "embedding.leaf_count": len(result.leaf_nodes)}):
                version = self.index.replace_ref_doc(document_id, result.leaf_nodes, result.parent_nodes)
            timings["embedding_and_storage"] = time.perf_counter() - t1
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.executemany(
                    "INSERT INTO chunks(chunk_id, document_id, parent_id, level, ordinal, is_leaf, page, section,"
                    " char_count) VALUES (?,?,?,?,?,?,?,?,?)",
                    [(r["chunk_id"], document_id, r["parent_id"], r["level"], r["ordinal"], int(r["is_leaf"]),
                      r["page"], r["section"], r["char_count"]) for r in result.records],
                )
            t2 = time.perf_counter()
            graph_status = self._publish_entities(document_id, row, result)
            timings["entity_extraction_and_graph"] = time.perf_counter() - t2
            self.documents.mark_indexed(document_id, len(result.all_nodes), len(result.leaf_nodes), version)
            for stage, secs in timings.items():
                metrics.stage_duration.record(secs, {"stage": stage})
            metrics.chunks_indexed.add(len(result.leaf_nodes))
            metrics.nodes_stored.add(len(result.all_nodes))
            sp.set_attribute("index.version", version)
            sp.set_attribute("index.leaf_nodes", len(result.leaf_nodes))
            sp.set_attribute("index.parent_nodes", len(result.parent_nodes))
        return {
            "document_id": document_id, "status": "indexed", "index_version": version,
            "nodes": len(result.all_nodes), "leaf_chunks": len(result.leaf_nodes),
            "parent_nodes": len(result.parent_nodes), "graph": graph_status,
            "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
        }

    def _publish_entities(self, document_id: str, row: dict[str, Any], result) -> dict[str, Any]:
        chunks = [(n.node_id, n.get_content()) for n in result.leaf_nodes]
        try:
            mentions, rels = self.extractor.extract(document_id, chunks)
        except Exception as exc:
            log.warning("entity extraction failed", extra={"document_id": document_id, "error": type(exc).__name__})
            return {"status": "failed", "reason": f"entity extraction failed: {type(exc).__name__}"}
        chunk_info = {r["chunk_id"]: r for r in result.records}
        with self.db.transaction() as conn:
            # glossary slugs and labels (renamed entities keep their slug); deleted entities are dropped
            mentions, rels = GlossaryStore.resolve_document_entities(conn, mentions, rels)
            conn.execute("DELETE FROM doc_entity_mentions WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM doc_relationships WHERE document_id = ?", (document_id,))
            conn.executemany(
                "INSERT OR IGNORE INTO doc_entity_mentions(document_id, chunk_id, entity_slug, entity_label)"
                " VALUES (?,?,?,?)", [(document_id, m.chunk_id, m.slug, m.label) for m in mentions])
            conn.executemany(
                "INSERT OR REPLACE INTO doc_relationships(relationship_id, document_id, chunk_id, source_slug,"
                " source_label, rel_type, target_slug, target_label, evidence, method) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(r.relationship_id, document_id, r.chunk_id, r.source_slug, r.source_label, r.rel_type,
                  r.target_slug, r.target_label, r.evidence, r.method) for r in rels])
        graph = build_document_context(row, chunk_info, mentions, rels)
        try:
            self.graph.publish(context_uri("document", document_id), graph, f"document {document_id}")
        except GraphValidationError as exc:
            return {"status": "rejected", "violations": len(exc.details.get("violations", []))}
        return {"status": "published", "entities": len({m.slug for m in mentions}), "relationships": len(rels)}

    def document_graph(self, document_id: str):
        """A document's graph context rebuilt from its stored mentions and relationships (no re-extraction),
        with the current glossary labels (None: unknown document). Used after a business entity is renamed
        or deleted."""
        row = self.db.query_one("SELECT * FROM documents WHERE document_id = ?", (document_id,))
        if row is None:
            return None
        chunks = {r["chunk_id"]: r for r in self.db.query(
            "SELECT chunk_id, page, section FROM chunks WHERE document_id = ?", (document_id,))}
        mentions = [Mention(r["chunk_id"], r["label"], r["entity_slug"]) for r in self.db.query(
            "SELECT m.chunk_id, m.entity_slug, COALESCE(e.label, m.entity_label) AS label FROM doc_entity_mentions m"
            " LEFT JOIN business_entities e ON e.slug = m.entity_slug WHERE m.document_id = ?", (document_id,))]
        rels = [Relationship(r["relationship_id"], r["chunk_id"], r["source_label"], r["source_slug"], r["rel_type"],
                             r["target_label"], r["target_slug"], r["evidence"] or "", r["method"])
                for r in self.db.query("SELECT * FROM doc_relationships WHERE document_id = ?", (document_id,))]
        return build_document_context(row, chunks, mentions, rels)

    def retire_index(self, document_id: str) -> bool:
        """Remove a document's chunks from the index, chunk table and graph (other sources untouched)."""
        removed = self.index.delete_ref_doc(document_id)
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM doc_entity_mentions WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM doc_relationships WHERE document_id = ?", (document_id,))
            conn.execute("UPDATE documents SET node_count = 0, leaf_count = 0, indexed_content_hash = NULL "
                         "WHERE document_id = ?", (document_id,))
        try:
            self.graph.remove(context_uri("document", document_id))
        except MeshError:  # removal never fails validation, but stay defensive
            log.exception("graph context removal failed")
        return removed

    def delete_document(self, document_id: str) -> dict[str, Any]:
        self.documents.get_row(document_id)
        self.retire_index(document_id)
        self.documents.mark_deleted(document_id)
        return {"document_id": document_id, "status": "deleted"}

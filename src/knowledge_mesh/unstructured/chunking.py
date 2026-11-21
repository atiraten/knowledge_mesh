"""Hierarchical chunking with LlamaIndex (independent of MCP/HTTP transport).

Each extracted section (a PDF page, a heading section, or a table) is parsed with
``HierarchicalNodeParser`` using the configured sizes (default 2048 -> 1024 -> 256 tokens).
Node IDs are then rewritten deterministically (``node-*`` for parents, ``chunk-*`` for leaves)
from the document ID, content hash and position, so re-indexing unchanged content yields the
same IDs and citations stay stable. Every node's SOURCE relationship points at the mesh
``document_id`` so all nodes of a document can be replaced or deleted together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo

from ..common.ids import chunk_id_for
from .loaders import ExtractedDocument

# Metadata kept on nodes but hidden from the embedding text and LLM rendering.
_HIDDEN_KEYS = ["document_id", "chunk_id", "parent_id", "level", "source_uri", "source_type",
                "content_hash", "page", "is_leaf"]


@dataclass
class ChunkResult:
    leaf_nodes: list[BaseNode]
    parent_nodes: list[BaseNode]
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_nodes(self) -> list[BaseNode]:
        return self.parent_nodes + self.leaf_nodes


class HierarchicalChunker:
    def __init__(self, chunk_sizes: list[int], chunk_overlap: int = 20):
        if len(chunk_sizes) < 1:
            raise ValueError("at least one chunk size is required")
        self.chunk_sizes = list(chunk_sizes)
        self.chunk_overlap = min(chunk_overlap, max(0, self.chunk_sizes[-1] // 4))

    def chunk(self, document_id: str, content_hash: str, extracted: ExtractedDocument,
              base_metadata: dict[str, Any]) -> ChunkResult:
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes, chunk_overlap=self.chunk_overlap, include_prev_next_rel=False
        )
        docs = []
        for i, section in enumerate(extracted.sections):
            meta = {
                **base_metadata,
                "document_id": document_id,
                "title": extracted.title or "",
                "section": section.heading or "",
                "page": section.page if section.page is not None else -1,
            }
            docs.append(Document(
                text=section.text, id_=f"{document_id}::section-{i}", metadata=meta,
                excluded_embed_metadata_keys=list(_HIDDEN_KEYS),
                excluded_llm_metadata_keys=list(_HIDDEN_KEYS),
            ))
        nodes = parser.get_nodes_from_documents(docs) if docs else []
        return self._assign_ids(document_id, content_hash, nodes)

    def _assign_ids(self, document_id: str, content_hash: str, nodes: list[BaseNode]) -> ChunkResult:
        by_id = {n.node_id: n for n in nodes}
        leaf_ids = {n.node_id for n in get_leaf_nodes(nodes)}

        def level_of(node: BaseNode) -> int:
            level = 0
            parent = node.relationships.get(NodeRelationship.PARENT)
            while parent is not None and parent.node_id in by_id:
                level += 1
                parent = by_id[parent.node_id].relationships.get(NodeRelationship.PARENT)
            return level

        counters: dict[int, int] = {}
        mapping: dict[str, str] = {}
        levels: dict[str, int] = {}
        for node in nodes:  # parser output order is deterministic for identical input
            level = level_of(node)
            ordinal = counters.get(level, 0)
            counters[level] = ordinal + 1
            mapping[node.node_id] = chunk_id_for(document_id, content_hash, level, ordinal, node.node_id in leaf_ids)
            levels[node.node_id] = level

        leaves, parents, records = [], [], []
        ordinals: dict[int, int] = {}
        for node in nodes:
            old_id = node.node_id
            is_leaf = old_id in leaf_ids
            new_id = mapping[old_id]
            rels = {}
            for rel, info in node.relationships.items():
                if rel == NodeRelationship.SOURCE:
                    rels[rel] = RelatedNodeInfo(node_id=document_id)
                elif isinstance(info, list):
                    rels[rel] = [RelatedNodeInfo(node_id=mapping.get(i.node_id, i.node_id)) for i in info]
                else:
                    rels[rel] = RelatedNodeInfo(node_id=mapping.get(info.node_id, info.node_id))
            if NodeRelationship.SOURCE not in rels:
                rels[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=document_id)
            node.id_ = new_id
            node.relationships = rels
            parent = rels.get(NodeRelationship.PARENT)
            parent_id = parent.node_id if parent is not None else None
            node.metadata["chunk_id"] = new_id
            node.metadata["parent_id"] = parent_id or ""
            node.metadata["level"] = levels[old_id]
            node.metadata["is_leaf"] = is_leaf
            for key in _HIDDEN_KEYS:
                if key not in node.excluded_embed_metadata_keys:
                    node.excluded_embed_metadata_keys.append(key)
                if key not in node.excluded_llm_metadata_keys:
                    node.excluded_llm_metadata_keys.append(key)
            level = levels[old_id]
            ordinal = ordinals.get(level, 0)
            ordinals[level] = ordinal + 1
            records.append({
                "chunk_id": new_id, "parent_id": parent_id, "level": level, "ordinal": ordinal,
                "is_leaf": is_leaf, "page": node.metadata.get("page") if node.metadata.get("page", -1) != -1 else None,
                "section": node.metadata.get("section") or None, "char_count": len(node.get_content()),
            })
            (leaves if is_leaf else parents).append(node)
        return ChunkResult(leaf_nodes=leaves, parent_nodes=parents, records=records)

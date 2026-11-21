"""Persistent LlamaIndex ``SimpleVectorStore`` index with docstore-backed node hierarchy.

Two instances exist: one for document passages and one for the business glossary. Each
records the embedding model, revision and dimension it was built with; opening it with an
incompatible model raises ``IncompatibleIndex`` (reindex or use a separate persist dir).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Sequence

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.simple import SimpleVectorStore

from ..common.errors import IncompatibleIndex
from ..common.ids import utcnow

log = logging.getLogger(__name__)
META_FILE = "mesh_index_meta.json"


class PersistentVectorIndex:
    def __init__(self, name: str, persist_dir: Path, embed_model, embedding_info: dict[str, Any] | None = None):
        self.name = name
        self.persist_dir = Path(persist_dir)
        self.embed_model = embed_model
        self._embedding_info = embedding_info
        self._lock = threading.RLock()
        self._index: VectorStoreIndex | None = None
        self.meta: dict[str, Any] = {}

    # -- lifecycle -----------------------------------------------------------------
    def _ensure(self) -> VectorStoreIndex:
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = self._open()
        return self._index

    def embedding_info(self) -> dict[str, Any]:
        if self._embedding_info is None:
            self._embedding_info = self.embed_model.info()
        return self._embedding_info

    def _open(self) -> VectorStoreIndex:
        meta_path = self.persist_dir / META_FILE
        if (self.persist_dir / "docstore.json").exists():
            self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            stored_model = self.meta.get("embedding_model")
            info = self.embedding_info()
            if stored_model and (stored_model != info["model"] or self.meta.get("dimension") != info["dimension"]):
                raise IncompatibleIndex(
                    f"{self.name} index was built with {stored_model} ({self.meta.get('dimension')} dims); "
                    f"configured model is {info['model']} ({info['dimension']} dims). Reindex or use a new persist dir.",
                    {"index": self.name, "persist_dir": str(self.persist_dir)},
                )
            storage = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            index = load_index_from_storage(storage, embed_model=self.embed_model)
            log.info("opened vector index", extra={"index": self.name, "nodes": len(storage.docstore.docs)})
            return index
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        storage = StorageContext.from_defaults(
            docstore=SimpleDocumentStore(), index_store=SimpleIndexStore(), vector_store=SimpleVectorStore()
        )
        index = VectorStoreIndex(nodes=[], storage_context=storage, embed_model=self.embed_model)
        info = self.embedding_info()
        self.meta = {
            "index_name": self.name,
            "embedding_provider": info["provider"],
            "embedding_model": info["model"],
            "embedding_revision": info.get("revision"),
            "dimension": info["dimension"],
            "normalize": info.get("normalize"),
            "created_at": utcnow(),
            "version": 0,
            "score_type": "cosine_similarity",
        }
        self._index = index
        self._persist()
        return index

    def _persist(self) -> None:
        assert self._index is not None
        self._index.storage_context.persist(persist_dir=str(self.persist_dir))
        self.meta["updated_at"] = utcnow()
        tmp = self.persist_dir / (META_FILE + ".tmp")
        tmp.write_text(json.dumps(self.meta, indent=2))
        os.replace(tmp, self.persist_dir / META_FILE)

    @property
    def version(self) -> str:
        self._ensure()
        return f"{self.name}-v{self.meta.get('version', 0)}"

    # -- writes ----------------------------------------------------------------------
    def replace_ref_doc(self, ref_doc_id: str, embed_nodes: Sequence[BaseNode],
                        store_only_nodes: Sequence[BaseNode] = ()) -> str:
        """Atomically (per process) replace every node of ``ref_doc_id``.

        ``embed_nodes`` are embedded into the vector store; ``store_only_nodes`` (parents)
        go to the docstore only so the hierarchy stays reconstructible.
        """
        index = self._ensure()
        with self._lock:
            self._delete_ref_doc(ref_doc_id)
            if embed_nodes:
                index.insert_nodes(list(embed_nodes))
            if store_only_nodes:
                index.storage_context.docstore.add_documents(list(store_only_nodes), allow_update=True)
            self.meta["version"] = int(self.meta.get("version", 0)) + 1
            self._persist()
        return self.version

    def delete_ref_doc(self, ref_doc_id: str) -> bool:
        index = self._ensure()
        with self._lock:
            removed = self._delete_ref_doc(ref_doc_id)
            if removed:
                self.meta["version"] = int(self.meta.get("version", 0)) + 1
                self._persist()
            return removed

    def _delete_ref_doc(self, ref_doc_id: str) -> bool:
        # Not VectorStoreIndex.delete_ref_doc: that assumes every docstore node of the ref doc is
        # embedded, but parent nodes live only in the docstore and would raise KeyError.
        index = self._ensure()
        storage = index.storage_context
        info = storage.docstore.get_ref_doc_info(ref_doc_id)
        if info is None:
            return False
        index.vector_store.delete(ref_doc_id)
        for node_id in list(info.node_ids):
            index.index_struct.nodes_dict.pop(node_id, None)
        storage.index_store.add_index_struct(index.index_struct)
        storage.docstore.delete_ref_doc(ref_doc_id, raise_error=False)
        return True

    # -- reads -----------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[NodeWithScore]:
        index = self._ensure()
        meta_filters = None
        if filters:
            items = []
            for key, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple, set)):
                    items.append(MetadataFilter(key=key, value=list(value), operator=FilterOperator.IN))
                else:
                    items.append(MetadataFilter(key=key, value=value, operator=FilterOperator.EQ))
            meta_filters = MetadataFilters(filters=items) if items else None
        if not index.index_struct.nodes_dict:
            return []
        retriever = index.as_retriever(similarity_top_k=top_k, filters=meta_filters)
        return retriever.retrieve(query)

    def get_node(self, node_id: str) -> BaseNode | None:
        index = self._ensure()
        return index.storage_context.docstore.get_node(node_id, raise_error=False)

    def ancestors(self, node_id: str, limit: int = 5) -> list[BaseNode]:
        """Parent chain of a node, nearest first (e.g. 1024-token parent, then 2048-token parent)."""
        out: list[BaseNode] = []
        node = self.get_node(node_id)
        pid = (node.metadata.get("parent_id") or None) if node is not None else None
        while pid and len(out) < limit:
            parent = self.get_node(pid)
            if parent is None:
                break
            out.append(parent)
            pid = parent.metadata.get("parent_id") or None
        return out

    def node_ids_for(self, ref_doc_id: str) -> list[str]:
        info = self._ensure().storage_context.docstore.get_ref_doc_info(ref_doc_id)
        return list(info.node_ids) if info else []

    def stats(self) -> dict[str, Any]:
        index = self._ensure()
        return {
            "index": self.name,
            "version": self.version,
            "stored_nodes": len(index.storage_context.docstore.docs),
            "embedded_nodes": len(index.index_struct.nodes_dict),
            "embedding_model": self.meta.get("embedding_model"),
            "embedding_revision": self.meta.get("embedding_revision"),
            "dimension": self.meta.get("dimension"),
            "persist_dir": str(self.persist_dir),
        }

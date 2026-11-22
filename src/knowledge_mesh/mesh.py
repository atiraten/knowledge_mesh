"""Knowledge Mesh orchestrator.

Wires all services from ``MeshConfig`` and exposes the tool-level operations used by both
the MCP server and the FastAPI app. Every method is directly callable from Python.
Long-running operations return a job reference (``job_id`` + paginated manifest); passing
``wait_seconds`` blocks up to that long and returns the finished job summary instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from rdflib import URIRef

from .agent import AgentUnavailable, McpToolBox, McpToolClient, SearchHarness, SearchScope
from .common.errors import Conflict, GraphValidationError, InvalidArgument, MeshError, NotConfigured, NotFound, \
    UnsupportedMediaType
from .common.ids import new_id, slugify, utcnow
from .common.paths import resolve_authorized_path, safe_filename
from .config import MeshConfig
from .definitions import DefinitionService
from .graph.builders import build_glossary_context, build_source_layers
from .structured.fk_rules import FKRules
from .structured.glossary_store import GlossaryStore
from .structured.relationships import RelationshipGenerationService
from .structured.reviews import ReviewStore
from .graph.ontology import context_uri, entity_uri, term_uri
from .graph.retriever import KGQueryRequest, KnowledgeGraphRetriever
from .graph.store import KnowledgeGraph
from .graph.view import graph_view
from .common.html_render import HtmlRequest, render_html
from .structured.sql_exec import SqlExecutor
from .structured.sql_generation import SqlGenerationRequest, SqlGenerator
from .indexing.embeddings import build_embedding
from .indexing.vector_index import META_FILE, PersistentVectorIndex
from .jobs import JobContext, JobRunner, JobStore
from .llm.client import build_llm
from .observability import current_trace_id, get_metrics, sanitize_params, setup_telemetry, span
from .observability.audit import AuditLog, RequestContext
from .observability.phoenix import PhoenixClient
from .observability.quality import SearchJudge, evaluations_for
from .retrieval.answer import AnswerService
from .retrieval.rerank import LayaReranker
from .retrieval.service import EVIDENCE_TYPES, RetrievalService
from .storage.db import Database, loads
from .structured.catalog import CatalogService
from .structured.connectors import build_connector
from .structured.glossary import GlossaryService
from .common.secrets import build_secret_store
from .structured.key_discovery import KeyDiscoveryService
from .structured.connectors.database import DB_TYPES
from .structured.sources import SourceRegistry, describe_roots
from .structured.terms import TermGenerationService
from .unstructured.crawler import WebCrawler
from .unstructured.documents import DocumentService, IngestResult
from .unstructured.entities import EntityExtractor
from .unstructured.indexing import IndexingService
from .unstructured.loaders import EXT_BY_MIME

log = logging.getLogger(__name__)
MAX_READ_CHARS = 20_000


def run_sync(coro):
    """Run a coroutine from sync code, even when called inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class _LazyLLM:
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._client = None

    def generate(self, *args, **kwargs):
        if self._client is None:
            self._client = self._factory()
        return self._client.generate(*args, **kwargs)


class KnowledgeMesh:
    def __init__(self, config: MeshConfig, *, embed_model=None, term_llm=None, answer_llm=None, agent_llm=None,
                 connector_factory: Callable | None = None, http_transport=None, resolver=None,
                 secret_store=None, reranker_agent=None, setup_observability: bool = True, phoenix_transport=None):
        self.config = config
        if setup_observability:
            setup_telemetry(config.observability)
        self.db = Database(config.storage.metadata_db)
        self.job_store = JobStore(self.db)
        recovered = self.job_store.recover_interrupted()
        if recovered:
            log.warning("marked interrupted jobs as failed", extra={"jobs": recovered})
        self.runner = JobRunner(self.job_store, config.jobs.workers)
        self.embed_model = embed_model or build_embedding(config.embedding)
        self.doc_index = PersistentVectorIndex("documents", config.vector_store.document_persist_dir, self.embed_model)
        self.glossary_index = PersistentVectorIndex("glossary", config.vector_store.glossary_persist_dir,
                                                    self.embed_model)
        self.graph = KnowledgeGraph(config.knowledge_graph.data_path, config.knowledge_graph.shapes_path)
        self.documents = DocumentService(config.storage, config.crawler, self.db)
        self.secrets = secret_store or build_secret_store(config.secrets)
        self.sources = SourceRegistry(config, self.db, self.secrets, connector_factory or build_connector)
        self.catalog = CatalogService(config, self.db, self.sources)
        self.glossary = GlossaryService(self.db, self.glossary_index)
        self.glossary_store = GlossaryStore(self.db)
        filled = self.glossary_store.fill_if_empty()  # one-time: business glossary from the per-column copies
        if filled["terms"] or filled["entities"]:
            log.info("business glossary filled from the catalog", extra=filled)
            self._publish_glossary()

        term_factory = None
        if term_llm is not None:
            term_factory = lambda: term_llm  # noqa: E731
        elif config.term_generation.configured:
            term_factory = lambda: build_llm(config.term_generation)  # noqa: E731
        self.terms = TermGenerationService(config.term_generation, self.db, self.catalog, self.glossary, term_factory)
        self.keys = KeyDiscoveryService(config.profiling, self.db, self.catalog)
        self.relationships = RelationshipGenerationService(config.relationship_generation, config.term_generation,
                                                           self.db, term_factory,
                                                           FKRules.from_settings(config.profiling))
        self.reviews = ReviewStore(self.db)
        extractor_llm = _LazyLLM(term_factory) if (config.entity_extraction.method == "llm" and term_factory) else None
        self.extractor = EntityExtractor(config.entity_extraction, self.catalog.entity_labels, extractor_llm)
        self.indexing = IndexingService(self.db, self.documents, self.doc_index, config.chunking, self.extractor,
                                        self.graph)
        self.crawler = WebCrawler(config.crawler, transport=http_transport, resolver=resolver)
        self.definitions = DefinitionService(self.db, self.graph)
        self.reranker = LayaReranker(config.reranker,
                                     (lambda: reranker_agent) if reranker_agent is not None else None)
        self.retrieval = RetrievalService(config.retrieval, self.db, self.doc_index, self.glossary, self.graph,
                                          self.reranker)
        self.kg = KnowledgeGraphRetriever(config.knowledge_graph, self.graph, self.embed_model,
                                          lambda q, k, sources, datasets: self.retrieval._glossary(q, k, sources,
                                                                                                    datasets))
        self.sql = SqlExecutor(config.sql_execution, self.db, self.catalog)
        answer_factory = None
        if answer_llm is not None:
            answer_factory = lambda: answer_llm  # noqa: E731
        elif config.answer_generation.enabled:
            answer_factory = lambda: build_llm(config.answer_generation)  # noqa: E731
        self.answers = AnswerService(config.answer_generation, self.retrieval, answer_factory)
        agent_factory = None
        if agent_llm is not None:
            agent_factory = lambda: agent_llm  # noqa: E731
        elif config.agent.enabled:
            agent_factory = lambda: build_llm(config.agent, prompt_version="agent")  # noqa: E731
        self.sql_generator = SqlGenerator(config.agent.sql_generation_prompt_path, self.sql, agent_factory)
        # the harness uses MCP tools only: in-process MCP session, or the /mcp endpoint (config ``mcp``)
        self.mcp_tools = McpToolClient(config, lambda: self)
        self.harness = SearchHarness(
            config.agent, agent_factory,
            lambda ledger, scope: McpToolBox(self.mcp_tools, config, ledger, scope))
        # observability: audit trail, Phoenix annotations, online LLM judge, infrastructure gauges
        self.audit = AuditLog(self.db)
        self.phoenix = PhoenixClient(config.observability.phoenix_url, config.observability.service_name,
                                     transport=phoenix_transport)
        self.judge = SearchJudge(config.observability, agent_factory, self.db, self.phoenix)
        ref = weakref.ref(self)
        get_metrics().register_mesh_gauges(ref)

    # -- jobs ------------------------------------------------------------------------------------
    def _submit(self, kind: str, params: dict[str, Any], fn: Callable[[JobContext], dict | None],
                wait_seconds: float = 0) -> dict[str, Any]:
        job_id = self.runner.submit(kind, params, fn)
        if wait_seconds and wait_seconds > 0:
            try:
                return self._job_summary(self.runner.wait(job_id, timeout=min(wait_seconds, 3600)))
            except TimeoutError:
                pass
        return self._job_summary(self.job_store.get(job_id))

    @staticmethod
    def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
        return {k: job[k] for k in ("job_id", "kind", "status", "counts", "result", "error", "trace_id",
                                    "item_count", "manifest_ref")} | {"job_ref": f"/api/jobs/{job['job_id']}"}

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self.job_store.get(job_id)
        # link to the job's trace in Phoenix (model calls with their prompts and responses), as for searches
        job["observability"] = {"trace_url": self.phoenix.trace_url(job.get("trace_id"))}
        return job

    def job_items(self, job_id: str, offset: int = 0, limit: int = 50, status: str | None = None) -> dict[str, Any]:
        return self.job_store.items(job_id, offset, limit, status)

    # -- documents -------------------------------------------------------------------------------
    def ingest_document(self, path: str | None = None, content: bytes | None = None, filename: str | None = None,
                        metadata: dict[str, Any] | None = None, index: bool = True, mime_type: str | None = None,
                        source_uri: str | None = None, wait_seconds: float = 0) -> dict[str, Any]:
        with span("tool.ingest_document", "TOOL", {"tool.name": "ingest_document",
                                                   "tool.parameters": sanitize_params({"path": path,
                                                                                       "filename": filename})}):
            if path:
                resolved = resolve_authorized_path(path, self.config.storage.allowed_ingest_roots)
                if resolved.is_dir():
                    return self.ingest_directory(str(resolved), metadata, wait_seconds)
                result = self.documents.ingest_path(resolved, metadata)
            elif content is not None:
                name = safe_filename(filename or "upload.txt")
                result = self.documents.ingest_bytes(content, name, source_type="upload",
                                                     source_uri=source_uri or f"upload://{name}",
                                                     declared_mime=mime_type, metadata=metadata)
            else:
                raise InvalidArgument("provide either a local path or uploaded content")
            self._after_ingest(result)
            out = {**result.to_dict(), "document_ref": f"/api/documents/{result.document_id}",
                   "trace_id": current_trace_id()}
            if index and result.needs_indexing:
                doc_id = result.document_id
                job = self._submit("index_documents", {"document_ids": [doc_id]},
                                   lambda jc: self._index_job(jc, [doc_id], None, False), wait_seconds)
                out.update(job_id=job["job_id"], job_ref=job["job_ref"], job_status=job["status"])
            return out

    def _after_ingest(self, result: IngestResult) -> None:
        if result.status in {"duplicate", "failed"} and self.db.scalar(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (result.document_id,)):
            self.indexing.retire_index(result.document_id)

    def ingest_directory(self, path: str | None = None, metadata: dict[str, Any] | None = None,
                         wait_seconds: float = 0) -> dict[str, Any]:
        target = path or str(self.config.storage.allowed_ingest_roots[0])
        files = list(self.documents.iter_directory(target))  # validates the path before queuing

        def run(job: JobContext) -> dict[str, Any]:
            for f in files:
                if not self.documents.supported_extension(f):
                    job.add_item(str(f), "skipped", f"unsupported file type {f.suffix or '(none)'}")
                    continue
                try:
                    res = self.documents.ingest_path(f, metadata)
                    self._after_ingest(res)
                    self._record_ingest(job, str(f), res)
                except MeshError as exc:
                    job.add_item(str(f), "skipped" if isinstance(exc, UnsupportedMediaType) else "failed",
                                 exc.message)
                except Exception as exc:  # parsing failures never stop the job
                    job.add_item(str(f), "failed", f"{type(exc).__name__}: {str(exc)[:200]}")
            return {"directory": target, "files": len(files)}

        return self._submit("ingest_directory", {"path": target}, run, wait_seconds)

    def _record_ingest(self, job: JobContext, ref: str, res: IngestResult) -> None:
        if res.needs_indexing:
            out = self.indexing.index_document(res.document_id)
            job.add_item(ref, "indexed" if out["status"] == "indexed" else "skipped", out.get("reason"),
                         document_id=res.document_id, leaf_chunks=out.get("leaf_chunks"))
        elif res.status == "unchanged":
            job.add_item(ref, "skipped", "unchanged", document_id=res.document_id)
        elif res.status == "duplicate":
            job.add_item(ref, "skipped", "duplicate_content", document_id=res.document_id,
                         duplicate_of=res.duplicate_of)
        else:
            job.add_item(ref, "failed", res.reason, document_id=res.document_id)

    def index_documents(self, document_ids: list[str], chunk_sizes: list[int] | None = None, force: bool = False,
                        wait_seconds: float = 0) -> dict[str, Any]:
        if not document_ids:
            raise InvalidArgument("document_ids is required")
        if len(document_ids) > 1000:
            raise InvalidArgument("at most 1000 document_ids per call")
        for doc_id in document_ids:
            self.documents.get_row(doc_id)
        if chunk_sizes is not None:
            if not chunk_sizes or chunk_sizes != sorted(chunk_sizes, reverse=True) or \
                    any(not 32 <= s <= 8192 for s in chunk_sizes):
                raise InvalidArgument("chunk_sizes must be descending values between 32 and 8192 tokens")
        ids = list(dict.fromkeys(document_ids))
        return self._submit("index_documents", {"document_ids": ids[:50], "count": len(ids), "chunk_sizes": chunk_sizes},
                            lambda jc: self._index_job(jc, ids, chunk_sizes, force), wait_seconds)

    def _index_job(self, job: JobContext, ids: list[str], chunk_sizes, force: bool) -> dict[str, Any]:
        for doc_id in ids:
            try:
                out = self.indexing.index_document(doc_id, chunk_sizes, force)
                job.add_item(doc_id, out["status"], out.get("reason"),
                             **{k: out[k] for k in ("leaf_chunks", "nodes", "index_version") if k in out})
            except MeshError as exc:
                self.documents.mark_failed(doc_id, exc.message)
                job.add_item(doc_id, "failed", exc.message)
            except Exception as exc:
                log.exception("indexing failed", extra={"document_id": doc_id})
                self.documents.mark_failed(doc_id, f"{type(exc).__name__}: {exc}")
                job.add_item(doc_id, "failed", f"{type(exc).__name__}: {str(exc)[:200]}")
        return {"index_version": self.doc_index.version}

    def delete_document(self, document_id: str) -> dict[str, Any]:
        return self.indexing.delete_document(document_id)

    def read_document(self, document_id: str, page: int | None = None, section: str | None = None,
                      chunk_id: str | None = None, offset: int = 0, max_chars: int = 4000) -> dict[str, Any]:
        doc = self.documents.get(document_id)
        max_chars = max(100, min(int(max_chars), MAX_READ_CHARS))
        refs = {"document_ref": f"/api/documents/{document_id}", "original_ref": doc["original_ref"]}
        if chunk_id:
            chunk = self.get_chunk(chunk_id)
            if chunk["document_id"] != document_id:
                raise NotFound(f"chunk {chunk_id} does not belong to {document_id}")
            return {"document_id": document_id, "metadata": doc, "chunk": chunk, **refs}
        if doc["status"] in {"deleted", "skipped"}:
            raise NotFound(f"document {document_id} has no readable text ({doc['status']})")
        extracted = self.documents.load_extracted(document_id)
        sections = extracted.sections
        if page is not None:
            sections = [s for s in sections if s.page == page]
        if section:
            needle = section.lower()
            sections = [s for s in sections if s.heading and needle in s.heading.lower()]
        if (page is not None or section) and not sections:
            raise NotFound("no text for the requested page/section", {"page": page, "section": section})
        text = "\n\n".join(s.text for s in sections)
        offset = max(0, int(offset))
        chunk_text = text[offset: offset + max_chars]
        nxt = offset + max_chars if offset + max_chars < len(text) else None
        return {
            "document_id": document_id, "metadata": doc, "text": chunk_text, "offset": offset,
            "next_offset": nxt, "total_chars": len(text), "page": page, "section": section,
            "available_pages": sorted({s.page for s in extracted.sections if s.page is not None})[:500],
            "available_sections": list(dict.fromkeys(s.heading for s in extracted.sections if s.heading))[:100],
            **refs,
        }

    def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        node = self.doc_index.get_node(chunk_id) if row else None
        if row is None or node is None:
            raise NotFound(f"chunk not found: {chunk_id}")
        parents = [{"node_id": p.node_id, "level": p.metadata.get("level"),
                    "text": p.get_content()[:MAX_READ_CHARS // 4]} for p in self.doc_index.ancestors(chunk_id)]
        doc = self.documents.get(row["document_id"])
        return {
            "chunk_id": chunk_id, "document_id": row["document_id"], "level": row["level"],
            "is_leaf": bool(row["is_leaf"]), "page": row["page"], "section": row["section"],
            "parent_id": row["parent_id"], "text": node.get_content()[:MAX_READ_CHARS], "parents": parents,
            "source_uri": doc["source_uri"], "title": doc["title"],
            "document_ref": f"/api/documents/{row['document_id']}", "original_ref": doc["original_ref"],
        }

    # -- crawling --------------------------------------------------------------------------------
    def crawl_website(self, url: str, max_depth: int | None = None, max_pages: int | None = None,
                      wait_seconds: float = 0) -> dict[str, Any]:
        cfg = self.config.crawler
        depth = cfg.max_depth if max_depth is None else int(max_depth)
        pages = cfg.max_pages if max_pages is None else int(max_pages)
        if not 1 <= depth <= 10:
            raise InvalidArgument("max_depth must be between 1 (this page only) and 10")
        if not 1 <= pages <= 10_000:
            raise InvalidArgument("max_pages must be between 1 and 10000")
        start = run_sync(self.crawler.validate_start(url))  # rejects blocked/private URLs before queuing

        def run(job: JobContext) -> dict[str, Any]:
            def handler(final_url: str, body: bytes, ctype: str, page_depth: int) -> dict[str, Any]:
                try:
                    res = self.documents.ingest_bytes(
                        body, _filename_for(final_url, ctype), source_type="web", source_uri=final_url,
                        declared_mime=ctype, metadata={"crawl_job_id": job.job_id, "depth": page_depth},
                        max_bytes=cfg.max_response_bytes)
                except UnsupportedMediaType as exc:
                    return {"status": "skipped", "reason": f"unsupported_type: {exc.message}"}
                self._after_ingest(res)
                if res.needs_indexing:
                    out = self.indexing.index_document(res.document_id)
                    return {"status": "indexed", "document_id": res.document_id, "leaf_chunks": out.get("leaf_chunks")}
                if res.status == "unchanged":
                    return {"status": "skipped", "reason": "unchanged", "document_id": res.document_id}
                if res.status == "duplicate":
                    return {"status": "skipped", "reason": "duplicate_content", "document_id": res.document_id,
                            "duplicate_of": res.duplicate_of}
                return {"status": "failed", "reason": res.reason, "document_id": res.document_id}

            def on_item(outcome) -> None:
                detail = {"depth": outcome.depth}
                if outcome.document_id:
                    detail["document_id"] = outcome.document_id
                if outcome.final_url and outcome.final_url != outcome.url:
                    detail["final_url"] = outcome.final_url
                detail.update({k: v for k, v in outcome.detail.items() if v is not None})
                job.add_item(outcome.url, outcome.status, outcome.reason, **detail)

            summary = asyncio.run(self.crawler.crawl(start, handler, on_item, max_depth=depth, max_pages=pages))
            return {**summary, "index_version": self.doc_index.version}

        return self._submit("crawl_website", {"url": start, "max_depth": depth, "max_pages": pages}, run, wait_seconds)

    # -- structured ------------------------------------------------------------------------------
    def list_structured_sources(self) -> list[dict[str, Any]]:
        return self.catalog.list_sources()

    def extract_structured_metadata(self, source_id: str, datasets: list[str] | None = None,
                                    discover_keys: bool = False, generate_terms: bool = False,
                                    wait_seconds: float = 0) -> dict[str, Any]:
        self.catalog.source_settings(source_id)
        return self._submit(
            "extract_structured_metadata",
            {"source_id": source_id, "datasets": datasets, "discover_keys": discover_keys,
             "generate_terms": generate_terms},
            lambda jc: self._catalog_job(jc, source_id, datasets, discover_keys, generate_terms), wait_seconds)

    def catalog_source(self, source_id: str, datasets: list[str] | None = None,
                       wait_seconds: float = 0) -> dict[str, Any]:
        """Catalog datasets, then run key discovery and (when configured) term generation."""
        return self.extract_structured_metadata(source_id, datasets, discover_keys=True, generate_terms=True,
                                                wait_seconds=wait_seconds)

    def _catalog_job(self, job: JobContext, source_id: str, datasets, discover_keys: bool,
                     generate_terms: bool) -> dict[str, Any]:
        job.stage = "extract"
        extracted = self.catalog.extract(source_id, datasets, job)
        result: dict[str, Any] = {"extract": {k: extracted[k] for k in ("retired_datasets", "catalog_refs")},
                                  "dataset_count": len(extracted["dataset_ids"])}
        ids = extracted["dataset_ids"]
        # Terms first: key discovery uses them to decide which columns are identifiers (fk_rules).
        if generate_terms and ids:
            job.stage = "terms"
            result["terms"] = self._terms_stage(job, ids, force=False)
        if discover_keys and ids:
            job.stage = "keys"
            result["keys"] = self.keys.discover(ids, job=job)
        # Relationships last: they are proposed for the FKs found above, using the terms generated above.
        if generate_terms and ids:
            job.stage = "relationships"
            result["relationships"] = self._relationships_stage(job, ids)
        job.stage = "publish"
        result["glossary"] = self.glossary.sync_source(source_id)
        result["graph"] = self._publish_source(job, source_id)
        return result

    def _terms_stage(self, job: JobContext, dataset_ids: list[str], force: bool) -> dict[str, Any]:
        ok, reason = self.terms.available()
        if not ok:
            job.add_item("term_generation", "skipped", reason)
            return {"status": "skipped", "reason": reason}
        summary = {"generated": 0, "failed_datasets": 0}
        for ds in dataset_ids:
            try:
                out = self.terms.generate_for_dataset(ds, force, job)
                summary["generated"] += out.get("generated", 0)
            except Exception as exc:  # reported against the affected dataset (FR-12.4)
                summary["failed_datasets"] += 1
                message = exc.message if isinstance(exc, MeshError) else f"{type(exc).__name__}: {str(exc)[:200]}"
                job.add_item(ds, "failed", f"term generation failed: {message}")
        return summary

    def _relationships_stage(self, job: JobContext, dataset_ids: list[str]) -> dict[str, Any]:
        ok, reason = self.relationships.available()
        if not ok:
            job.add_item("relationship_generation", "skipped", reason)
            return {"status": "skipped", "reason": reason}
        summary = {"relationships": 0, "term_relations": 0, "term_synonyms": 0, "failed_datasets": 0}
        for ds in dataset_ids:
            try:
                out = self.relationships.generate_for_dataset(ds, job)
            except Exception as exc:  # reported against the affected dataset; other datasets continue
                summary["failed_datasets"] += 1
                message = exc.message if isinstance(exc, MeshError) else f"{type(exc).__name__}: {str(exc)[:200]}"
                job.add_item(f"relationships:{ds}", "failed", f"relationship generation failed: {message}")
                continue
            if out.get("status") == "failed":
                summary["failed_datasets"] += 1
            for key in ("relationships", "term_relations", "term_synonyms"):
                summary[key] += out.get(key, 0)
        return summary

    def generate_relationships(self, dataset_ids: list[str], wait_seconds: float = 0) -> dict[str, Any]:
        """Propose named relationships, term relations and synonyms for these datasets, then republish."""
        if not dataset_ids:
            raise InvalidArgument("dataset_ids is required")
        sources = {self.catalog.dataset_row(d)["source_id"] for d in dataset_ids}

        def run(job: JobContext) -> dict[str, Any]:
            job.stage = "relationships"
            result = self._relationships_stage(job, dataset_ids)
            job.stage = "publish"
            result["graph"] = {s: self._publish_source(job, s) for s in sorted(sources)}
            return result

        return self._submit("generate_relationships", {"dataset_ids": dataset_ids}, run, wait_seconds)

    def dataset_relationships(self, dataset_id: str) -> dict[str, Any]:
        return self.relationships.for_dataset(dataset_id)

    def review(self, object_type: str, object_id: str, status: str, actor: str | None = None,
               note: str | None = None) -> dict[str, Any]:
        """Verify or reject an inferred FK, relationship, term relation or mapping; ``inferred`` clears it.
        The affected source's graph layers are republished with the new status."""
        with span("graph.review", "CHAIN", {"review.object_type": object_type, "review.status": status}):
            out = self.reviews.set(object_type, object_id, status, actor, note)
            out["graph"] = self._publish_source(None, out["source_id"])
            log.info("review decision", extra={"object_type": object_type, "status": status, "actor": actor})
            return out

    def _publish_source(self, job: JobContext | None, source_id: str) -> dict[str, Any]:
        layers = build_source_layers(self.db, source_id, FKRules.from_settings(self.config.profiling),
                                     min_fk_confidence=self.config.relationship_generation.min_fk_confidence)
        try:
            out = self.graph.publish_layers(source_id, layers, f"structured source {source_id}")
        except GraphValidationError as exc:
            if job:
                job.add_item(f"graph:{source_id}", "failed", exc.message,
                             violations=exc.details.get("violations", [])[:10])
            return {"status": "rejected", "violations": len(exc.details.get("violations", []))}
        if job:
            job.add_item(f"graph:{source_id}", "succeeded", None, triples=out["triples"], contexts=out["contexts"])
        self._publish_glossary()  # term generation can have changed generated labels and descriptions
        return {"status": "published", "triples": out["triples"], "contexts": out["contexts"]}

    def _publish_glossary(self) -> dict[str, Any]:
        """(Re)publish the business glossary context: every active term and entity with its description,
        after dropping generated terms / entities nothing uses any more (see ``GlossaryStore.prune_unused``)."""
        self.glossary_store.prune_unused()
        try:
            out = self.graph.publish(context_uri("glossary", "business"), build_glossary_context(self.db),
                                     "business glossary")
        except GraphValidationError as exc:
            log.warning("business glossary rejected by SHACL", extra={"violations": len(exc.details.get("violations", []))})
            return {"status": "rejected", "violations": len(exc.details.get("violations", []))}
        return {"status": "published", "triples": out["triples"]}

    def provision_graph(self, source_ids: list[str] | None = None) -> dict[str, Any]:
        """Publish the vocabulary and (re)build the layered contexts of cataloged sources from SQLite.

        Without ``source_ids``: every source still in the earlier single-context layout (startup migration).
        A source whose layers fail SHACL keeps its previous contexts and is reported.
        """
        with span("graph.provision", "CHAIN") as sp:
            result: dict[str, Any] = {"ontology": self.graph.publish_ontology(self.config.knowledge_graph.ontology_path)}
            targets = source_ids if source_ids is not None else self.graph.legacy_source_contexts()
            result["sources"] = {s: self._publish_source(None, s) for s in targets}
            for s, out in result["sources"].items():
                if out["status"] != "published":
                    log.warning("graph provisioning rejected for a source; its previous contexts are kept",
                                extra={"source_id": s, "violations": out.get("violations")})
            sp.set_attribute("graph.provisioned_sources", len(targets))
            return result

    def discover_keys(self, dataset_ids: list[str], max_rows: int | None = None,
                      wait_seconds: float = 0) -> dict[str, Any]:
        if not dataset_ids:
            raise InvalidArgument("dataset_ids is required")
        sources = {self.catalog.dataset_row(d)["source_id"] for d in dataset_ids}

        def run(job: JobContext) -> dict[str, Any]:
            job.stage = "keys"
            result = self.keys.discover(dataset_ids, max_rows, job)
            job.stage = "publish"
            result["graph"] = {s: self._publish_source(job, s) for s in sorted(sources)}
            return result

        return self._submit("discover_keys", {"dataset_ids": dataset_ids, "max_rows": max_rows}, run, wait_seconds)

    def generate_terms(self, dataset_ids: list[str], force: bool = False, wait_seconds: float = 0) -> dict[str, Any]:
        sources = {self.catalog.dataset_row(d)["source_id"] for d in dataset_ids}

        def run(job: JobContext) -> dict[str, Any]:
            job.stage = "terms"
            result = self._terms_stage(job, dataset_ids, force)
            job.stage = "publish"
            result["graph"] = {s: self._publish_source(job, s) for s in sorted(sources)}
            return result

        return self._submit("generate_terms", {"dataset_ids": dataset_ids, "force": force}, run, wait_seconds)

    # -- settings: structured sources -------------------------------------------------------------
    def settings_overview(self) -> dict[str, Any]:
        return {
            "sources": self.catalog.list_sources(),
            "secret_store": {"backend": self.secrets.backend, "available": self.secrets.available,
                             "reason": getattr(self.secrets, "reason", None)},
            "allowed_source_roots": describe_roots(self.config),
            "database_types": {k: {"label": v["label"], "port": v["port"]} for k, v in DB_TYPES.items()},
        }

    def add_source(self, payload: dict[str, Any], actor: str | None = None) -> dict[str, Any]:
        with span("settings.add_source", "CHAIN", {"source.id": str(payload.get("id")),
                                                   "source.kind": str(payload.get("kind"))}):
            out = self.sources.create(payload, actor)
            log.info("structured source added", extra={"source_id": out["source_id"], "kind": out["kind"],
                                                       "actor": actor})
            return out

    def update_source(self, source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with span("settings.update_source", "CHAIN", {"source.id": source_id}):
            out = self.sources.update(source_id, payload)
            log.info("structured source updated", extra={"source_id": source_id})
            return out

    def remove_source(self, source_id: str) -> dict[str, Any]:
        """Remove a Settings-page source and retire its catalog, glossary and graph evidence."""
        with span("settings.remove_source", "CHAIN", {"source.id": source_id}):
            self.sources.delete(source_id)
            retired = self.catalog.retire_source(source_id)
            glossary = self.glossary.sync_source(source_id)
            self.graph.remove_source(source_id)
            self._publish_glossary()  # the source's generated terms and entities leave the glossary
            log.info("structured source removed", extra={"source_id": source_id, "retired_datasets": retired})
            return {"source_id": source_id, "status": "removed", "retired_datasets": retired,
                    "glossary_vectors_removed": glossary["removed"]}

    def test_source(self, payload: dict[str, Any] | None = None, source_id: str | None = None) -> dict[str, Any]:
        with span("settings.test_source", "TOOL", {"source.id": source_id or str((payload or {}).get("id"))}):
            return self.sources.test(payload, source_id)

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self.catalog.get_dataset(dataset_id)

    def catalog_tree(self) -> list[dict[str, Any]]:
        return [{**s, **self.catalog.source_tree(s["source_id"])} for s in self.catalog.list_sources()]

    # -- business glossary: terms and entities --------------------------------------------------------
    def list_business_terms(self, query: str | None = None) -> dict[str, Any]:
        terms = self.glossary_store.list_terms(query)
        return {"total": len(terms), "terms": [
            {**t, "sensitive": None if t["sensitive"] is None else bool(t["sensitive"]),
             "ref": f"/api/business-terms/{t['slug']}"} for t in terms]}

    def get_business_term(self, slug: str) -> dict[str, Any]:
        t = self.glossary_store.term(slug)
        cols = self.db.query(
            "SELECT c.column_id, c.name, c.catalog_ref, c.term_explanation, d.dataset_id, d.name AS dataset,"
            " d.source_id FROM columns c JOIN datasets d ON d.dataset_id = c.dataset_id "
            "WHERE c.term_slug = ? AND c.active = 1 AND c.term_status = 'generated' ORDER BY d.name, c.ordinal", (slug,))
        node = term_uri(slug)
        return {"slug": slug, "label": t["label"], "description": t["description"], "graph_ref": str(node),
                "sensitive": None if t["sensitive"] is None else bool(t["sensitive"]),  # can identify a person
                "origin": t["origin"], "created_at": t["created_at"], "updated_at": t["updated_at"],
                "updated_by": t["updated_by"], "ref": f"/api/business-terms/{slug}",
                "columns": [{**c, "term_label": t["label"], "term_description": t["description"],
                             "dataset_ref": f"/api/datasets/{c['dataset_id']}"} for c in cols],
                "definitions": self.definitions.for_term(slug), "graph": self.graph.describe_concept(node)}

    def create_business_term(self, label: str, description: str | None = None, sensitive: bool | None = None,
                             actor: str | None = None) -> dict[str, Any]:
        with span("glossary.term.create", "CHAIN"):
            row = self.glossary_store.add_term(label, description, sensitive, actor)
            self._after_glossary_change({})
            self.audit.record("create_business_term", caller=actor, resource_type="business_term",
                              resource_ids=[row["slug"]])
            return self.get_business_term(row["slug"])

    def update_business_term(self, slug: str, label: str, description: str | None = None,
                             sensitive: bool | None = None, actor: str | None = None) -> dict[str, Any]:
        """Edit a term's label, description and flag; the slug (graph URI) stays. Republishes the graph
        contexts and glossary records that show it."""
        with span("glossary.term.update", "CHAIN", {"term.slug": slug}):
            scope = self.glossary_store.term_scope(slug)
            self.glossary_store.update_term(slug, label, description, sensitive, actor)
            self._after_glossary_change(scope, definition_ids=[d["definition_id"]
                                                               for d in self.definitions.for_term(slug)])
            self.audit.record("update_business_term", caller=actor, resource_type="business_term",
                              resource_ids=[slug])
            return self.get_business_term(slug)

    def delete_business_term(self, slug: str, actor: str | None = None) -> dict[str, Any]:
        """Refused (409) while a business definition references the term."""
        with span("glossary.term.delete", "CHAIN", {"term.slug": slug}):
            out = self.glossary_store.delete_term(slug, actor)
            out["graph"] = self._after_glossary_change(out)
            self.audit.record("delete_business_term", caller=actor, resource_type="business_term",
                              resource_ids=[slug])
            return out

    def list_business_entities(self, query: str | None = None) -> dict[str, Any]:
        entities = self.glossary_store.list_entities(query)
        return {"total": len(entities),
                "entities": [{**e, "ref": f"/api/business-entities/{e['slug']}"} for e in entities]}

    def get_business_entity(self, slug: str) -> dict[str, Any]:
        e = self.glossary_store.entity(slug)
        node = entity_uri(slug)
        datasets = self.db.query("SELECT dataset_id, name, catalog_ref, source_id FROM datasets "
                                 "WHERE entity_slug = ? AND active = 1 ORDER BY name", (slug,))
        mentions = self.db.query("SELECT document_id, chunk_id FROM doc_entity_mentions WHERE entity_slug = ? LIMIT 25",
                                 (slug,))
        return {"slug": slug, "label": e["label"], "description": e["description"], "origin": e["origin"],
                "created_at": e["created_at"], "updated_at": e["updated_at"], "updated_by": e["updated_by"],
                "graph_ref": str(node), "ref": f"/api/business-entities/{slug}",
                "datasets": [{**d, "ref": f"/api/datasets/{d['dataset_id']}"} for d in datasets],
                "document_mentions": [{**m, "chunk_ref": f"/api/chunks/{m['chunk_id']}"} for m in mentions],
                "mention_count": self.db.scalar("SELECT COUNT(*) FROM doc_entity_mentions WHERE entity_slug = ?",
                                                (slug,)),
                "graph": self.graph.describe_concept(node)}

    def update_business_entity(self, slug: str, label: str, description: str | None = None,
                               actor: str | None = None) -> dict[str, Any]:
        """Edit an entity's label and description; the slug (graph URI) stays."""
        with span("glossary.entity.update", "CHAIN", {"entity.slug": slug}):
            scope = self.glossary_store.entity_scope(slug)
            self.glossary_store.update_entity(slug, label, description, actor)
            self._after_glossary_change(scope)
            self.audit.record("update_business_entity", caller=actor, resource_type="business_entity",
                              resource_ids=[slug])
            return self.get_business_entity(slug)

    def delete_business_entity(self, slug: str, actor: str | None = None) -> dict[str, Any]:
        """Remove the entity everywhere: datasets lose it, its document mentions and document relationships
        are removed and relationships to or from it are deactivated; generation does not re-create it."""
        with span("glossary.entity.delete", "CHAIN", {"entity.slug": slug}):
            out = self.glossary_store.delete_entity(slug, actor)
            out["graph"] = self._after_glossary_change(out)
            self.audit.record("delete_business_entity", caller=actor, resource_type="business_entity",
                              resource_ids=[slug])
            return out

    def _after_glossary_change(self, scope: dict[str, Any], definition_ids: list[str] | None = None) -> dict[str, Any]:
        """Republish what shows the changed terms / entities: the glossary context, the sources' layers,
        the documents' and definitions' contexts (one write), and the glossary vector records."""
        sources = scope.get("source_ids", [])
        # publishing a source also republishes the glossary context
        out: dict[str, Any] = {"sources": {s: self._publish_source(None, s) for s in sources}}
        if not sources:
            out["glossary"] = self._publish_glossary()
        contexts = {}
        for doc_id in scope.get("document_ids", []):
            g = self.indexing.document_graph(doc_id)
            if g is not None:
                contexts[context_uri("document", doc_id)] = g
        for def_id in definition_ids or []:
            g = self.definitions.graph_for(def_id)
            if g is not None:
                contexts[context_uri("definition", def_id)] = g
        if contexts:
            try:
                out["contexts"] = self.graph.publish_many(contexts, "business glossary change")
            except GraphValidationError as exc:
                out["contexts"] = {"status": "rejected", "violations": len(exc.details.get("violations", []))}
        for ds in scope.get("dataset_ids", []):
            self.glossary.upsert_dataset(ds)
        self.glossary.upsert_unbound_terms()
        return out

    def get_profiling_run(self, run_id: str) -> dict[str, Any]:
        return self.keys.get_run(run_id)

    # -- definitions -----------------------------------------------------------------------------
    def create_definition(self, name: str, definition: str, terms: list[str] | None, document_id: str | None,
                          location: str | None = None, author: str | None = None) -> dict[str, Any]:
        """A business definition of existing business terms, optionally defined in an ingested policy document."""
        with span("definitions.create", "CHAIN", {"definition.slug": slugify(name or "")}):
            return self.definitions.create(name, definition, terms, document_id, location, author)

    def update_definition(self, definition_id: str, name: str, definition: str, terms: list[str] | None,
                          document_id: str | None, location: str | None = None,
                          author: str | None = None) -> dict[str, Any]:
        with span("definitions.update", "CHAIN", {"definition.slug": definition_id}):
            return self.definitions.update(definition_id, name, definition, terms, document_id, location, author)

    # -- retrieval -------------------------------------------------------------------------------
    def retrieve_documents(self, queries: list[str], top_k: int | None = None, evidence_types: list[str] | None = None,
                           document_ids: list[str] | None = None, source_types: list[str] | None = None,
                           structured_source_ids: list[str] | None = None, dataset_ids: list[str] | None = None,
                           expand_parents: bool | None = None, rerank: bool | None = None) -> dict[str, Any]:
        """Search a list of queries in parallel and merge them into one deduplicated result."""
        return self.retrieval.retrieve_many(queries, top_k, evidence_types, document_ids, source_types,
                                            structured_source_ids, dataset_ids, expand_parents, rerank)

    # -- knowledge graph, SQL and HTML tools (MCP + agent) -------------------------------------------
    def query_knowledge_graph(self, request: KGQueryRequest | dict[str, Any],
                              structured_source_ids: list[str] | None = None, dataset_ids: list[str] | None = None,
                              document_ids: list[str] | None = None) -> dict[str, Any]:
        """Graph-aware retrieval for one intent: entities, terms, ranked multi-hop paths and physical mappings."""
        return self.kg.query(request, structured_source_ids, dataset_ids, document_ids)

    def execute_sql(self, source_id: str, sql: str, max_rows: int | None = None,
                    structured_source_ids: list[str] | None = None, dataset_ids: list[str] | None = None,
                    context: RequestContext | None = None) -> dict[str, Any]:
        """One read-only SELECT on a cataloged database / Databricks source (status ok | error)."""
        out = self.sql.execute(source_id, sql, max_rows, structured_source_ids, dataset_ids)
        code = None if out["status"] == "ok" else out["error"]["code"]
        denied = code in ("out_of_scope", "not_read_only", "uncataloged_table", "ambiguous_table", "unknown_source")
        self.audit.record("execute_sql", caller=(context or RequestContext()).caller, resource_type="dataset",
                          resource_ids=[source_id] + [t["dataset_id"] for t in out.get("tables", [])],
                          decision="denied" if denied else "allowed", reason=code)
        return out

    def generate_sql(self, request: SqlGenerationRequest | dict[str, Any],
                     structured_source_ids: list[str] | None = None,
                     dataset_ids: list[str] | None = None) -> dict[str, Any]:
        """One read-only SELECT written by the agent model from the evidence in ``request`` (checked, not run)."""
        req = request if isinstance(request, SqlGenerationRequest) else SqlGenerationRequest.model_validate(request)
        return self.sql_generator.generate(req, structured_source_ids, dataset_ids)

    def generate_html(self, request: HtmlRequest | dict[str, Any]) -> dict[str, Any]:
        """An escaped HTML fragment from typed content blocks (no LLM)."""
        try:
            return render_html(request, self.config.html_output.max_chars, self.config.html_output.max_table_rows)
        except ValueError as exc:
            raise InvalidArgument(str(exc)[:500]) from exc

    def graph_view(self, source_id: str, schema: str | None = None, dataset_id: str | None = None) -> dict[str, Any]:
        """Nodes and edges of the knowledge graph for a catalog selection (database/source, schema or table)."""
        self.catalog.source_settings(source_id)
        return graph_view(self.graph, self.config.knowledge_graph.view_max_nodes, source_id, schema, dataset_id)

    def answer(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self.answers.answer(query, **kwargs)

    def search(self, query: str, top_k: int | None = None, evidence_types: list[str] | None = None,
               document_ids: list[str] | None = None, source_types: list[str] | None = None,
               structured_source_ids: list[str] | None = None, dataset_ids: list[str] | None = None,
               expand_parents: bool | None = None, rerank: bool | None = None,
               context: RequestContext | None = None, judge: str | None = None,
               output_format: str = "text") -> dict[str, Any]:
        """Search with the agent harness (planner -> orchestration loop of task executor loops -> cited answer).

        The request filters bound every retrieval the agent makes. Without a usable agent model the
        search falls back to direct retrieval (search must work without a generative LLM); the
        response then has ``mode: "retrieval"`` and ``agent.reason``. ``output_format="html"`` asks the
        executor for HTML output (``generate_html``); fragments come back in ``html``. A search that ends
        with ``outcome: "needs_clarification"`` is continued with ``continue_search``.
        """
        args = (top_k, evidence_types, document_ids, source_types, structured_source_ids, dataset_ids,
                expand_parents, rerank)
        if output_format not in ("text", "html"):
            raise InvalidArgument("output_format must be text or html")
        reason = self.harness.disabled_reason()
        if reason is None:
            self._check_search_args(query, top_k, evidence_types)
            scope = SearchScope(top_k, evidence_types, document_ids, source_types, structured_source_ids,
                                dataset_ids, expand_parents, rerank, output_format,
                                (context or RequestContext()).caller)
            try:
                return self._after_search(self.harness.run(query.strip(), scope, context), context, judge)
            except AgentUnavailable as exc:
                log.warning("agent unavailable; using direct retrieval", extra={"reason": exc.message[:300]})
                get_metrics().agent_recoveries.add(1, {"kind": "retrieval_fallback"})
                reason = exc.message
        out = self.retrieve_documents([query], *args)
        self.audit.record("search", caller=(context or RequestContext()).caller, resource_type="document",
                          resource_ids=[r["document_id"] for r in out["results"]], trace_id=out.get("trace_id"),
                          reason="direct retrieval")
        return {**out, "mode": "retrieval", "agent": {"used": False, "reason": reason}}

    def continue_search(self, trace_id: str, reply: str, context: RequestContext | None = None,
                        judge: str | None = None) -> dict[str, Any]:
        """Continue a search that asked the user for clarification (spec 5.10) with the user's reply.

        The original request, plan, task results and evidence were kept in SQLite; the harness resumes
        through its replan step with the reply and runs the same loop. Each pending question accepts one
        reply; it expires after ``agent.clarification_ttl_hours``."""
        reply = " ".join(str(reply or "").split())
        if not reply:
            raise InvalidArgument("reply is required")
        if len(reply) > self.config.retrieval.max_query_chars:
            raise InvalidArgument(f"reply exceeds {self.config.retrieval.max_query_chars} characters")
        self._purge_clarifications()
        row = self.db.query_one("SELECT * FROM pending_clarifications WHERE trace_id = ?", (trace_id,))
        if row is None:
            raise NotFound(f"no pending clarification for search {trace_id} (unknown or expired)")
        claimed = self.db.execute("UPDATE pending_clarifications SET status = 'answering' WHERE trace_id = ? "
                                  "AND status = 'pending'", (trace_id,))
        if claimed != 1:
            raise Conflict(f"search {trace_id} was already continued", {"status": row["status"]})
        state = loads(row["state_json"], {})
        scope = SearchScope(**state["scope"]) if state.get("scope") else None
        if scope is not None:
            scope.caller = (context or RequestContext()).caller
        try:
            result = self.harness.resume(state, reply, scope, context)
        except Exception:
            self.db.execute("UPDATE pending_clarifications SET status = 'pending' WHERE trace_id = ?", (trace_id,))
            raise
        self.db.execute("UPDATE pending_clarifications SET status = 'answered', answered_at = ?, state_json = '{}', "
                        "continuation_trace_id = ? WHERE trace_id = ?", (utcnow(), result["trace_id"], trace_id))
        return self._after_search(result, context, judge)

    def _purge_clarifications(self) -> None:
        """Expired questions and the state of answered ones are deleted."""
        self.db.execute("DELETE FROM pending_clarifications WHERE expires_at < ?", (utcnow(),))

    def _store_clarification(self, state: dict[str, Any], context: RequestContext) -> None:
        from datetime import datetime, timedelta, timezone

        expires = (datetime.now(timezone.utc) + timedelta(hours=self.config.agent.clarification_ttl_hours)
                   ).isoformat(timespec="seconds")
        self._purge_clarifications()
        self.db.execute(
            "INSERT OR REPLACE INTO pending_clarifications(trace_id, root_trace_id, round, created_at, expires_at,"
            " caller, session_id, question, state_json, status) VALUES (?,?,?,?,?,?,?,?,?,'pending')",
            (state["trace_id"], state["root_trace_id"], state["round"] + 1, utcnow(), expires, context.caller,
             context.session_id, state["question"], json.dumps(state, default=str)))

    def _after_search(self, result: dict[str, Any], context: RequestContext | None,
                      judge_mode: str | None = None) -> dict[str, Any]:
        """Record the search, audit what it touched and run the LLM judge: queued in the background when
        ``observability.online_judge`` is on, or right away with ``judge_mode="sync"`` (offline evaluation)."""
        context = context or RequestContext()
        judge_input = result.pop("_judge_input")
        resources = result.pop("_resources")
        clarification_state = result.pop("_clarification_state", None)
        if clarification_state is not None:
            self._store_clarification(clarification_state, context)
            result["clarification"]["expires_in_hours"] = self.config.agent.clarification_ttl_hours
        usage, evaluation = result["usage"], result["evaluation"]
        sync = judge_mode == "sync" and self.judge.available
        judge_status = "pending" if (sync or self.judge.enabled) else "disabled"
        replans = sum(1 for d in result["decisions"] if d["decision"] == "replan")
        self.db.execute(
            "INSERT OR REPLACE INTO searches(trace_id, span_id, created_at, caller, session_id, query, mode, outcome,"
            " latency_ms, llm_calls, tool_calls, input_tokens, output_tokens, cost_usd, replans, compliance,"
            " checks_json, judge_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (result["trace_id"], result["observability"]["span_id"], utcnow(), context.caller, context.session_id,
             result["query"][:2000], "agent", result["outcome"], usage["latency_ms"], usage["llm_calls"],
             usage["tool_calls"], usage["input_tokens"], usage["output_tokens"], usage["cost_usd"], replans,
             evaluation["compliance"], json.dumps({"checks": evaluation["checks"],
                                                    "violations": evaluation["violations"]}, default=str),
             judge_status))
        for c in evaluation["checks"]:
            self.db.execute("INSERT OR REPLACE INTO search_evaluations(trace_id, evaluator, name, score, label,"
                            " explanation, created_at) VALUES (?,?,?,?,?,?,?)",
                            (result["trace_id"], "rules", c["name"], 1.0 if c["passed"] else 0.0,
                             "pass" if c["passed"] else "fail", c.get("detail"), utcnow()))
        self.phoenix.annotate(result["observability"]["span_id"], "compliance", score=evaluation["compliance"],
                              label=", ".join(evaluation["violations"]) or "compliant", kind="CODE",
                              metadata={"failed_checks": [c["name"] for c in evaluation["checks"] if not c["passed"]]})
        self.audit.record("search", caller=context.caller, resource_type="document",
                          resource_ids=resources["documents"], trace_id=result["trace_id"])
        if resources["datasets"]:
            self.audit.record("search", caller=context.caller, resource_type="dataset",
                              resource_ids=resources["datasets"], trace_id=result["trace_id"])
        if resources["tools"]:
            self.audit.record("tool_use", caller=context.caller, resource_type="tool",
                              resource_ids=resources["tools"], trace_id=result["trace_id"])
        result["evaluation"]["judge"] = {"status": judge_status}
        if sync:
            verdict = self.judge.evaluate_now(judge_input)
            result["evaluation"]["judge"] = ({"status": "done", **verdict} if verdict
                                             else {"status": "failed"})
            result["_eval_context"] = {"evidence": judge_input.get("evidence")}
        elif self.judge.enabled:
            self.judge.submit(judge_input)
        result["observability"]["trace_url"] = self.phoenix.trace_url(result["trace_id"])
        result["models"]["judge"] = result["agent"]["model"] if judge_status != "disabled" else None
        return result

    def get_search(self, trace_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM searches WHERE trace_id = ?", (trace_id,))
        if row is None:
            raise NotFound(f"search not found: {trace_id}")
        checks = loads(row.pop("checks_json"), {})
        evaluations = evaluations_for(self.db, trace_id)
        judge = {e["name"]: {"score": e["score"], "label": e["label"], "explanation": e["explanation"]}
                 for e in evaluations if e["evaluator"] == "llm_judge"}
        feedback = self.db.query("SELECT rating, comment, caller, created_at FROM search_feedback WHERE trace_id = ?"
                                 " ORDER BY created_at", (trace_id,))
        return {**row, **checks, "judge": judge, "feedback": feedback,
                "trace_url": self.phoenix.trace_url(trace_id)}

    def add_feedback(self, trace_id: str, rating: str, comment: str | None = None,
                     context: RequestContext | None = None) -> dict[str, Any]:
        context = context or RequestContext()
        if rating not in ("up", "down"):
            raise InvalidArgument("rating must be 'up' or 'down'", {"fields": ["rating"]})
        row = self.db.query_one("SELECT span_id FROM searches WHERE trace_id = ?", (trace_id,))
        if row is None:
            raise NotFound(f"search not found: {trace_id}")
        comment = (comment or "").strip()[:2000] or None
        feedback_id = new_id("fb")
        self.db.execute("INSERT INTO search_feedback(feedback_id, trace_id, rating, comment, caller, created_at)"
                        " VALUES (?,?,?,?,?,?)", (feedback_id, trace_id, rating, comment, context.caller, utcnow()))
        get_metrics().feedback.add(1, {"rating": rating})
        sent = self.phoenix.annotate(row["span_id"], "user_feedback", score=1.0 if rating == "up" else 0.0,
                                     label=rating, explanation=comment, kind="HUMAN",
                                     metadata={"caller": context.caller})
        self.audit.record("feedback", caller=context.caller, resource_type="search", resource_ids=[trace_id],
                          reason=rating)
        return {"feedback_id": feedback_id, "trace_id": trace_id, "rating": rating, "sent_to_phoenix": sent}

    def record_alerts(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Store alerts posted by Alertmanager's webhook (firing and resolved)."""
        stored = 0
        for alert in payload.get("alerts") or []:
            labels = alert.get("labels") or {}
            name = labels.get("alertname") or "unknown"
            status = alert.get("status") or payload.get("status") or "firing"
            fingerprint = alert.get("fingerprint") or f"{name}:{alert.get('startsAt')}"
            self.db.execute(
                "INSERT OR REPLACE INTO ops_alerts(fingerprint, alertname, status, severity, summary, starts_at,"
                " ends_at, received_at, labels_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (fingerprint, name, status, labels.get("severity"),
                 (alert.get("annotations") or {}).get("summary"), alert.get("startsAt"), alert.get("endsAt"),
                 utcnow(), json.dumps(labels)))
            get_metrics().alerts_received.add(1, {"alertname": name, "status": status})
            log.warning("alert %s: %s", status, name, extra={"alertname": name, "alert_status": status,
                                                             "severity": labels.get("severity")})
            stored += 1
        return {"stored": stored}

    def list_alerts(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM ops_alerts", []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY received_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        rows = self.db.query(sql, params)
        for r in rows:
            r["labels"] = loads(r.pop("labels_json"), {})
        return rows

    def _check_search_args(self, query: str, top_k: int | None, evidence_types: list[str] | None) -> None:
        cfg = self.config.retrieval
        q = (query or "").strip()
        if not q:
            raise InvalidArgument("query is required")
        if len(q) > cfg.max_query_chars:
            raise InvalidArgument(f"query exceeds {cfg.max_query_chars} characters")
        if top_k is not None and not 1 <= int(top_k) <= cfg.max_top_k:
            raise InvalidArgument(f"top_k must be between 1 and {cfg.max_top_k}", {"top_k": top_k})
        unknown = set(evidence_types or []) - set(EVIDENCE_TYPES)
        if unknown:
            raise InvalidArgument(f"unknown evidence types: {sorted(unknown)}", {"allowed": list(EVIDENCE_TYPES)})

    # -- maintenance -----------------------------------------------------------------------------
    def reindex_all_documents(self, wait_seconds: float = 0) -> dict[str, Any]:
        ids = [r["document_id"] for r in self.db.query(
            "SELECT document_id FROM documents WHERE active = 1 AND status IN ('indexed','extracted')")]
        if not ids:
            raise NotFound("no documents to reindex")
        return self._submit("index_documents", {"count": len(ids), "reindex": True},
                            lambda jc: self._index_job(jc, ids, None, True), wait_seconds)

    def rebuild_indexes(self) -> dict[str, Any]:
        """Rebuild both vector indexes from stored extracted text and the catalog (e.g. after an
        embedding-model change). Old index directories are moved aside, never deleted."""
        stamp = time.strftime("%Y%m%dT%H%M%S")
        moved = {}
        for name, idx in (("documents", self.doc_index), ("glossary", self.glossary_index)):
            path = Path(idx.persist_dir)
            if path.exists():
                backup = path.with_name(f"{path.name}.bak-{stamp}")
                path.rename(backup)
                moved[name] = str(backup)
        self.doc_index = PersistentVectorIndex("documents", self.config.vector_store.document_persist_dir,
                                               self.embed_model)
        self.glossary_index = PersistentVectorIndex("glossary", self.config.vector_store.glossary_persist_dir,
                                                    self.embed_model)
        self.indexing.index = self.doc_index
        self.retrieval.doc_index = self.doc_index
        self.glossary.index = self.glossary_index
        self.db.execute("UPDATE documents SET indexed_content_hash = NULL WHERE active = 1")
        docs = [r["document_id"] for r in self.db.query(
            "SELECT document_id FROM documents WHERE active = 1 AND status IN ('indexed','extracted')")]
        counts = {"documents_indexed": 0, "documents_failed": 0, "datasets": 0}
        for doc_id in docs:
            try:
                self.indexing.index_document(doc_id, force=True)
                counts["documents_indexed"] += 1
            except Exception as exc:
                counts["documents_failed"] += 1
                self.documents.mark_failed(doc_id, f"reindex failed: {type(exc).__name__}")
        for d in self.db.query("SELECT dataset_id FROM datasets WHERE active = 1"):
            self.glossary.upsert_dataset(d["dataset_id"])
            counts["datasets"] += 1
        self.glossary.upsert_unbound_terms()
        return {**counts, "previous_indexes": moved, "document_index": self.doc_index.stats(),
                "glossary_index": self.glossary_index.stats()}

    def warmup(self) -> None:
        """Load the embedding model and open indexes ahead of the first request; migrate the graph layout."""
        try:
            out = self.provision_graph()
            if out["sources"]:
                log.info("graph layout migrated", extra={"sources": sorted(out["sources"])})
        except Exception as exc:  # the previously published graph stays in use
            log.error("graph provisioning failed: %s", type(exc).__name__)
        start = time.perf_counter()
        try:
            self.doc_index.stats()
            self.glossary_index.stats()
            log.info("indexes ready", extra={"seconds": round(time.perf_counter() - start, 2)})
        except MeshError as exc:
            log.error("index warmup failed: %s", exc.message)
        if self.config.retrieval.reranker_enabled:
            try:
                self.reranker.agent()
            except Exception as exc:  # search falls back to similarity order; reported per request
                log.error("reranker warmup failed: %s: %s", type(exc).__name__, str(exc)[:200])

    def health(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        status = "ok"
        try:
            self.db.scalar("SELECT 1")
            checks["metadata_db"] = {"status": "ok", "path": str(self.config.storage.metadata_db)}
        except Exception as exc:
            status = "degraded"
            checks["metadata_db"] = {"status": "error", "error": type(exc).__name__}
        for name, idx in (("document_index", self.doc_index), ("glossary_index", self.glossary_index)):
            meta_path = Path(idx.persist_dir) / META_FILE
            meta = loads(meta_path.read_text()) if meta_path.exists() else {}
            checks[name] = {"status": "ok", "loaded": idx._index is not None, "version": meta.get("version"),
                            "embedding_model": meta.get("embedding_model") or self.config.embedding.model,
                            "dimension": meta.get("dimension")}
        checks["knowledge_graph"] = {"status": "ok", **self.graph.stats()}
        ok, reason = self.terms.available() if self.config.term_generation.configured else (False, "not configured")
        checks["term_generation"] = {"configured": self.config.term_generation.configured, "available": ok,
                                     "reason": reason, "provider": self.config.term_generation.provider,
                                     "model": self.config.term_generation.model}
        checks["answer_generation"] = {"enabled": self.answers.enabled}
        checks["agent"] = {"enabled": self.harness.enabled, "reason": self.harness.disabled_reason(),
                           "provider": self.config.agent.provider, "model": self.config.agent.model,
                           "max_iterations": self.config.agent.max_iterations}
        obs = self.config.observability
        checks["observability"] = {"phoenix_url": obs.phoenix_url, "online_judge": self.judge.enabled,
                                   "capture_sensitive_content": obs.capture_sensitive_content,
                                   "loki_enabled": obs.loki_enabled, "prometheus_enabled": obs.prometheus_enabled}
        rr_ok, rr_reason = self.reranker.available()
        checks["reranker"] = {"enabled": self.config.retrieval.reranker_enabled, "provider": "laya",
                              "model": self.config.reranker.model, "available": rr_ok, "reason": rr_reason,
                              "loaded": self.reranker._agent is not None,
                              "min_probability": self.config.reranker.min_probability,
                              "min_quality": self.config.reranker.min_quality}
        checks["structured_sources"] = len(self.sources.all())
        checks["secret_store"] = {"backend": self.secrets.backend, "available": self.secrets.available}
        checks["jobs_running"] = self.db.scalar("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')")
        return {"status": status, "checks": checks, "trace_id": current_trace_id()}

    def close(self) -> None:
        self.mcp_tools.close()
        self.judge.close()
        self.phoenix.close()
        self.runner.shutdown()
        self.db.close()


def _filename_for(url: str, ctype: str) -> str:
    name = unquote(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]) or "index"
    ext = EXT_BY_MIME.get(ctype)
    if ext and not name.lower().endswith(ext):
        name = f"{name}{ext}"
    return safe_filename(name, "index.html")


__all__ = ["KnowledgeMesh", "run_sync", "URIRef"]

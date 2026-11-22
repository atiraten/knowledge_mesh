"""Mixed retrieval: document passages + business glossary + knowledge-graph evidence (FR-07).

Works without any generative LLM. Scores are cosine similarities from the local embedding
model (``score_type: cosine_similarity``). Results below the configured minimum similarity
are dropped; if nothing remains the response is an explicit insufficient-evidence result.
Retrieved text is untrusted data: passages that look like embedded instructions are flagged.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from rdflib import URIRef

from ..common.errors import InvalidArgument, NotConfigured
from ..common.guardrails import suspicious_instruction
from ..graph.ontology import entity_uri
from ..graph.store import KnowledgeGraph
from ..indexing.vector_index import PersistentVectorIndex
from ..observability import capture_sensitive, current_trace_id, get_metrics, span
from ..storage.db import Database
from ..structured.glossary import GlossaryService

log = logging.getLogger(__name__)
EVIDENCE_TYPES = ("documents", "glossary", "graph")


class RetrievalService:
    def __init__(self, settings, db: Database, doc_index: PersistentVectorIndex, glossary: GlossaryService,
                 graph: KnowledgeGraph, reranker=None):
        self.cfg = settings
        self.db = db
        self.doc_index = doc_index
        self.glossary = glossary
        self.graph = graph
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int | None = None, evidence_types: list[str] | None = None,
                 document_ids: list[str] | None = None, source_types: list[str] | None = None,
                 structured_source_ids: list[str] | None = None, dataset_ids: list[str] | None = None,
                 expand_parents: bool | None = None, rerank: bool | None = None) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise InvalidArgument("query is required")
        if len(query) > self.cfg.max_query_chars:
            raise InvalidArgument(f"query exceeds {self.cfg.max_query_chars} characters")
        k = self.cfg.top_k if top_k is None else int(top_k)
        if not 1 <= k <= self.cfg.max_top_k:
            raise InvalidArgument(f"top_k must be between 1 and {self.cfg.max_top_k}", {"top_k": top_k})
        types = list(evidence_types or EVIDENCE_TYPES)
        unknown = set(types) - set(EVIDENCE_TYPES)
        if unknown:
            raise InvalidArgument(f"unknown evidence types: {sorted(unknown)}", {"allowed": list(EVIDENCE_TYPES)})
        expand = self.cfg.expand_parent_context if expand_parents is None else expand_parents
        do_rerank = self.cfg.reranker_enabled if rerank is None else bool(rerank)
        if do_rerank and self.reranker is None:
            raise NotConfigured("reranking requested but no reranker is configured")
        metrics = get_metrics()
        start = time.perf_counter()
        reranking: dict[str, Any] | None = None
        with span("retrieval.retrieve_documents", "RETRIEVER",
                  {"input.value": query[:500], "retrieval.top_k": k, "retrieval.evidence_types": types}) as sp:
            documents: list[dict[str, Any]] = []
            if "documents" in types:
                documents = self._documents(query, k, document_ids, source_types)
                if do_rerank:
                    documents, reranking = self._rerank(query, documents)
                if expand:
                    self._expand_parents(documents, sp)
            glossary: list[dict[str, Any]] = []
            if "glossary" in types:
                glossary = self._glossary(query, k, structured_source_ids, dataset_ids)
            graph_items: list[dict[str, Any]] = []
            if "graph" in types and self.cfg.max_graph_results:
                seeds = self._seeds(query, glossary, documents)
                graph_items = self.graph.evidence_for(seeds, self.cfg.max_graph_results)
                sp.set_attribute("retrieval.graph.seed_count", len(seeds))
            for i, d in enumerate(documents):
                sp.set_attribute(f"retrieval.documents.{i}.document.id", d["chunk_id"])
                sp.set_attribute(f"retrieval.documents.{i}.document.score", d["score"])
                sp.set_attribute(f"retrieval.documents.{i}.document.metadata",
                                 f'{{"document_id": "{d["document_id"]}", "rank": {d["rank"]}}}')
                if capture_sensitive():
                    sp.set_attribute(f"retrieval.documents.{i}.document.content", d["text"])
            sp.set_attribute("retrieval.glossary.ids", [g["graph_ref"] for g in glossary])
            sp.set_attribute("retrieval.glossary.scores", [g["score"] for g in glossary])
            sp.set_attribute("retrieval.graph.ids", [g["graph_ref"] for g in graph_items])
            if capture_sensitive():
                sp.set_attribute("output.value", json.dumps({"glossary": glossary, "graph": graph_items},
                                                            default=str)[:20000])
            insufficient = not (documents or glossary or graph_items)
            sp.set_attribute("retrieval.insufficient_evidence", insufficient)
            flagged = sum(1 for d in documents if d.get("flags"))
            if flagged:
                metrics.guardrail_events.add(flagged, {"kind": "suspicious_document_instruction"})
                sp.set_attribute("guardrail.suspicious_passages", flagged)
            trace_id = current_trace_id()
        metrics.stage_duration.record(time.perf_counter() - start, {"stage": "retrieval"})
        metrics.retrieval_results.record(len(documents), {"index": "documents"})
        metrics.retrieval_results.record(len(glossary), {"index": "glossary"})
        metrics.retrieval_results.record(len(graph_items), {"index": "graph"})
        for index, items, searched in (("documents", documents, "documents" in types),
                                       ("glossary", glossary, "glossary" in types)):
            if not searched:
                continue
            if items:
                metrics.retrieval_top_score.record(max(i["score"] for i in items), {"index": index})
            else:
                metrics.retrieval_empty.add(1, {"index": index})
        response: dict[str, Any] = {
            "query": query, "top_k": k, "score_type": "cosine_similarity",
            "results": documents, "glossary_results": glossary, "graph_evidence": graph_items,
            "insufficient_evidence": insufficient, "trace_id": trace_id,
        }
        if reranking is not None:
            response["reranking"] = reranking
            if reranking.get("applied"):
                chosen = documents[0] if documents else None
                response["response"] = ({"found": True, "text": chosen["text"], "chunk_id": chosen["chunk_id"],
                                         "document_id": chosen["document_id"], "citation": chosen["citation"]}
                                        if chosen else {"found": False, "text": reranking["message"]})
        if reranking and reranking.get("outcome") == "no_response":
            response["message"] = reranking["message"]
        elif insufficient:
            response["message"] = ("No indexed passage, glossary entry or graph evidence is relevant enough "
                                   "to this query; no sources are cited.")
        return response

    def retrieve_many(self, queries: list[str], top_k: int | None = None, evidence_types: list[str] | None = None,
                      document_ids: list[str] | None = None, source_types: list[str] | None = None,
                      structured_source_ids: list[str] | None = None, dataset_ids: list[str] | None = None,
                      expand_parents: bool | None = None, rerank: bool | None = None) -> dict[str, Any]:
        """Search several queries in parallel and merge the results into one output.

        Exact duplicates are removed: a passage (chunk ID), glossary entry (catalog ref) or graph fact
        (graph ref) found by several queries appears once, with its best score and ``matched_queries``.
        Passages and glossary entries are sorted by score.
        """
        qs: list[str] = []
        seen: set[str] = set()
        for q in queries or []:
            q = " ".join(str(q or "").split())
            if q and q.casefold() not in seen:
                seen.add(q.casefold())
                qs.append(q)
        if not qs:
            raise InvalidArgument("at least one query is required")
        if len(qs) > self.cfg.max_queries:
            raise InvalidArgument(f"at most {self.cfg.max_queries} queries per retrieval", {"queries": len(qs)})
        for q in qs:
            if len(q) > self.cfg.max_query_chars:
                raise InvalidArgument(f"query exceeds {self.cfg.max_query_chars} characters")
        args = (top_k, evidence_types, document_ids, source_types, structured_source_ids, dataset_ids,
                expand_parents, rerank)
        with span("retrieval.retrieve_many", "RETRIEVER", {"retrieval.queries": qs}) as sp:
            if len(qs) == 1:
                results = [self.retrieve(qs[0], *args)]
            else:
                workers = min(self.cfg.max_parallel_queries, len(qs))
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="retrieve") as pool:
                    futures = [pool.submit(contextvars.copy_context().run, self.retrieve, q, *args) for q in qs]
                    results = [f.result() for f in futures]
            documents = _merge(qs, results, "results", lambda d: d["chunk_id"], by_score=True)
            glossary = _merge(qs, results, "glossary_results", lambda g: g.get("catalog_ref") or g["graph_ref"],
                              by_score=True)
            graph_items = _merge(qs, results, "graph_evidence", lambda k: k["graph_ref"], by_score=False)
            insufficient = not (documents or glossary or graph_items)
            raw = sum(len(r["results"]) + len(r["glossary_results"]) + len(r["graph_evidence"]) for r in results)
            merged = len(documents) + len(glossary) + len(graph_items)
            sp.set_attribute("retrieval.merged.documents", len(documents))
            sp.set_attribute("retrieval.merged.glossary", len(glossary))
            sp.set_attribute("retrieval.merged.graph", len(graph_items))
            sp.set_attribute("retrieval.duplicates_removed", raw - merged)
            sp.set_attribute("retrieval.duplicate_ratio", round((raw - merged) / raw, 3) if raw else 0.0)
            sp.set_attribute("retrieval.per_query", json.dumps(
                [{"query": q, "documents": len(r["results"]), "glossary": len(r["glossary_results"]),
                  "graph": len(r["graph_evidence"])} for q, r in zip(qs, results)]))
            if documents:
                sp.set_attribute("retrieval.top_score", documents[0]["score"])
            trace_id = current_trace_id()
        response: dict[str, Any] = {
            "queries": qs, "top_k": results[0]["top_k"], "score_type": "cosine_similarity",
            "results": documents, "glossary_results": glossary, "graph_evidence": graph_items,
            "per_query": [{"query": q, "documents": len(r["results"]), "glossary": len(r["glossary_results"]),
                           "graph": len(r["graph_evidence"])} for q, r in zip(qs, results)],
            "insufficient_evidence": insufficient, "trace_id": trace_id,
        }
        # Laya decisions per query, with the passage it selected (``response``)
        reranking = [{"query": q, **r["reranking"], "response": r.get("response")}
                     for q, r in zip(qs, results) if r.get("reranking")]
        if reranking:
            response["reranking"] = reranking
        if len(results) == 1 and results[0].get("message"):  # one query: its own message (e.g. Laya's no-response)
            response["message"] = results[0]["message"]
        elif insufficient:
            response["message"] = ("No indexed passage, glossary entry or graph evidence is relevant enough "
                                   "to these queries; no sources are cited.")
        return response

    def _rerank(self, query: str, documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Keep only the passage the reranker selects; on reranker failure keep similarity order."""
        try:
            decision = self.reranker.rerank(query, documents)
        except Exception as exc:  # the decision model is optional: never fail the search because of it
            log.warning("reranking failed; returning similarity order", extra={"error": type(exc).__name__})
            get_metrics().rerank_outcomes.add(1, {"outcome": "failed"})
            return documents, {"applied": False, "outcome": "failed",
                               "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        index = decision.get("selected_index")
        if index is None:
            return [], decision
        chosen = dict(documents[index])
        chosen["rank"] = 1
        chosen["rerank"] = next(d for d in decision["decisions"] if d["selected"])
        return [chosen], decision

    # -- documents -------------------------------------------------------------------------------
    def _documents(self, query: str, k: int, document_ids, source_types) -> list[dict[str, Any]]:
        hits = self.doc_index.retrieve(query, k, {"document_id": document_ids, "source_type": source_types})
        out = []
        for hit in hits:
            score = float(hit.score or 0.0)
            if score < self.cfg.min_document_similarity:
                continue
            node, md = hit.node, hit.node.metadata
            text = node.get_content()
            page = md.get("page")
            item = {
                "rank": len(out) + 1,
                "document_id": md.get("document_id"),
                "chunk_id": node.node_id,
                "parent_id": md.get("parent_id") or None,
                "source_uri": md.get("source_uri"),
                "title": md.get("title") or None,
                "page": page if isinstance(page, int) and page >= 0 else None,
                "section": md.get("section") or None,
                "score": round(score, 4),
                "score_type": "cosine_similarity",
                "text": text[: self.cfg.excerpt_chars],
                "citation": {
                    "document_ref": f"/api/documents/{md.get('document_id')}",
                    "chunk_ref": f"/api/chunks/{node.node_id}",
                    "original_ref": f"/api/documents/{md.get('document_id')}/original",
                },
            }
            if suspicious_instruction(text):
                item["flags"] = ["suspicious_instruction"]
            out.append(item)
        return out

    def _expand_parents(self, documents: list[dict[str, Any]], sp) -> None:
        """Attach each leaf's whole parent chain (nearest first) within the context budget.

        Leaf text counts first; then parents are filled one hierarchy step at a time (every result's
        nearest parent before any grandparent). A parent shared by several results counts once.
        Parents that do not fit stay listed by ID with ``expanded: false``.
        """
        budget = self.cfg.max_context_tokens * 4  # ~4 characters per token
        used = sum(len(d["text"]) for d in documents)
        chains = {d["chunk_id"]: self.doc_index.ancestors(d["chunk_id"]) for d in documents}
        fits: dict[str, bool] = {}
        for step in range(max((len(c) for c in chains.values()), default=0)):
            for d in documents:
                chain = chains[d["chunk_id"]]
                if step >= len(chain) or chain[step].node_id in fits:
                    continue
                size = len(chain[step].get_content())
                fits[chain[step].node_id] = used + size <= budget
                if fits[chain[step].node_id]:
                    used += size
        for d in documents:
            d["parents"] = [
                {"node_id": p.node_id, "level": p.metadata.get("level"), "expanded": True, "text": p.get_content()}
                if fits[p.node_id] else
                {"node_id": p.node_id, "level": p.metadata.get("level"), "expanded": False,
                 "reason": "context budget exceeded"}
                for p in chains[d["chunk_id"]]
            ]
        sp.set_attribute("retrieval.parent_expansion.expanded", sum(fits.values()))
        sp.set_attribute("retrieval.parent_expansion.pruned", len(fits) - sum(fits.values()))
        sp.set_attribute("retrieval.context_chars", used)

    # -- glossary --------------------------------------------------------------------------------
    def _glossary(self, query: str, k: int, source_ids, dataset_ids) -> list[dict[str, Any]]:
        hits = self.glossary.search(query, k, None, source_ids, dataset_ids)
        out = []
        for hit in hits:
            score = float(hit.score or 0.0)
            if score < self.cfg.min_glossary_similarity:
                continue
            md = hit.node.metadata
            is_term = md.get("record_type") == "business_term"
            # current label and description from the business glossary (the indexed text can be older)
            row = self.db.query_one(f"SELECT label, description FROM {'business_terms' if is_term else 'business_entities'}"
                                    " WHERE slug = ? AND deleted = 0", (md.get("slug"),))
            if row is None:
                continue  # deleted from the business glossary since it was indexed
            out.append({
                "rank": len(out) + 1,
                "type": md.get("record_type"),
                "label": row["label"],
                "description": row["description"],
                "score": round(score, 4),
                "score_type": "cosine_similarity",
                "catalog_ref": md.get("catalog_ref"),
                "source_id": md.get("source_id"),
                "dataset_id": md.get("dataset_id"),
                "dataset": md.get("dataset"),
                "column_id": md.get("column_id") or None,
                "column": md.get("column") or None,
                "graph_ref": md.get("graph_ref"),
                "source_ref": f"/api/datasets/{md['dataset_id']}" if md.get("dataset_id") else None,
                "ref": f"/api/business-{'terms' if is_term else 'entities'}/{md.get('slug')}",
            })
        return out

    # -- graph -----------------------------------------------------------------------------------
    def _seeds(self, query: str, glossary: list[dict[str, Any]], documents: list[dict[str, Any]]) -> list[URIRef]:
        seeds: list[URIRef] = list(self.graph.concepts_in_text(query))
        for g in glossary:
            ref = URIRef(g["graph_ref"])
            if ref not in seeds:
                seeds.append(ref)
        chunk_ids = [d["chunk_id"] for d in documents[:3]]
        if chunk_ids:
            marks = ",".join("?" * len(chunk_ids))
            for r in self.db.query(f"SELECT DISTINCT entity_slug FROM doc_entity_mentions WHERE chunk_id IN ({marks})",
                                   chunk_ids):
                ref = entity_uri(r["entity_slug"])
                if ref not in seeds:
                    seeds.append(ref)
        return seeds[:30]


def _merge(queries: list[str], results: list[dict[str, Any]], field: str, key, by_score: bool) -> list[dict[str, Any]]:
    """Union of one result list across queries: one item per key, best score kept, ``matched_queries`` added."""
    merged: dict[str, dict[str, Any]] = {}
    for query, res in zip(queries, results):
        for item in res[field]:
            k = key(item)
            current = merged.get(k)
            if current is None:
                merged[k] = {**item, "matched_queries": [query]}
                continue
            current["matched_queries"].append(query)
            if by_score and item.get("score", 0) > current.get("score", 0):
                merged[k] = {**item, "matched_queries": current["matched_queries"]}
    items = list(merged.values())
    if by_score:
        items.sort(key=lambda x: -(x.get("score") or 0))
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
    return items

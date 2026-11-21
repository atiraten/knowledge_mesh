"""``query_knowledge_graph``: graph-aware retrieval for one planner intent (GraphRAG, local search).

Steps for one request (``KGQueryRequest``):

1. **Resolve** each retrieval query (and the intent) to business entities, business terms and named
   relationships: stable IRI in the text, then exact / normalized / alternate-label matches (plural- and
   word-order-insensitive), then, when ``knowledge_graph.embedding_retrieval_enabled`` and no term resolved,
   the existing glossary vector index for looser phrasings (limited to the resolved entities' terms). Every hit keeps ``match_type`` and ``matched_queries``;
   hits found by several queries are merged by IRI. Not every query token has to resolve.
2. **Traverse** a semantic adjacency index (rebuilt per graph version) between every ordered pair of
   resolved entities: depth-first, simple paths only (no node twice in a path), bounded by
   ``max_hops``, ``max_candidate_paths``, ``max_expansions`` and ``query_timeout_seconds``. Edges:
   named relationships in their own direction (reverse only with an inverse name), FK-derived
   navigational edges (``references`` / reverse ``isReferencedBy``) and document relationships
   (forward only). Rejected edges are never used; inferred ones only with
   ``include_inferred_relationships``.
3. **Rank** paths: ``semantic * mean(edge relevance) + coverage * join coverage + status * mean(status
   weight)``. Edge relevance is 1 for an alias match of the relationship's name or synonyms in the
   intent, otherwise the embedding similarity of the intent and the relationship text. Join coverage:
   verified mapping 1, inferred 0.5, none 0. Status weight: verified 1, inferred 0.6, navigational 0.3.
   Means keep the score length-neutral; fewer hops only break ties. Paths using FK-derived navigational
   edges rank after paths that use none (fewer navigational edges first). Scores are uncalibrated.
4. **Expand** the top paths and resolved concepts with definitions, aliases, mapped tables/columns,
   ordered FK column pairs, join predicates, keys and term relations, plus references to the definitions
   and passages that mention them (IDs only; documents stay in ``retrieve_context``).

The result describes schema-level connectivity (``assertion_level: "schema"``): it says how data could
be joined, never that a particular record exists. Missing mappings are reported, never invented.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rdflib import URIRef

from ..common.errors import InvalidArgument, NotConfigured
from ..observability import capture_sensitive, get_metrics, span
from .ontology import LAYERS, MESH, PROV, RDF, RDFS, SKOS
from .store import KnowledgeGraph

log = logging.getLogger(__name__)
RECORD_NOTE = ("Schema-level knowledge: paths describe how data can be joined, not facts about particular records. "
               "Record-level claims require rows returned by execute_sql.")
_STOP = {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "by", "with", "at", "per", "is", "are", "was",
         "were", "be", "who", "which", "what", "that", "list", "show", "find", "get", "all", "me", "their", "its"}
_IRI = re.compile(r"urn:mesh:[a-z-]+:[A-Za-z0-9_.:-]+")
_STATUS_WEIGHT = {"verified": 1.0, "inferred": 0.6}
_NAVIGATIONAL_WEIGHT = 0.3
_COVERAGE = {"verified": 1.0, "inferred": 0.5}


class KGQueryRequest(BaseModel):
    """Request of ``query_knowledge_graph`` (spec 3.3)."""

    model_config = ConfigDict(extra="forbid")
    intent_id: str | None = Field(default=None, max_length=64,
                                  description="Planner intent id (e.g. I1); echoed back for correlation.")
    intent: str | None = Field(default=None, max_length=2000,
                               description="The intent's natural-language business objective.")
    retrieval_queries: list[str] = Field(default_factory=list, max_length=5,
                                         description="Alternative search formulations of the intent (1-5).")
    max_hops: int | None = Field(default=None, ge=1, le=20,
                                 description="Maximum relationship traversals per path (default 5).")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Maximum ranked paths returned (default 5).")
    include_physical_mappings: bool = Field(default=True,
                                            description="Include tables, columns and join column pairs.")
    include_related_terms: bool = Field(default=True,
                                        description="Include term relations and the terms of join columns.")

    @model_validator(mode="after")
    def _needs_text(self) -> "KGQueryRequest":
        self.retrieval_queries = [" ".join(q.split())[:2000] for q in self.retrieval_queries if q and q.strip()]
        if not self.retrieval_queries and not (self.intent or "").strip():
            raise ValueError("intent or retrieval_queries is required")
        return self


# -- graph snapshot used for one graph version ------------------------------------------------------
@dataclass
class _Edge:
    iri: str
    kind: str                 # semantic | navigational | document
    source: str               # entity IRIs
    target: str
    name: str
    synonyms: list[str]
    definition: str | None
    inverse_name: str | None
    status: str               # verified | inferred | rejected
    joins: list[dict[str, Any]]   # {fk, method, direction, status}
    provenance: dict[str, Any]
    sources: set[str] = field(default_factory=set)      # structured sources of its join FKs
    documents: set[str] = field(default_factory=set)    # documents of its passages

    def text(self) -> str:
        words = _camel_words(self.name)
        parts = [words] + ([f"synonyms: {', '.join(self.synonyms)}"] if self.synonyms else []) + \
            ([self.definition] if self.definition else [])
        return ". ".join(parts)


@dataclass
class _Snapshot:
    entities: dict[str, dict[str, Any]]
    terms: dict[str, dict[str, Any]]
    tables: dict[str, dict[str, Any]]
    columns: dict[str, dict[str, Any]]
    keys: dict[str, dict[str, Any]]
    edges: dict[str, _Edge]
    adjacency: dict[str, list[tuple[_Edge, str]]]     # entity IRI -> (edge, forward|reverse)
    term_relations: dict[str, dict[str, Any]]
    labels: list[tuple[tuple[str, ...], str, str, str]]   # (stemmed tokens, iri, kind, label|alias)
    definitions: dict[str, list[dict[str, Any]]]      # concept IRI -> definitions describing it
    mentions: dict[str, list[dict[str, Any]]]         # entity IRI -> passages mentioning it
    version: int
    published_at: str


def _s(v) -> str | None:
    return None if v is None else str(v)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _camel_words(name: str) -> str:
    return " ".join(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "").replace("_", " ").split()).lower()


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _norm(_camel_words(text) if re.search(r"[a-z][A-Z]", text or "") else text).split()]


def _content(tokens: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(t for t in tokens if t not in _STOP)


def build_snapshot(graph: KnowledgeGraph) -> _Snapshot:
    ds = graph.dataset
    with graph._lock:
        val = ds.value
        # tables and columns
        tables: dict[str, dict[str, Any]] = {}
        for t in set(ds.subjects(RDF.type, MESH.Dataset)):
            tables[str(t)] = {"iri": str(t), "dataset_id": _s(val(t, MESH.datasetId)), "name": _s(val(t, RDFS.label)),
                              "schema": _s(val(t, MESH.schemaName)), "source_id": _s(val(t, MESH.sourceId)),
                              "catalog_ref": _s(val(t, MESH.catalogRef)), "entities": {}}
        columns: dict[str, dict[str, Any]] = {}
        for c in set(ds.subjects(RDF.type, MESH.Column)):
            columns[str(c)] = {"iri": str(c), "column_id": _s(val(c, MESH.columnId)), "name": _s(val(c, RDFS.label)),
                               "table": _s(val(c, MESH.inDataset)), "data_type": _s(val(c, MESH.dataType)),
                               "catalog_ref": _s(val(c, MESH.catalogRef)), "terms": {}}
        for em in set(ds.subjects(RDF.type, MESH.EntityMapping)):
            t, e = _s(val(em, MESH.dataset)), _s(val(em, MESH.entity))
            if t in tables and e:
                tables[t]["entities"][e] = _s(val(em, MESH.validationStatus)) or "inferred"
        for t, e in ds.subject_objects(MESH.representsEntity):  # earlier layout: no mapping record
            if str(t) in tables:
                tables[str(t)]["entities"].setdefault(str(e), "inferred")
        # keys
        keys: dict[str, dict[str, Any]] = {}
        for k in set(ds.subjects(RDF.type, MESH.KeyConstraint)):
            types = {str(x).rsplit(":", 1)[-1] for x in ds.objects(k, RDF.type)}
            ktype = "foreign" if "ForeignKey" in types else ("primary" if "PrimaryKey" in types else "unique")
            origin = _s(val(k, MESH.origin))
            members = sorted((int(val(m, MESH.position)), _s(val(m, MESH.column))) for m in ds.objects(k, MESH.keyColumn))
            pairs = sorted((int(val(m, MESH.position)), _s(val(m, MESH.fromColumn)), _s(val(m, MESH.toColumn)))
                           for m in ds.objects(k, MESH.columnMapping))
            keys[str(k)] = {"iri": str(k), "type": ktype, "table": _s(val(k, MESH.onDataset)),
                            "references": _s(val(k, MESH.referencesDataset)), "origin": origin,
                            "status": _s(val(k, MESH.validationStatus)) or ("verified" if origin == "declared"
                                                                            else "inferred"),
                            "confidence": _s(val(k, MESH.fkConfidence)), "columns": [c for _, c in members],
                            "pairs": [(f, t) for _, f, t in pairs],
                            "pk_candidate": bool(val(k, MESH.primaryKeyCandidate) and
                                                 val(k, MESH.primaryKeyCandidate).toPython())}
        declared_sig = {(k["table"], k["references"], tuple(k["pairs"])) for k in keys.values()
                        if k["type"] == "foreign" and k["origin"] == "declared"}
        for k in keys.values():
            k["duplicate_of_declared"] = (k["type"] == "foreign" and k["origin"] == "inferred"
                                          and (k["table"], k["references"], tuple(k["pairs"])) in declared_sig)
        # entities and terms
        definitions: dict[str, list[dict[str, Any]]] = {}
        for d, concept in ds.subject_objects(MESH.describes):
            definitions.setdefault(str(concept), []).append({
                "iri": str(d), "name": _s(val(d, RDFS.label)), "text": _s(val(d, MESH.definitionText)),
                "definition_id": str(d).rsplit(":", 1)[-1],
                "ref": f"/api/business-definitions/{str(d).rsplit(':', 1)[-1]}"})
        entities: dict[str, dict[str, Any]] = {}
        for e in set(ds.subjects(RDF.type, MESH.BusinessEntity)):
            types = sorted({str(x) for x in ds.objects(e, RDF.type)})
            entities[str(e)] = {"iri": str(e), "name": _s(val(e, SKOS.prefLabel)) or _s(val(e, RDFS.label)),
                                "alt_names": sorted({str(x) for x in ds.objects(e, SKOS.altLabel)}),
                                "definitions": sorted({str(x) for x in ds.objects(e, SKOS.definition)}),
                                "classification": ["owl:Class" if t.endswith("#Class") else
                                                   "mesh:" + t.rsplit(":", 1)[-1] for t in types],
                                "tables": {}, "terms": []}
        for tiri, t in tables.items():
            for e, status in t["entities"].items():
                if e in entities:
                    entities[e]["tables"][tiri] = status
        terms: dict[str, dict[str, Any]] = {}
        for t in set(ds.subjects(RDF.type, MESH.BusinessTerm)):
            ents = sorted(str(x) for x in ds.objects(t, MESH.termOfEntity))
            terms[str(t)] = {"iri": str(t), "name": _s(val(t, SKOS.prefLabel)) or _s(val(t, RDFS.label)),
                             "alt_names": sorted({str(x) for x in ds.objects(t, SKOS.altLabel)}),
                             "definitions": sorted({str(x) for x in ds.objects(t, SKOS.definition)}),
                             "entities": ents, "property": _s(val(t, MESH.termProperty)), "columns": {}}
            for e in ents:
                if e in entities:
                    entities[e]["terms"].append(str(t))
        for b in set(ds.subjects(RDF.type, MESH.TermBinding)):
            t, c = _s(val(b, MESH["term"])), _s(val(b, MESH.column))
            if t in terms and c in columns:
                status = _s(val(b, MESH.validationStatus)) or "inferred"
                terms[t]["columns"][c] = status
                columns[c]["terms"][t] = status
        # relationships (edges)
        edges: dict[str, _Edge] = {}
        for r in set(ds.subjects(RDF.type, MESH.Relationship)):
            src, dst = _s(val(r, MESH.sourceEntity)), _s(val(r, MESH.targetEntity))
            if src not in entities or dst not in entities:
                continue
            rtypes = set(ds.objects(r, RDF.type))
            derived = list(ds.objects(r, PROV.wasDerivedFrom))
            status = _s(val(r, MESH.validationStatus))
            joins: list[dict[str, Any]] = []
            prov: dict[str, Any] = {"method": _s(val(r, MESH.extractionMethod))}
            docs: set[str] = set()
            if MESH.SemanticRelationship in rtypes:
                kind = "semantic"
                status = status or "inferred"
                for jm in ds.objects(r, MESH.joinMapping):
                    fk = _s(val(jm, MESH.foreignKey))
                    if fk in keys and keys[fk]["duplicate_of_declared"]:
                        continue
                    joins.append({"iri": str(jm), "fk": fk, "method": _s(val(jm, MESH.mappingMethod)),
                                  "direction": _s(val(jm, MESH.keyDirection)),
                                  "status": _s(val(jm, MESH.validationStatus)) or "inferred",
                                  "fk_status": keys[fk]["status"] if fk in keys else None})
                run = val(r, PROV.wasGeneratedBy)
                prov.update(kind="relationship_proposal", run=_s(val(run, MESH.runId)) if run is not None else None)
            elif MESH.NavigationalEdge in rtypes or any((d, RDF.type, MESH.KeyConstraint) in ds for d in derived):
                kind = "navigational"
                fk = next((str(d) for d in derived if str(d) in keys), None)
                if fk and keys[fk]["duplicate_of_declared"]:
                    continue  # the declared key with the same column pairs already gives this edge
                if fk:
                    status = status or keys[fk]["status"]
                    joins.append({"iri": fk, "fk": fk, "method": "foreign_key", "direction": "forward",
                                  "status": keys[fk]["status"], "fk_status": keys[fk]["status"]})
                prov.update(kind="catalog_key", key_ref=fk)
                status = status or "inferred"
            else:
                kind = "document"
                status = status or "inferred"
                for d in derived:
                    if (d, RDF.type, MESH.Passage) in ds:
                        doc = _s(val(d, MESH.documentId))
                        docs.add(doc or "")
                        prov.update(kind="document_passage", document_id=doc, chunk_id=_s(val(d, MESH.chunkId)))
            sources = {tables[keys[j["fk"]]["table"]]["source_id"] for j in joins
                       if j["fk"] in keys and keys[j["fk"]]["table"] in tables}
            edges[str(r)] = _Edge(
                iri=str(r), kind=kind, source=src, target=dst, name=_s(val(r, MESH.relationshipType)) or "related",
                synonyms=sorted({str(x) for x in ds.objects(r, SKOS.altLabel)}),
                definition=_s(val(r, SKOS.definition)) or _s(val(r, MESH.evidenceText)),
                inverse_name=_s(val(r, MESH.inverseName)), status=status, joins=joins, provenance=prov,
                sources=sources, documents=docs)
        adjacency: dict[str, list[tuple[_Edge, str]]] = {}
        # explored in this order (spec 6.8): verified before inferred, semantic before document before FK-only edges
        kind_rank = {"semantic": 0, "document": 1, "navigational": 2}
        for e in sorted(edges.values(), key=lambda x: (x.status != "verified", kind_rank[x.kind], x.iri)):
            if e.status == "rejected":
                continue
            adjacency.setdefault(e.source, []).append((e, "forward"))
            if e.kind == "navigational" or (e.kind == "semantic" and e.inverse_name):
                adjacency.setdefault(e.target, []).append((e, "reverse"))
        # term relations
        term_relations: dict[str, dict[str, Any]] = {}
        for tr in set(ds.subjects(RDF.type, MESH.TermRelation)):
            subj, obj, pred = _s(val(tr, MESH.subjectTerm)), _s(val(tr, MESH.objectTerm)), _s(val(tr, MESH.predicate))
            if subj in terms and obj in terms and pred:
                term_relations[str(tr)] = {
                    "iri": str(tr), "subject_iri": subj, "subject": terms[subj]["name"],
                    "predicate": pred.rsplit(":", 1)[-1], "predicate_iri": pred, "object_iri": obj,
                    "object": terms[obj]["name"], "definition": _s(val(tr, SKOS.definition)),
                    "status": _s(val(tr, MESH.validationStatus)) or "inferred",
                    "assertion_level": "descriptive",
                    "note": "Describes expected meaning only; not validated against records."}
        # mentions in document passages
        mentions: dict[str, list[dict[str, Any]]] = {}
        for e, p in ds.subject_objects(MESH.mentionedIn):
            if str(e) in entities:
                chunk = _s(val(p, MESH.chunkId))
                mentions.setdefault(str(e), []).append({"type": "document_passage", "chunk_id": chunk,
                                                        "document_id": _s(val(p, MESH.documentId)),
                                                        "ref": f"/api/chunks/{chunk}"})
        # label index: (tokens, iri, kind, match)
        labels: list[tuple[tuple[str, ...], str, str, str]] = []
        for kind, items in (("entity", entities), ("term", terms)):
            for iri, item in items.items():
                if item["name"]:
                    labels.append((tuple(_tokens(item["name"])), iri, kind, "label"))
                for alt in item["alt_names"]:
                    labels.append((tuple(_tokens(alt)), iri, kind, "alias"))
        for e in edges.values():
            if e.kind != "navigational" and e.status != "rejected":
                labels.append((tuple(_tokens(_camel_words(e.name))), e.iri, "relationship", "label"))
                if e.inverse_name:
                    labels.append((tuple(_tokens(_camel_words(e.inverse_name))), e.iri, "relationship", "alias"))
                for syn in e.synonyms:
                    labels.append((tuple(_tokens(syn)), e.iri, "relationship", "alias"))
        labels = [x for x in labels if x[0]]
        return _Snapshot(entities, terms, tables, columns, keys, edges, adjacency, term_relations, labels,
                         definitions, mentions, graph.version, graph.published_at)


@dataclass
class _Hit:
    iri: str
    kind: str
    match_type: str
    score: float
    queries: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)


class KnowledgeGraphRetriever:
    def __init__(self, settings, graph: KnowledgeGraph, embed_model=None,
                 glossary_search: Callable[..., list[dict[str, Any]]] | None = None):
        self.cfg = settings
        self.graph = graph
        self.embed_model = embed_model
        self.glossary_search = glossary_search

    # -- public ------------------------------------------------------------------------------------
    def query(self, request: KGQueryRequest | dict[str, Any], structured_source_ids: list[str] | None = None,
              dataset_ids: list[str] | None = None, document_ids: list[str] | None = None) -> dict[str, Any]:
        """Run one request. Never raises for a bad graph state: errors are ``status: "error"`` results."""
        req = request if isinstance(request, KGQueryRequest) else KGQueryRequest.model_validate(request)
        if not self.cfg.enabled:
            raise NotConfigured("the knowledge graph tool is disabled (knowledge_graph.enabled is false)")
        max_hops = req.max_hops or self.cfg.default_max_hops
        top_k = req.top_k or self.cfg.default_top_k
        if max_hops > self.cfg.max_hops_limit:
            raise InvalidArgument(f"max_hops must be at most {self.cfg.max_hops_limit}", {"max_hops": max_hops})
        if top_k > self.cfg.max_top_k:
            raise InvalidArgument(f"top_k must be at most {self.cfg.max_top_k}", {"top_k": top_k})
        queries = list(dict.fromkeys(req.retrieval_queries)) or [(req.intent or "").strip()]
        intent = (req.intent or "").strip() or queries[0]
        metrics = get_metrics()
        start = time.perf_counter()
        attrs = {"kg.intent_id": req.intent_id or "", "kg.query_count": len(queries), "kg.max_hops": max_hops,
                 "kg.top_k": top_k}
        if capture_sensitive():
            attrs["input.value"] = json.dumps({"intent": intent, "retrieval_queries": queries})[:2000]
        with span("graph.query", "RETRIEVER", attrs) as sp:
            try:
                out = self._run(req, intent, queries, max_hops, top_k, structured_source_ids, dataset_ids,
                                document_ids, sp)
            except (InvalidArgument, NotConfigured):
                raise
            except Exception as exc:  # safe diagnostic; the graph itself is untouched
                log.exception("knowledge graph query failed")
                out = self._empty(req, "error", queries)
                out["error"] = {"code": "graph_query_failed",
                                "message": f"knowledge graph query failed ({type(exc).__name__})"}
            elapsed = time.perf_counter() - start
            sp.set_attribute("kg.status", out["status"])
            sp.set_attribute("kg.duration_ms", int(elapsed * 1000))
            sp.set_attribute("kg.warnings", len(out.get("warnings") or []))
        metrics.graph_query_duration.record(elapsed, {"status": out["status"]})
        metrics.graph_queries.add(1, {"status": out["status"]})
        return self._fit(out)

    # -- implementation ----------------------------------------------------------------------------
    def _empty(self, req: KGQueryRequest, status: str, queries: list[str]) -> dict[str, Any]:
        return {"intent_id": req.intent_id, "status": status, "queries": queries, "matched_queries": [],
                "business_entities": [], "business_terms": [], "relationships": [], "paths": [],
                "physical_mappings": {"tables": [], "columns": [], "joins": []}, "constraints": [],
                "evidence": [], "warnings": [], "ambiguities": [], "assertion_level": "schema",
                "note": RECORD_NOTE, "graph": {"version": self.graph.version, "published_at": self.graph.published_at,
                                               "named_graphs": []}}

    def _run(self, req, intent, queries, max_hops, top_k, sources, dataset_ids, document_ids, sp) -> dict[str, Any]:
        snap: _Snapshot = self.graph.cached("kg_snapshot", lambda: build_snapshot(self.graph))
        scope = _Scope(snap, sources, dataset_ids, document_ids)
        out = self._empty(req, "ok", queries)
        out["graph"]["version"], out["graph"]["published_at"] = snap.version, snap.published_at
        warnings: list[str] = out["warnings"]
        # 1. resolve
        hits: dict[str, _Hit] = {}
        lookup_types: set[str] = set()
        for q in queries + ([intent] if intent not in queries else []):
            found = self._resolve(snap, q, scope, out["ambiguities"])
            for h in found:
                lookup_types.add(h.match_type)
                cur = hits.get(h.iri)
                if cur is None:
                    hits[h.iri] = h
                    cur = h
                else:
                    if h.score > cur.score:
                        cur.score, cur.match_type = h.score, h.match_type
                    cur.phrases = list(dict.fromkeys(cur.phrases + h.phrases))
                if q in queries and q not in cur.queries:
                    cur.queries.append(q)
            if q in queries and not found:
                warnings.append(f"No business entity, term or relationship matched the query {q!r}.")
        matched = [q for q in queries if any(q in h.queries for h in hits.values())]
        out["matched_queries"] = matched
        entity_hits = [h for h in hits.values() if h.kind == "entity"]
        term_hits = [h for h in hits.values() if h.kind == "term"]
        rel_hits = {h.iri for h in hits.values() if h.kind == "relationship"}
        sp.set_attribute("kg.lookup_types", sorted(lookup_types))
        sp.set_attribute("kg.matched_entities", len(entity_hits))
        sp.set_attribute("kg.matched_terms", len(term_hits))
        if not hits:
            out["status"] = "no_match"
            return out
        # 2. traverse between resolved entities (and the endpoints of alias-matched relationships)
        endpoints = [h.iri for h in sorted(entity_hits, key=lambda h: (-h.score, h.iri))][:8]
        if len(endpoints) < 2:  # one entity (or none): the relationships named in the intent give the other end
            for iri in sorted(rel_hits):
                e = snap.edges[iri]
                for node in (e.source, e.target):
                    if node not in endpoints and scope.entity_visible(node):
                        endpoints.append(node)
        candidates, bound = self._paths(snap, endpoints, max_hops, scope)
        if bound:
            warnings.append(bound)
        sp.set_attribute("kg.paths_considered", len(candidates))
        get_metrics().graph_paths.record(len(candidates))
        if len(endpoints) >= 2 and not candidates:
            names = ", ".join(snap.entities[e]["name"] for e in endpoints[:4])
            warnings.append(f"No relationship path connects {names} within {max_hops} hops.")
        # 3. rank
        ranked = self._rank(snap, candidates, intent, queries, rel_hits)[:top_k]
        sp.set_attribute("kg.paths_returned", len(ranked))
        sp.set_attribute("kg.max_path_hops", max((len(r["path"]) for r in ranked), default=0))
        # 4. expand
        self._expand(snap, out, hits, ranked, rel_hits, req, scope)
        if not (out["business_entities"] or out["business_terms"] or out["paths"] or out["relationships"]):
            out["status"] = "no_match"
        out["graph"]["named_graphs"] = sorted(
            {f"urn:mesh:context:source:{t['source_id']}:{layer}" for t in out["physical_mappings"]["tables"]
             for layer in LAYERS if t.get("source_id")})
        return out

    # -- resolution --------------------------------------------------------------------------------
    def _resolve(self, snap: _Snapshot, text: str, scope: "_Scope", ambiguities: list) -> list[_Hit]:
        hits: dict[str, _Hit] = {}
        spans: dict[str, list[tuple[int, int]]] = {}   # contiguous matches: token spans in the query

        def visible(iri: str, kind: str) -> bool:
            if kind == "entity":
                return scope.entity_visible(iri)
            if kind == "term":
                return scope.term_visible(snap, iri)
            return scope.edge_visible(snap.edges[iri])

        def add(iri: str, kind: str, match: str, score: float, phrase: str) -> None:
            if not visible(iri, kind):
                return
            cur = hits.get(iri)
            if cur is None or score > cur.score:
                hits[iri] = _Hit(iri, kind, match, score, phrases=[phrase])

        for iri in _IRI.findall(text or ""):
            iri = iri.rstrip(".,;:")
            for kind, table in (("entity", snap.entities), ("term", snap.terms), ("relationship", snap.edges)):
                if iri in table:
                    add(iri, kind, "iri", 1.0, iri)
        q_tokens = _tokens(text)
        q_norm = " ".join(q_tokens)
        q_content = set(_content(q_tokens))
        by_phrase: dict[tuple[str, ...], set[tuple[str, str]]] = {}
        for tokens, iri, kind, match in snap.labels:
            phrase = " ".join(tokens)
            content = _content(tokens)
            starts = _occurrences(q_tokens, list(tokens))
            if q_norm == phrase:
                kind_match, score = ("exact" if match == "label" else "alias"), 1.0
            elif starts:
                kind_match, score = ("label" if match == "label" else "alias"), 0.95
            elif len(content) >= 2 and set(content) <= q_content:
                kind_match, score = "token_set", 0.85
            else:
                continue
            if not visible(iri, kind):
                continue
            add(iri, kind, kind_match, score + 0.001 * len(tokens), phrase)
            spans.setdefault(iri, []).extend((i, i + len(tokens)) for i in starts)
            by_phrase.setdefault(tokens, set()).add((iri, kind))
        # a shorter contiguous match lying entirely inside a longer one of the same kind is dropped
        # ("order" inside "order line"), unless the shorter phrase also occurs elsewhere in the query
        for iri, own in list(spans.items()):
            h = hits.get(iri)
            if h is None or h.match_type not in ("label", "alias") or not own:
                continue
            longer = [sp for other, sps in spans.items() if other != iri and hits.get(other)
                      and hits[other].kind == h.kind for sp in sps]
            if all(any(a <= s0 and e0 <= b and (b - a) > (e0 - s0) for a, b in longer) for s0, e0 in own):
                hits.pop(iri, None)
        # one phrase naming several concepts of the same kind is ambiguous (e.g. a synonym shared by two terms)
        for tokens, items in by_phrase.items():
            for kind in ("entity", "term"):
                same = sorted(iri for iri, k in items if k == kind and iri in hits)
                if len(same) > 1:
                    table = snap.entities if kind == "entity" else snap.terms
                    entry = {"query": text, "phrase": " ".join(tokens), "type": kind,
                             "candidates": [{"iri": i, "name": table[i]["name"]} for i in same]}
                    if entry not in ambiguities:
                        ambiguities.append(entry)
        # looser phrasings ("born" for "Date Of Birth"): the glossary vector index, when no term resolved
        # deterministically. With resolved entities, only their own terms are taken from it.
        entities = {h.iri for h in hits.values() if h.kind == "entity"}
        covered = {i for iri, sps in spans.items() if iri in hits for a, b in sps for i in range(a, b)}
        leftover = [t for i, t in enumerate(q_tokens) if i not in covered and t not in _STOP and not t.isdigit()]
        if self.cfg.embedding_retrieval_enabled and self.glossary_search is not None and leftover \
                and not any(h.kind == "term" for h in hits.values()):
            try:
                taken, best = 0, None
                for g in self.glossary_search(text, 8, scope.sources, scope.datasets):
                    iri = g.get("graph_ref")
                    kind = {"business_term": "term", "business_entity": "entity"}.get(g.get("type"))
                    score = float(g.get("score") or 0)
                    if not iri or kind is None or iri in hits or taken >= 2 or (best is not None and score < best - 0.05):
                        continue
                    if kind == "entity" and entities:
                        continue  # entities already resolved by name
                    if kind == "term" and (iri not in snap.terms or (
                            entities and not set(snap.terms[iri]["entities"]) & entities)):
                        continue
                    if kind == "entity" and iri not in snap.entities:
                        continue
                    add(iri, kind, "semantic", round(score * 0.8, 4), g.get("label") or iri)
                    best = score if best is None else best
                    taken += 1
            except Exception as exc:  # the vector index is optional for resolution
                log.warning("glossary lookup failed during graph resolution", extra={"error": type(exc).__name__})
        return list(hits.values())

    # -- traversal ---------------------------------------------------------------------------------
    def _paths(self, snap: _Snapshot, endpoints: list[str], max_hops: int, scope: "_Scope"):
        deadline = time.monotonic() + self.cfg.query_timeout_seconds
        budget = {"expansions": 0}
        paths: list[list[tuple[_Edge, str]]] = []
        bound: str | None = None
        targets = set(endpoints)

        def usable(edge: _Edge) -> bool:
            if edge.status == "rejected":
                return False
            if edge.status != "verified" and not self.cfg.include_inferred_relationships:
                return False
            return scope.edge_visible(edge)

        for start in endpoints:
            stack: list[tuple[str, list[tuple[_Edge, str]], frozenset[str]]] = [(start, [], frozenset([start]))]
            while stack:
                node, path, visited = stack.pop()
                if path and node in targets and node != start:
                    paths.append(path)
                    if len(paths) >= self.cfg.max_candidate_paths:
                        return paths, (f"Traversal stopped at max_candidate_paths ({self.cfg.max_candidate_paths}); "
                                       "more paths may exist.")
                if len(path) >= max_hops:
                    continue
                budget["expansions"] += 1
                if budget["expansions"] > self.cfg.max_expansions:
                    return paths, f"Traversal stopped at max_expansions ({self.cfg.max_expansions})."
                if time.monotonic() > deadline:
                    return paths, f"Traversal stopped at the {self.cfg.query_timeout_seconds}s timeout."
                for edge, direction in reversed(snap.adjacency.get(node, [])):
                    nxt = edge.target if direction == "forward" else edge.source
                    if nxt in visited or not usable(edge) or not scope.entity_visible(nxt):
                        continue
                    stack.append((nxt, path + [(edge, direction)], visited | {nxt}))
        return paths, bound

    # -- ranking -----------------------------------------------------------------------------------
    def _rank(self, snap: _Snapshot, paths, intent: str, queries: list[str], rel_hits: set[str]) -> list[dict]:
        if not paths:
            return []
        weights = self.cfg.ranking_weights
        relevance = self._edge_relevance(snap, {e.iri for p in paths for e, _ in p}, intent, queries, rel_hits)
        ranked = []
        for p in paths:
            rel = [relevance.get(e.iri, 0.0) for e, _ in p]
            cov = [_join_coverage(e) for e, _ in p]
            stat = [_NAVIGATIONAL_WEIGHT if e.kind == "navigational" else _STATUS_WEIGHT.get(e.status, 0.6)
                    for e, _ in p]
            semantic, coverage, status = sum(rel) / len(rel), sum(cov) / len(cov), sum(stat) / len(stat)
            score = weights.semantic * semantic + weights.coverage * coverage + weights.status * status
            navigational = sum(1 for e, _ in p if e.kind == "navigational")
            ranked.append({"path": p, "score": round(score, 4), "navigational": navigational, "components": {
                "semantic_relevance": round(semantic, 4), "join_coverage": round(coverage, 4),
                "status": round(status, 4), "hops": len(p), "navigational_edges": navigational,
                "weights": {"semantic": weights.semantic, "coverage": weights.coverage, "status": weights.status},
                "edge_relevance": [round(x, 4) for x in rel],
                "rank_rule": "paths without navigational (FK-only) edges first, then fewer navigational edges, "
                             "then higher score, then fewer hops"}})
        # FK-derived navigational edges rank below semantic relationships (they carry no business semantics)
        ranked.sort(key=lambda r: (r["navigational"], -r["score"], len(r["path"]), [e.iri for e, _ in r["path"]]))
        return ranked

    def _edge_relevance(self, snap: _Snapshot, edge_iris: set[str], intent: str, queries: list[str],
                        rel_hits: set[str]) -> dict[str, float]:
        out = {iri: 1.0 for iri in edge_iris if iri in rel_hits}  # alias / name match in the intent
        rest = [iri for iri in sorted(edge_iris) if iri not in out]
        if not rest or not self.cfg.embedding_retrieval_enabled or self.embed_model is None:
            return {**{iri: 0.0 for iri in rest}, **out}
        cache: dict[str, list[float]] = self.graph.cached("kg_edge_vectors", dict)
        missing = [iri for iri in rest if iri not in cache]
        if missing:
            texts = [f"{snap.entities[snap.edges[i].source]['name']} {snap.edges[i].text()} "
                     f"{snap.entities[snap.edges[i].target]['name']}" for i in missing]
            for iri, vec in zip(missing, self.embed_model.get_text_embedding_batch(texts)):
                cache[iri] = vec
        qvecs = [self.embed_model.get_query_embedding(t) for t in dict.fromkeys([intent] + queries)]
        for iri in rest:
            out[iri] = max(0.0, max(_cosine(q, cache[iri]) for q in qvecs))
        return out

    # -- expansion ---------------------------------------------------------------------------------
    def _expand(self, snap: _Snapshot, out: dict[str, Any], hits: dict[str, _Hit], ranked: list[dict],
                rel_hits: set[str], req: KGQueryRequest, scope: "_Scope") -> None:
        warnings = out["warnings"]
        entity_iris: list[str] = [h.iri for h in sorted(hits.values(), key=lambda h: (-h.score, h.iri))
                                  if h.kind == "entity"]
        term_iris: list[str] = [h.iri for h in sorted(hits.values(), key=lambda h: (-h.score, h.iri))
                                if h.kind == "term"]
        joins: dict[tuple[str, str], dict[str, Any]] = {}
        table_iris: list[str] = []
        column_iris: list[str] = []

        def add_join(edge: _Edge, j: dict[str, Any]) -> str | None:
            fk = snap.keys.get(j["fk"] or "")
            if fk is None or not fk["pairs"] or not scope.key_visible(snap, fk) or j["status"] == "rejected":
                return None
            key = (edge.iri, fk["iri"])
            if key not in joins:
                frm, to = snap.tables.get(fk["table"]), snap.tables.get(fk["references"])
                pairs = []
                for pos, (a, b) in enumerate(fk["pairs"], start=1):
                    ca, cb = snap.columns.get(a, {}), snap.columns.get(b, {})
                    pairs.append({"position": pos, "from_column": ca.get("name"), "to_column": cb.get("name"),
                                  "from_column_iri": a, "to_column_iri": b})
                    column_iris.extend(x for x in (a, b) if x not in column_iris)
                fq = lambda t: ".".join(x for x in (t.get("schema"), t.get("name")) if x) if t else "?"  # noqa: E731
                predicate = " AND ".join(f"{fq(frm)}.{p['from_column']} = {fq(to)}.{p['to_column']}" for p in pairs)
                joins[key] = {"join_id": f"J{len(joins) + 1}", "foreign_key_iri": fk["iri"],
                              "relationship_iri": edge.iri, "relationship": edge.name, "method": j["method"],
                              "direction": j["direction"], "from_table": frm["name"] if frm else None,
                              "from_table_iri": fk["table"], "to_table": to["name"] if to else None,
                              "to_table_iri": fk["references"], "column_pairs": pairs, "join_predicate": predicate,
                              "origin": fk["origin"], "status": j["status"], "foreign_key_status": fk["status"],
                              "confidence": fk["confidence"]}
                table_iris.extend(x for x in (fk["table"], fk["references"]) if x and x not in table_iris)
            return joins[key]["join_id"]

        # paths
        for n, r in enumerate(ranked, start=1):
            pid = f"P{n}"
            nodes, edges_out, missing = [], [], []
            p = r["path"]
            for i, (edge, direction) in enumerate(p):
                a = edge.source if direction == "forward" else edge.target
                b = edge.target if direction == "forward" else edge.source
                if i == 0:
                    nodes.append(a)
                nodes.append(b)
                refs = [x for x in (add_join(edge, j) for j in edge.joins) if x]
                if not refs:
                    missing.append(edge.iri)
                name = edge.name if direction == "forward" else (edge.inverse_name or
                                                                 ("isReferencedBy" if edge.kind == "navigational"
                                                                  else f"inverse of {edge.name}"))
                edges_out.append({"relationship_iri": edge.iri, "predicate": name, "direction": direction,
                                  "kind": edge.kind, "status": edge.status, "join_refs": refs,
                                  "from": a, "to": b})
            entity_iris.extend(x for x in nodes if x not in entity_iris)
            statuses = {e["status"] for e in edges_out}
            validation = "verified" if statuses == {"verified"} and all(e["kind"] != "navigational"
                                                                        for e in edges_out) else "inferred"
            issues = []
            nav = [f"{snap.entities[e['from']]['name']} {e['predicate']} {snap.entities[e['to']]['name']}"
                   for e in edges_out if e["kind"] == "navigational"]
            if nav:
                issues.append("schema-level navigational edge(s) from foreign keys, without business semantics: "
                              + ", ".join(nav))
            unverified = sorted({snap.edges[e["relationship_iri"]].name for e in edges_out
                                 if e["status"] != "verified" and e["kind"] != "navigational"})
            if unverified:
                issues.append("inferred (unverified) relationship(s): " + ", ".join(unverified))
            if missing:
                issues.append("not executable for SQL, no join mapping for " + ", ".join(snap.edges[m].name
                                                                                     for m in missing))
            if issues:
                warnings.append(f"{pid}: " + "; ".join(issues) + ".")
            out["paths"].append({
                "path_id": pid, "hops": len(p), "nodes": nodes,
                "node_names": [snap.entities[x]["name"] for x in nodes], "edges": edges_out,
                "label": " -> ".join([snap.entities[nodes[0]]["name"]] + [
                    f"[{e['predicate']}] {snap.entities[e['to']]['name']}" for e in edges_out]),
                "executable": not missing, "missing_mappings": missing, "validation_status": validation,
                "relevance_score": r["score"], "score_explanation": {**r["components"], "calibrated": False}})
        # every relationship on a returned path, then those matched by name/synonym (details listed once)
        on_paths = list(dict.fromkeys(e["relationship_iri"] for p in out["paths"] for e in p["edges"]))
        for iri in on_paths + sorted(rel_hits - set(on_paths)):
            e = snap.edges[iri]
            refs = [x for x in (add_join(e, j) for j in e.joins) if x]
            item = {"iri": e.iri, "name": e.name, "kind": e.kind,
                    "predicate_iri": f"urn:mesh:predicate:{e.iri.rsplit(':sem-', 1)[-1]}"
                    if e.kind == "semantic" else None, "source_iri": e.source,
                    "source": snap.entities[e.source]["name"], "target_iri": e.target,
                    "target": snap.entities[e.target]["name"], "definition": e.definition, "synonyms": e.synonyms,
                    "inverse_name": e.inverse_name, "status": e.status, "join_refs": refs,
                    "provenance": e.provenance, "on_path": iri in on_paths,
                    "matched_queries": hits[iri].queries if iri in hits else []}
            if e.kind == "navigational":
                item["note"] = "Derived from a foreign key: schema-level navigation without business semantics."
            out["relationships"].append(item)
            for node in (e.source, e.target):
                if node not in entity_iris:
                    entity_iris.append(node)
        # entities
        for iri in entity_iris:
            ent = snap.entities[iri]
            h = hits.get(iri)
            item = {"iri": iri, "name": ent["name"], "definition": "; ".join(ent["definitions"]) or None,
                    "alt_names": ent["alt_names"], "classification": ent["classification"],
                    "role": "matched" if h else "path", "match_type": h.match_type if h else None,
                    "matched_queries": h.queries if h else [],
                    "definitions": [{k: d[k] for k in ("iri", "name", "ref")} for d in snap.definitions.get(iri, [])],
                    "term_iris": [t for t in ent["terms"] if scope.term_visible(snap, t)][:15]}
            if req.include_physical_mappings:
                item["tables"] = []
                for t, status in ent["tables"].items():
                    if scope.table_visible(snap, t) and status != "rejected":
                        item["tables"].append({"iri": t, "name": snap.tables[t]["name"], "mapping_status": status})
                        if t not in table_iris:
                            table_iris.append(t)
                if not item["tables"]:
                    warnings.append(f"Entity {ent['name']} has no mapped table in scope.")
            out["business_entities"].append(item)
        # terms: matched ones, then (related) the terms of join columns
        if req.include_related_terms:
            for c in column_iris:
                for t in snap.columns.get(c, {}).get("terms", {}):
                    if t not in term_iris and scope.term_visible(snap, t):
                        term_iris.append(t)
        related_relations: set[str] = set()
        for iri in term_iris:
            term = snap.terms[iri]
            h = hits.get(iri)
            item = {"iri": iri, "name": term["name"], "entity_iris": term["entities"],
                    "entity_iri": term["entities"][0] if len(term["entities"]) == 1 else None,
                    "definition": "; ".join(term["definitions"][:3]) or None, "synonyms": term["alt_names"],
                    "property_iri": term["property"], "role": "matched" if h else "join_column",
                    "match_type": h.match_type if h else None, "matched_queries": h.queries if h else [],
                    "definitions": [{k: d[k] for k in ("iri", "name", "ref")} for d in snap.definitions.get(iri, [])]}
            if req.include_physical_mappings:
                item["columns"] = []
                for c, status in term["columns"].items():
                    col = snap.columns.get(c)
                    if col is None or status == "rejected" or not scope.table_visible(snap, col["table"]):
                        continue
                    item["columns"].append({"iri": c, "name": col["name"],
                                            "table": snap.tables.get(col["table"], {}).get("name"),
                                            "catalog_ref": col["catalog_ref"], "data_type": col["data_type"],
                                            "dataset_id": snap.tables.get(col["table"], {}).get("dataset_id"),
                                            "mapping_status": status})
                    if c not in column_iris:
                        column_iris.append(c)
                    if col["table"] not in table_iris:
                        table_iris.append(col["table"])
                if h and not item["columns"]:
                    warnings.append(f"Term {term['name']} has no mapped column in scope.")
            if req.include_related_terms:
                item["related_terms"] = []
                for tr in snap.term_relations.values():
                    if tr["status"] == "rejected" or iri not in (tr["subject_iri"], tr["object_iri"]):
                        continue
                    other = tr["object_iri"] if tr["subject_iri"] == iri else tr["subject_iri"]
                    item["related_terms"].append({"relation_iri": tr["iri"], "predicate": tr["predicate"],
                                                  "direction": "outgoing" if tr["subject_iri"] == iri else "incoming",
                                                  "term_iri": other, "term": snap.terms[other]["name"],
                                                  "status": tr["status"]})
                    related_relations.add(tr["iri"])
            out["business_terms"].append(item)
        # physical mappings
        if req.include_physical_mappings:
            for t in table_iris:
                tb = snap.tables.get(t)
                if tb is None or not scope.table_visible(snap, t):
                    continue
                out["physical_mappings"]["tables"].append({
                    "iri": t, "dataset_id": tb["dataset_id"], "name": tb["name"], "schema": tb["schema"],
                    "source_id": tb["source_id"], "catalog_ref": tb["catalog_ref"],
                    "entities": [{"iri": e, "name": snap.entities.get(e, {}).get("name"), "mapping_status": s}
                                 for e, s in tb["entities"].items() if s != "rejected"]})
            for c in column_iris:
                col = snap.columns.get(c)
                if col is None or not scope.table_visible(snap, col["table"]):
                    continue
                out["physical_mappings"]["columns"].append({
                    "iri": c, "column_id": col["column_id"], "name": col["name"], "table_iri": col["table"],
                    "table": snap.tables.get(col["table"], {}).get("name"), "data_type": col["data_type"],
                    "terms": [{"iri": t, "name": snap.terms.get(t, {}).get("name"), "mapping_status": s}
                              for t, s in col["terms"].items() if s != "rejected"]})
            out["physical_mappings"]["joins"] = list(joins.values())
        # constraints: keys of the returned tables, term relations of the returned terms
        for k in sorted(snap.keys.values(), key=lambda k: k["iri"]):
            if k["type"] == "foreign" or k["table"] not in table_iris or k["status"] == "rejected":
                continue
            if k["type"] == "unique" and k["origin"] != "declared" and not k["pk_candidate"]:
                continue  # inferred unique column combinations are profiling detail, not constraints
            if not req.include_physical_mappings:
                break
            out["constraints"].append({"type": f"{k['type']}_key", "iri": k["iri"],
                                       "table": snap.tables.get(k["table"], {}).get("name"),
                                       "columns": [snap.columns.get(c, {}).get("name") for c in k["columns"]],
                                       "origin": k["origin"], "status": k["status"],
                                       "primary_key_candidate": k["pk_candidate"]})
        for iri in sorted(related_relations):
            out["constraints"].append({"type": "term_relation", **snap.term_relations[iri]})
        # evidence references (IDs only; content stays in the document store / definitions)
        refs: list[dict[str, Any]] = []
        for iri in entity_iris + term_iris:
            for d in snap.definitions.get(iri, []):
                refs.append({"type": "business_definition", "iri": d["iri"], "name": d["name"],
                             "definition_id": d["definition_id"], "ref": d["ref"], "describes": iri})
            for m in snap.mentions.get(iri, []):
                if scope.document_visible(m["document_id"]):
                    refs.append({**m, "mentions": iri})
        seen: set[str] = set()
        for r in refs:
            key = r.get("iri") or r.get("chunk_id")
            if key not in seen:
                seen.add(key)
                out["evidence"].append(r)
            if len(out["evidence"]) >= 20:
                break

    _TRIM_ORDER = (("evidence",), ("constraints",), ("physical_mappings", "columns"), ("business_terms",),
                   ("physical_mappings", "tables"), ("paths",), ("relationships",), ("physical_mappings", "joins"),
                   ("business_entities",))

    def _fit(self, out: dict[str, Any]) -> dict[str, Any]:
        """Bound the response (``max_response_chars``): least important lists lose their last items first;
        the best path is always kept. Omissions are listed in ``truncated``."""
        limit = self.cfg.max_response_chars - 300  # room for the "truncated" note
        size = lambda: len(json.dumps(out, default=str))  # noqa: E731
        if size() <= limit:
            return out
        truncated: dict[str, int] = {}
        keep = {"paths": 3, "relationships": 3, "joins": 3, "business_entities": 2, "business_terms": 1}
        for minimums in (keep, {"paths": 1, "warnings": 1}):  # second pass: only the best path is protected
            order = self._TRIM_ORDER + ((("ambiguities",), ("warnings",)) if minimums is not keep else ())
            for path in order:
                container = out
                for k in path[:-1]:
                    container = container[k]
                key, name = path[-1], ".".join(path)
                while len(container[key]) > minimums.get(key, 0) and size() > limit:
                    container[key].pop()
                    truncated[name] = truncated.get(name, 0) + 1
                if size() <= limit:
                    break
            if size() <= limit:
                break
        if truncated:  # items omitted per list (knowledge_graph.max_response_chars)
            out["truncated"] = truncated
        limit = self.cfg.max_response_chars
        if size() > limit:  # a single path still too long: shorten long texts, never drop the best path
            from ..agent.tools import _cut_strings, _strings

            cap = max((len(x) for x in _strings(out)), default=0)
            while size() > limit and cap > 80:
                cap = max(80, cap // 2)
                out = _cut_strings(out, cap)
        return out


class _Scope:
    """The request's source / dataset / document filters (the search scope; spec 3.15)."""

    def __init__(self, snap: _Snapshot, sources, datasets, documents):
        self.sources = list(sources) if sources else None
        self.datasets = list(datasets) if datasets else None
        self.documents = set(documents) if documents else None
        self._snap = snap
        self._tables = None
        if self.sources is not None or self.datasets is not None:
            self._tables = {iri for iri, t in snap.tables.items()
                            if (self.sources is None or t["source_id"] in self.sources)
                            and (self.datasets is None or t["dataset_id"] in self.datasets)}

    @property
    def restricted(self) -> bool:
        return self._tables is not None

    def table_visible(self, snap: _Snapshot, iri: str | None) -> bool:
        return iri is not None and (self._tables is None or iri in self._tables)

    def key_visible(self, snap: _Snapshot, key: dict[str, Any]) -> bool:
        return self.table_visible(snap, key["table"]) and (key["references"] is None
                                                           or self.table_visible(snap, key["references"]))

    def entity_visible(self, iri: str) -> bool:
        if self._tables is None:
            return True
        ent = self._snap.entities.get(iri)
        if ent is None:
            return False
        if not ent["tables"]:
            return True  # a concept from documents or definitions only
        return any(t in self._tables for t in ent["tables"])

    def term_visible(self, snap: _Snapshot, iri: str) -> bool:
        if self._tables is None:
            return True
        term = snap.terms.get(iri)
        if term is None:
            return False
        if not term["columns"]:
            return True
        return any(snap.columns.get(c, {}).get("table") in self._tables for c in term["columns"])

    def edge_visible(self, edge: _Edge) -> bool:
        if edge.kind == "document":
            return self.documents is None or bool(edge.documents & self.documents)
        if self._tables is None:
            return True
        return any(self.key_visible(self._snap, self._snap.keys[j["fk"]]) for j in edge.joins
                   if j["fk"] in self._snap.keys)

    def document_visible(self, document_id: str | None) -> bool:
        return self.documents is None or document_id in self.documents


def _occurrences(haystack: list[str], needle: list[str]) -> list[int]:
    n = len(needle)
    return [i for i in range(len(haystack) - n + 1) if n and haystack[i:i + n] == needle]


def _join_coverage(edge: _Edge) -> float:
    """How well the edge is backed by a physical join: the FK behind an explicit mapping (or behind a navigational
    edge) counts 1 when verified (declared or reviewed) and 0.5 when inferred; an automatic mapping is a
    candidate (0.5); no mapping counts 0."""
    best = 0.0
    for j in edge.joins:
        if j["status"] == "rejected" or j.get("fk_status") == "rejected":
            continue
        value = 0.5 if j["method"] == "automatic" else _COVERAGE.get(j.get("fk_status") or "inferred", 0.5)
        best = max(best, value)
    return best


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


__all__ = ["KGQueryRequest", "KnowledgeGraphRetriever", "RECORD_NOTE", "build_snapshot"]

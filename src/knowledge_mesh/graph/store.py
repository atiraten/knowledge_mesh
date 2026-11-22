"""RDFLib knowledge graph with pySHACL-gated publication and evidence lookup.

Storage model: an RDF Dataset with one named graph ("context") per source — a document,
a structured source, or a captured definition. Refreshing a source replaces only its
context; deleting a source drops it; evidence from other sources is untouched.

Every context is built to be self-contained (it types every node it references), so a
candidate update is SHACL-validated on its own before it replaces the published context.
A rejected update raises ``GraphValidationError`` with the violated shapes and focus nodes,
and the previously published graph stays available to retrieval.

A structured source is published as three layer contexts (physical, semantic, mapping) that
reference each other: ``publish_layers`` validates their union once and swaps all three
atomically. ``version`` increases with every publication; caches keyed by it (label index,
traversal index) are rebuilt lazily. ``reload`` re-reads the TriG file and swaps it in only when
it parses, so a corrupt file never replaces the usable graph.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from rdflib import Dataset, Graph, Literal, URIRef
from rdflib.namespace import SH

from ..common.errors import GraphValidationError, OperationFailed
from ..common.ids import utcnow
from ..observability import get_metrics, span
from .ontology import LAYERS, MESH, ONTOLOGY_CONTEXT, PROV, RDF, RDFS, SKOS, context_uri, layer_context_uri

log = logging.getLogger(__name__)
MAX_VIOLATIONS = 50


class KnowledgeGraph:
    def __init__(self, data_path: Path, shapes_path: Path):
        self.data_path = Path(data_path)
        self.shapes_path = Path(shapes_path)
        self._lock = threading.RLock()
        self.shapes = Graph()
        self.shapes.parse(self.shapes_path, format="turtle")
        self.dataset = Dataset(default_union=True)
        if self.data_path.exists() and self.data_path.stat().st_size:
            self.dataset.parse(self.data_path, format="trig")
        self._label_index: dict[str, list[URIRef]] | None = None
        self.version = 1
        self.published_at = utcnow()
        self._caches: dict[str, tuple[int, Any]] = {}

    def cached(self, name: str, build):
        """A value derived from the graph, rebuilt after every publication (keyed by ``version``)."""
        with self._lock:
            hit = self._caches.get(name)
            if hit is not None and hit[0] == self.version:
                return hit[1]
            value = build()
            self._caches[name] = (self.version, value)
            return value

    def _changed(self) -> None:
        self._label_index = None
        self.version += 1
        self.published_at = utcnow()

    # -- publication -----------------------------------------------------------------
    def validate(self, graph: Graph) -> list[dict[str, Any]]:
        import pyshacl

        conforms, results, _text = pyshacl.validate(
            graph, shacl_graph=self.shapes, inference="none", advanced=True, allow_warnings=True,
        )
        if conforms:
            return []
        violations = []
        for result in results.subjects(RDF.type, SH.ValidationResult):
            severity = results.value(result, SH.resultSeverity)
            if severity is not None and severity != SH.Violation:
                continue
            violations.append({
                "focus_node": str(results.value(result, SH.focusNode)),
                "path": _str(results.value(result, SH.resultPath)),
                "shape": _str(results.value(result, SH.sourceShape)),
                "constraint": _str(results.value(result, SH.sourceConstraintComponent)),
                "message": _str(results.value(result, SH.resultMessage)),
            })
            if len(violations) >= MAX_VIOLATIONS:
                break
        return violations

    def publish(self, context: URIRef, graph: Graph | None, description: str = "") -> dict[str, Any]:
        """Validate ``graph`` and atomically replace ``context`` with it (None removes the context)."""
        metrics = get_metrics()
        kind = str(context).split(":")[3] if str(context).count(":") >= 3 else "unknown"
        with span("graph.publish", "CHAIN", {"graph.context": str(context), "graph.triples": len(graph or [])}) as sp:
            start = time.perf_counter()
            if graph is not None and len(graph):
                violations = self.validate(graph)
                metrics.stage_duration.record(time.perf_counter() - start, {"stage": "shacl_validation"})
                if violations:
                    metrics.graph_validations.add(1, {"outcome": "rejected", "context_type": kind})
                    metrics.graph_validation_failures.add(1, {"context_type": kind})
                    sp.set_attribute("graph.validation", "rejected")
                    log.warning("graph update rejected by SHACL",
                                extra={"context": str(context), "violations": len(violations)})
                    raise GraphValidationError(
                        f"graph update for {description or context} violates {len(violations)} SHACL constraint(s); "
                        "the previously published graph is unchanged",
                        {"context": str(context), "violations": violations},
                    )
                metrics.graph_validations.add(1, {"outcome": "accepted", "context_type": kind})
                sp.set_attribute("graph.validation", "accepted")
            with self._lock:
                self.dataset.remove_graph(self.dataset.graph(context))
                if graph is not None and len(graph):
                    target = self.dataset.graph(context)
                    for triple in graph:
                        target.add(triple)
                self._persist()
                self._changed()
            return {"context": str(context), "triples": len(graph or []), "conforms": True}

    def remove(self, context: URIRef) -> None:
        self.publish(context, None)

    def publish_many(self, graphs: dict[URIRef, Graph], description: str = "") -> dict[str, Any]:
        """Validate every graph, then replace all their contexts at once (one write of the data file).
        Nothing changes when one of them violates the shapes. An empty graph removes its context."""
        with span("graph.publish", "CHAIN", {"graph.context": f"{len(graphs)} contexts",
                                             "graph.triples": sum(len(g) for g in graphs.values())}) as sp:
            for context, graph in graphs.items():
                violations = self.validate(graph) if len(graph) else []
                if violations:
                    sp.set_attribute("graph.validation", "rejected")
                    raise GraphValidationError(
                        f"graph update for {description or context} violates {len(violations)} SHACL constraint(s); "
                        "the previously published graph is unchanged",
                        {"context": str(context), "violations": violations})
            sp.set_attribute("graph.validation", "accepted")
            with self._lock:
                for context, graph in graphs.items():
                    self.dataset.remove_graph(self.dataset.graph(context))
                    if len(graph):
                        target = self.dataset.graph(context)
                        for triple in graph:
                            target.add(triple)
                self._persist()
                self._changed()
            return {"contexts": len(graphs), "triples": sum(len(g) for g in graphs.values()), "conforms": True}

    def publish_layers(self, source_id: str, layers: dict[str, Graph], description: str = "") -> dict[str, Any]:
        """Validate a structured source's layers together, then atomically replace its layer contexts (and the
        single context of the earlier layout). Empty layers remove their context."""
        metrics = get_metrics()
        union = Graph()
        for g in layers.values():
            for triple in g:
                union.add(triple)
        with span("graph.publish", "CHAIN", {"graph.context": f"urn:mesh:context:source:{source_id}:*",
                                             "graph.triples": len(union)}) as sp:
            start = time.perf_counter()
            if len(union):
                violations = self.validate(union)
                metrics.stage_duration.record(time.perf_counter() - start, {"stage": "shacl_validation"})
                if violations:
                    metrics.graph_validations.add(1, {"outcome": "rejected", "context_type": "source"})
                    metrics.graph_validation_failures.add(1, {"context_type": "source"})
                    sp.set_attribute("graph.validation", "rejected")
                    log.warning("graph update rejected by SHACL",
                                extra={"context": f"source:{source_id}", "violations": len(violations)})
                    raise GraphValidationError(
                        f"graph update for {description or source_id} violates {len(violations)} SHACL "
                        "constraint(s); the previously published graph is unchanged",
                        {"context": f"urn:mesh:context:source:{source_id}", "violations": violations})
                metrics.graph_validations.add(1, {"outcome": "accepted", "context_type": "source"})
                sp.set_attribute("graph.validation", "accepted")
            with self._lock:
                self.dataset.remove_graph(self.dataset.graph(context_uri("source", source_id)))
                for layer in LAYERS:
                    ctx = layer_context_uri(source_id, layer)
                    self.dataset.remove_graph(self.dataset.graph(ctx))
                    g = layers.get(layer)
                    if g is not None and len(g):
                        target = self.dataset.graph(ctx)
                        for triple in g:
                            target.add(triple)
                self._persist()
                self._changed()
            return {"contexts": {layer: len(layers.get(layer) or []) for layer in LAYERS},
                    "triples": sum(len(g) for g in layers.values()), "conforms": True}

    def remove_source(self, source_id: str) -> None:
        """Drop every context of a structured source (all layers and the earlier single context)."""
        with self._lock:
            for ctx in [context_uri("source", source_id)] + [layer_context_uri(source_id, x) for x in LAYERS]:
                self.dataset.remove_graph(self.dataset.graph(ctx))
            self._persist()
            self._changed()

    def legacy_source_contexts(self) -> list[str]:
        """Source ids still published in the single-context layout (``urn:mesh:context:source:<id>``)."""
        prefix = "urn:mesh:context:source:"
        out = []
        with self._lock:
            for c in self.dataset.contexts():
                ident = str(c.identifier)
                if ident.startswith(prefix) and len(c) and not any(ident.endswith(f":{x}") for x in LAYERS):
                    out.append(ident[len(prefix):])
        return sorted(out)

    def publish_ontology(self, path: Path | None) -> dict[str, Any]:
        """(Re)publish the vocabulary file as ``urn:mesh:context:ontology:core``."""
        if path is None or not Path(path).exists():
            log.warning("ontology file not found; vocabulary context not published", extra={"path": str(path)})
            return {"status": "skipped", "reason": "ontology file not found"}
        g = Graph()
        g.parse(path, format="turtle")
        with self._lock:
            current = self.dataset.graph(ONTOLOGY_CONTEXT)
            if set(current) == set(g):
                return {"status": "unchanged", "triples": len(g)}
        self.publish(ONTOLOGY_CONTEXT, g, "ontology")
        return {"status": "published", "triples": len(g)}

    def reload(self) -> dict[str, Any]:
        """Re-read the persisted TriG file; the current graph stays in use if the file does not parse."""
        fresh = Dataset(default_union=True)
        try:
            if self.data_path.exists() and self.data_path.stat().st_size:
                fresh.parse(self.data_path, format="trig")
        except Exception as exc:
            log.error("graph reload failed; keeping the current graph", extra={"error": type(exc).__name__})
            raise OperationFailed(f"graph reload failed ({type(exc).__name__}); the current graph is unchanged") \
                from exc
        with self._lock:
            self.dataset = fresh
            self._changed()
        return {"status": "reloaded", **self.stats()}

    def _persist(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.data_path.with_name(self.data_path.name + ".tmp")
        self.dataset.serialize(destination=str(tmp), format="trig")
        os.replace(tmp, self.data_path)

    def context_triples(self, context: URIRef) -> int:
        """Triples of a context; for a structured source (``urn:mesh:context:source:<id>``) all its layers."""
        with self._lock:
            n = len(self.dataset.graph(context))
            if str(context).startswith("urn:mesh:context:source:") and str(context).count(":") == 4:
                n += sum(len(self.dataset.graph(URIRef(f"{context}:{layer}"))) for layer in LAYERS)
            return n

    def stats(self) -> dict[str, Any]:
        with self._lock:
            contexts = [c for c in self.dataset.contexts() if len(c)]
            return {
                "triples": sum(len(c) for c in contexts),
                "contexts": len(contexts),
                "entities": len(set(self.dataset.subjects(RDF.type, MESH.BusinessEntity))),
                "terms": len(set(self.dataset.subjects(RDF.type, MESH.BusinessTerm))),
                "definitions": len(set(self.dataset.subjects(RDF.type, MESH.BusinessDefinition))),
                "relationships": len(set(self.dataset.subjects(RDF.type, MESH.Relationship))),
                "keys": len(set(self.dataset.subjects(RDF.type, MESH.KeyConstraint))),
                "semantic_relationships": len(set(self.dataset.subjects(RDF.type, MESH.SemanticRelationship))),
                "term_relations": len(set(self.dataset.subjects(RDF.type, MESH.TermRelation))),
                "data_path": str(self.data_path),
                "version": self.version, "published_at": self.published_at,
            }

    # -- lookup ----------------------------------------------------------------------
    def _labels(self) -> dict[str, list[URIRef]]:
        if self._label_index is None:
            index: dict[str, list[URIRef]] = {}
            for cls in (MESH.BusinessEntity, MESH.BusinessTerm, MESH.BusinessDefinition):
                for node in set(self.dataset.subjects(RDF.type, cls)):
                    labels = list(self.dataset.objects(node, RDFS.label)) + list(self.dataset.objects(node, SKOS.altLabel))
                    for label in labels:
                        key = _norm(str(label))
                        if key and node not in index.setdefault(key, []):
                            index[key].append(node)
            self._label_index = index
        return self._label_index

    def entity_labels(self) -> list[str]:
        with self._lock:
            return sorted({str(self.dataset.value(n, RDFS.label) or "")
                           for n in set(self.dataset.subjects(RDF.type, MESH.BusinessEntity))} - {""})

    def concepts_in_text(self, text: str, limit: int = 20) -> list[URIRef]:
        """Concepts whose label (or its plural) appears in ``text``, longest labels first."""
        with self._lock:
            normalized = f" {_norm(text)} "
            hits: list[tuple[int, URIRef]] = []
            for label, nodes in self._labels().items():
                variants = {label, label + "s", label + "es", label[:-1] + "ies" if label.endswith("y") else label}
                if any(f" {v} " in normalized for v in variants):
                    hits.extend((len(label), n) for n in nodes)
            hits.sort(key=lambda t: -t[0])
            out: list[URIRef] = []
            for _, n in hits:
                if n not in out:
                    out.append(n)
            return out[:limit]

    def evidence_for(self, seeds: list[URIRef], limit: int) -> list[dict[str, Any]]:
        """Definitions, relationships and constraints reachable from the seed concepts (bounded)."""
        if limit <= 0:
            return []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        ds = self.dataset

        def add(item: dict[str, Any] | None) -> bool:
            if item and item["graph_ref"] not in seen:
                seen.add(item["graph_ref"])
                out.append(item)
            return len(out) >= limit

        with self._lock:
            concepts: list[URIRef] = []
            for seed in seeds:
                if seed not in concepts:
                    concepts.append(seed)
                if (seed, RDF.type, MESH.BusinessTerm) in ds:
                    for ent in ds.objects(seed, MESH.termOfEntity):
                        if ent not in concepts:
                            concepts.append(ent)
            # 1. definitions (directly matched, or describing a matched concept)
            for c in concepts:
                if (c, RDF.type, MESH.BusinessDefinition) in ds and add(self._definition(c)):
                    return out
                for d in sorted(set(ds.subjects(MESH.describes, c))):
                    if add(self._definition(d)):
                        return out
            # 1b. descriptive relations of the directly matched terms (only graphs with term relations)
            for seed in seeds:
                if (seed, RDF.type, MESH.BusinessTerm) not in ds:
                    continue
                for rel in sorted(set(ds.subjects(MESH.subjectTerm, seed)) | set(ds.subjects(MESH.objectTerm, seed))):
                    if not self._rejected(rel) and add(self._term_relation(rel)):
                        return out
            # 2. relationships involving matched entities (rejected ones are left out)
            for c in concepts:
                rels = set(ds.subjects(MESH.sourceEntity, c)) | set(ds.subjects(MESH.targetEntity, c))
                for r in sorted(rels):
                    if self._rejected(r):
                        continue
                    if add(self._relationship(r)):
                        return out
            # 3. descriptive relations between matched terms (or the terms of matched entities)
            terms = [c for c in concepts if (c, RDF.type, MESH.BusinessTerm) in ds]
            for c in concepts:
                if (c, RDF.type, MESH.BusinessEntity) in ds:
                    terms += sorted(t for t in ds.subjects(MESH.termOfEntity, c) if t not in terms)
            for t in terms:
                relations = set(ds.subjects(MESH.subjectTerm, t)) | set(ds.subjects(MESH.objectTerm, t))
                for rel in sorted(relations):
                    if self._rejected(rel):
                        continue
                    if add(self._term_relation(rel)):
                        return out
            # 4. key constraints on datasets representing matched entities (or owning matched terms' columns)
            for c in concepts:
                datasets = set(ds.subjects(MESH.representsEntity, c))
                for col in ds.objects(c, MESH.boundToColumn):
                    datasets |= set(ds.objects(col, MESH.inDataset))
                for dset in sorted(datasets):
                    for key in sorted(set(ds.subjects(MESH.onDataset, dset))):
                        if self._rejected(key):
                            continue
                        if add(self._key(key)):
                            return out
        return out

    def _rejected(self, node: URIRef) -> bool:
        return _str(self.dataset.value(node, MESH.validationStatus)) == "rejected"

    def _term_relation(self, node: URIRef) -> dict[str, Any] | None:
        ds = self.dataset
        subj, obj, pred = ds.value(node, MESH.subjectTerm), ds.value(node, MESH.objectTerm), ds.value(node, MESH.predicate)
        if subj is None or obj is None or pred is None:
            return None
        status = _str(ds.value(node, MESH.validationStatus)) or "inferred"
        name = str(pred).rsplit(":", 1)[-1]
        label = f"{_str(ds.value(subj, RDFS.label))} {name} {_str(ds.value(obj, RDFS.label))}"
        text = label
        definition = _str(ds.value(node, SKOS.definition))
        if definition:
            text += f" — {definition[:300]}"
        text += f" (descriptive term relation, not validated against records; status {status})"
        run = ds.value(node, PROV.wasGeneratedBy)
        return {"type": "term_relation", "label": label, "text": text, "graph_ref": str(node),
                "source_ref": f"/api/business-terms/{str(subj).rsplit(':', 1)[-1]}", "inferred": status != "verified",
                "status": status,
                "provenance": {"kind": "relationship_proposal", "status": status, "predicate": str(pred),
                               "run": _str(ds.value(run, MESH.runId)) if run is not None else None}}

    def _definition(self, node: URIRef) -> dict[str, Any] | None:
        ds = self.dataset
        if (node, RDF.type, MESH.BusinessDefinition) not in ds:
            return None
        def_id = str(node).rsplit(":", 1)[-1]
        describes = [{"graph_ref": str(t), "label": _str(ds.value(t, RDFS.label))} for t in ds.objects(node, MESH.describes)]
        document = ds.value(node, MESH.definedIn)
        document_id = str(document).rsplit(":", 1)[-1] if document is not None else None
        return {
            "type": "business_definition",
            "label": _str(ds.value(node, RDFS.label)),
            "text": _str(ds.value(node, MESH.definitionText)),
            "graph_ref": str(node),
            "source_ref": f"/api/business-definitions/{def_id}",
            "inferred": False,
            "provenance": {"kind": "captured_definition", "definition_id": def_id,
                           "author": _str(ds.value(node, MESH.author)),
                           "document_id": document_id,
                           "document_ref": f"/api/documents/{document_id}" if document_id else None,
                           "location": _str(ds.value(node, MESH.location)),
                           "describes": describes},
        }

    def _relationship(self, node: URIRef) -> dict[str, Any] | None:
        ds = self.dataset
        src, dst = ds.value(node, MESH.sourceEntity), ds.value(node, MESH.targetEntity)
        if src is None or dst is None:
            return None
        rtype = _str(ds.value(node, MESH.relationshipType))
        src_label, dst_label = _str(ds.value(src, RDFS.label)), _str(ds.value(dst, RDFS.label))
        derived = ds.value(node, PROV.wasDerivedFrom)
        prov: dict[str, Any] = {"method": _str(ds.value(node, MESH.extractionMethod))}
        source_ref = None
        inferred = False
        if derived is not None and (derived, RDF.type, MESH.Passage) in ds:
            chunk_id = _str(ds.value(derived, MESH.chunkId))
            prov.update(kind="document_passage", document_id=_str(ds.value(derived, MESH.documentId)),
                        chunk_id=chunk_id, source_uri=_str(ds.value(derived, MESH.sourceUri)),
                        page=_int(ds.value(derived, MESH.page)), section=_str(ds.value(derived, MESH.section)))
            source_ref = f"/api/chunks/{chunk_id}"
        elif derived is not None and (derived, RDF.type, MESH.KeyConstraint) in ds:
            origin = _str(ds.value(derived, MESH.origin))
            inferred = origin == "inferred"
            prov.update(kind="catalog_key", key_ref=str(derived), origin=origin,
                        profiling_run=_str(ds.value(derived, MESH.profilingRun)))
            dset = ds.value(derived, MESH.onDataset)
            source_ref = f"/api/datasets/{_str(ds.value(dset, MESH.datasetId))}" if dset is not None else None
        text = f"{src_label} {rtype} {dst_label}"
        evidence = _str(ds.value(node, MESH.evidenceText))
        if evidence:
            text += f' — "{evidence[:300]}"'
        status = _str(ds.value(node, MESH.validationStatus))
        if (node, RDF.type, MESH.SemanticRelationship) in ds:
            # proposed by the relationship stage for a catalog FK: status comes from review, not the key
            status = status or "inferred"
            inferred = status != "verified"
            definition = _str(ds.value(node, SKOS.definition))
            if definition:
                text += f" — {definition[:300]}"
            run = ds.value(node, PROV.wasGeneratedBy)
            prov.update(kind="relationship_proposal", status=status,
                        run=_str(ds.value(run, MESH.runId)) if run is not None else None)
            text += f" (proposed relationship; status {status})"
        elif inferred:
            text += " (inferred from profiling)"
        out = {"type": "relationship", "label": f"{src_label} {rtype} {dst_label}", "text": text,
               "graph_ref": str(node), "source_ref": source_ref, "inferred": inferred, "provenance": prov}
        if status:
            out["status"] = status
        return out

    def _key(self, node: URIRef) -> dict[str, Any] | None:
        ds = self.dataset
        dset = ds.value(node, MESH.onDataset)
        if dset is None:
            return None
        types = {str(t).rsplit(":", 1)[-1] for t in ds.objects(node, RDF.type)}
        key_type = "foreign_key" if "ForeignKey" in types else ("primary_key" if "PrimaryKey" in types else "unique_key")
        origin = _str(ds.value(node, MESH.origin))
        cols = sorted(((int(ds.value(m, MESH.position)), _str(ds.value(ds.value(m, MESH.column), RDFS.label)))
                       for m in ds.objects(node, MESH.keyColumn)))
        col_names = [c for _, c in cols]
        dname = _str(ds.value(dset, RDFS.label))
        text = f"{key_type.replace('_', ' ')} on {dname}({', '.join(col_names)})"
        if key_type == "foreign_key":
            maps = sorted(((int(ds.value(m, MESH.position)), _str(ds.value(ds.value(m, MESH.toColumn), RDFS.label)))
                           for m in ds.objects(node, MESH.columnMapping)))
            ref = ds.value(node, MESH.referencesDataset)
            text += f" references {_str(ds.value(ref, RDFS.label))}({', '.join(c for _, c in maps)})"
        if _bool(ds.value(node, MESH.primaryKeyCandidate)):
            text += " [primary-key candidate]"
        inferred = origin == "inferred"
        confidence = _str(ds.value(node, MESH.fkConfidence))
        if inferred:
            graded = f"confidence {confidence} ({_str(ds.value(node, MESH.fkConfidenceBasis))}); " if confidence else ""
            text += f" (inferred from profiling; {graded}scope {_str(ds.value(node, MESH.profilingScope))})"
        else:
            text += " (declared by source)"
        return {
            "type": "key_constraint", "label": text.split(" (")[0], "text": text, "graph_ref": str(node),
            "source_ref": f"/api/datasets/{_str(ds.value(dset, MESH.datasetId))}", "inferred": inferred,
            "provenance": {"kind": "catalog_key", "origin": origin, "catalog_ref": _str(ds.value(dset, MESH.catalogRef)),
                           "profiling_run": _str(ds.value(node, MESH.profilingRun)) or None,
                           **({"confidence": confidence} if confidence else {})},
            **({"status": status} if (status := _str(ds.value(node, MESH.validationStatus))) else {}),
        }

    def describe_concept(self, node: URIRef) -> dict[str, Any]:
        """Bounded summary of an entity/term for the glossary pages."""
        ds = self.dataset
        with self._lock:
            return {
                "graph_ref": str(node),
                "label": _str(ds.value(node, RDFS.label)),
                "types": sorted(str(t).rsplit(":", 1)[-1] for t in ds.objects(node, RDF.type)),
                "descriptions": sorted({str(d) for d in ds.objects(node, SKOS.definition)})[:10],
                "evidence": self.evidence_for([node], 25),
                "mentioned_in": [
                    {"chunk_id": _str(ds.value(p, MESH.chunkId)), "document_id": _str(ds.value(p, MESH.documentId)),
                     "source_ref": f"/api/chunks/{_str(ds.value(p, MESH.chunkId))}"}
                    for p in list(ds.objects(node, MESH.mentionedIn))[:25]
                ],
            }


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _str(value) -> str | None:
    return None if value is None else str(value)


def _int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bool(value) -> bool:
    if isinstance(value, Literal):
        return bool(value.toPython())
    return False

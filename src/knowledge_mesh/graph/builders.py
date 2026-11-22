"""Build self-contained RDF contexts from catalog records, document entities and definitions.

A structured source is published as three layers (``build_source_layers``): physical, semantic and
mapping (see ``graph/ontology.py``). The layers reference each other's nodes, so they are
SHACL-validated together and published together.
"""

from __future__ import annotations

from typing import Any, Iterable

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD

from ..common.ids import slugify, stable_hash
from ..storage.db import Database, dumps, loads
from ..structured.fk_rules import FKRules, assess_foreign_key
from ..structured.glossary_store import GlossaryStore, columns_select, datasets_select
from ..structured.reviews import ReviewStore, combine, lookup
from .ontology import (DCTERMS, GLOSSARY_SCHEME, MESH, OWL, PROV, RDF, RDFS, SKOS, entity_uri, predicate_uri,
                       property_uri, term_predicate, term_uri, uri)


def _g() -> Graph:
    g = Graph()
    g.bind("mesh", MESH)
    g.bind("prov", PROV)
    g.bind("skos", SKOS)
    g.bind("owl", OWL)
    return g


def _lit(value: Any) -> Literal:
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)
    return Literal(str(value), datatype=XSD.string)


def _entity(g: Graph, label: str, slug: str | None = None) -> URIRef:
    """A business entity type: an OWL class (also typed mesh:BusinessEntity). ``slug``: its glossary slug
    (kept when the label is edited); derived from the label when not given."""
    node = entity_uri(slug or slugify(label))
    g.add((node, RDF.type, MESH.BusinessEntity))
    g.add((node, RDF.type, OWL.Class))
    g.add((node, RDFS.label, _lit(label)))
    return node


def _term(g: Graph, label: str, slug: str | None = None) -> URIRef:
    """A governed business term: a SKOS concept of the glossary scheme (also typed mesh:BusinessTerm).
    ``slug``: its glossary slug (kept when the label is edited); derived from the label when not given."""
    node = term_uri(slug or slugify(label))
    g.add((node, RDF.type, MESH.BusinessTerm))
    g.add((node, RDF.type, SKOS.Concept))
    g.add((node, RDFS.label, _lit(label)))
    g.add((node, SKOS.prefLabel, _lit(label)))
    g.add((node, SKOS.inScheme, GLOSSARY_SCHEME))
    return node


# -- documents -------------------------------------------------------------------------
def build_document_context(document: dict[str, Any], chunks: dict[str, dict[str, Any]],
                           mentions: Iterable, relationships: Iterable) -> Graph:
    g = _g()
    doc_node = uri("document", document["document_id"])
    mentions, relationships = list(mentions), list(relationships)
    if not mentions and not relationships:
        return g
    g.add((doc_node, RDF.type, MESH.Document))
    g.add((doc_node, MESH.documentId, _lit(document["document_id"])))
    g.add((doc_node, MESH.sourceUri, _lit(document["source_uri"])))
    if document.get("title"):
        g.add((doc_node, DCTERMS.title, _lit(document["title"])))

    def passage(chunk_id: str) -> URIRef:
        node = uri("passage", chunk_id)
        g.add((node, RDF.type, MESH.Passage))
        g.add((node, MESH.chunkId, _lit(chunk_id)))
        g.add((node, MESH.documentId, _lit(document["document_id"])))
        g.add((node, MESH.inDocument, doc_node))
        g.add((node, MESH.sourceUri, _lit(document["source_uri"])))
        info = chunks.get(chunk_id) or {}
        if info.get("page") is not None:
            g.add((node, MESH.page, _lit(int(info["page"]))))
        if info.get("section"):
            g.add((node, MESH.section, _lit(info["section"])))
        return node

    for m in mentions:
        ent = _entity(g, m.label, m.slug)
        g.add((ent, MESH.mentionedIn, passage(m.chunk_id)))
    for r in relationships:
        node = uri("relationship", r.relationship_id)
        g.add((node, RDF.type, MESH.Relationship))
        g.add((node, MESH.sourceEntity, _entity(g, r.source_label, r.source_slug)))
        g.add((node, MESH.targetEntity, _entity(g, r.target_label, r.target_slug)))
        g.add((node, MESH.relationshipType, _lit(r.rel_type)))
        g.add((node, RDFS.label, _lit(f"{r.source_label} {r.rel_type} {r.target_label}")))
        g.add((node, PROV.wasDerivedFrom, passage(r.chunk_id)))
        g.add((node, MESH.extractionMethod, _lit(r.method)))
        if r.evidence:
            g.add((node, MESH.evidenceText, _lit(r.evidence[:500])))
    return g


# -- structured sources ------------------------------------------------------------------
def _join_status(relationship: str, key: str, method: str) -> str:
    """Explicit mapping (the FK the relationship was proposed for): verified only when both the relationship
    and the FK are verified. Automatic mapping (another FK between the two entities' tables): a candidate,
    never verified by itself."""
    if method == "explicit":
        return combine(relationship, key)
    return "rejected" if "rejected" in (relationship, key) else "inferred"


_LEVELS = {"low": 0, "medium": 1, "high": 2}


def build_source_layers(db: Database, source_id: str, fk_rules: FKRules | None = None,
                        decisions: dict | None = None, min_fk_confidence: str = "medium") -> dict[str, Graph]:
    """The physical, semantic and mapping layers of one structured source (see ``graph/ontology.py``).

    ``min_fk_confidence``: inferred FKs below this confidence are not used as automatic join mappings
    (the same threshold as the relationship stage)."""
    phys, sem, mp = _g(), _g(), _g()
    layers = {"physical": phys, "semantic": sem, "mapping": mp}
    datasets = db.query(datasets_select(db) + " WHERE d.source_id = ? AND d.active = 1", (source_id,))
    if not datasets:
        return layers
    decisions = decisions if decisions is not None else ReviewStore(db).all()
    term_labels, entity_labels = GlossaryStore(db).labels()
    ds_ids = [d["dataset_id"] for d in datasets]
    marks = ",".join("?" * len(ds_ids))
    columns = db.query(columns_select(db) + f" WHERE c.dataset_id IN ({marks}) AND c.active = 1", ds_ids)
    keys = db.query(f"SELECT * FROM keys WHERE dataset_id IN ({marks}) AND active = 1", ds_ids)
    col_by_ds_name = {(c["dataset_id"], c["name"]): c for c in columns}
    ds_by_id = {d["dataset_id"]: d for d in datasets}

    def dataset_node(d: dict[str, Any]) -> URIRef:
        node = uri("dataset", d["dataset_id"])
        phys.add((node, RDF.type, MESH.Dataset))
        phys.add((node, RDFS.label, _lit(d["name"])))
        phys.add((node, MESH.datasetId, _lit(d["dataset_id"])))
        phys.add((node, MESH.catalogRef, _lit(d["catalog_ref"])))
        phys.add((node, MESH.sourceId, _lit(d["source_id"])))
        if d.get("schema_name"):
            phys.add((node, MESH.schemaName, _lit(d["schema_name"])))
        if d.get("kind"):
            phys.add((node, MESH.datasetKind, _lit(d["kind"])))
        return node

    def column_node(c: dict[str, Any], d: dict[str, Any]) -> URIRef:
        node = uri("column", c["column_id"])
        dnode = dataset_node(d)
        phys.add((node, RDF.type, MESH.Column))
        phys.add((node, RDFS.label, _lit(c["name"])))
        phys.add((node, MESH.columnId, _lit(c["column_id"])))
        phys.add((node, MESH.catalogRef, _lit(c["catalog_ref"])))
        phys.add((node, MESH.inDataset, dnode))
        phys.add((dnode, MESH.hasColumn, node))
        if c.get("data_type"):
            phys.add((node, MESH.dataType, _lit(c["data_type"])))
        return node

    # entities (slug, label) of the source's tables (absent when the table -> entity link is rejected)
    entity_of: dict[str, tuple[str, str]] = {}
    term_entities: dict[str, set[str]] = {}
    for d in datasets:
        if d.get("entity_label") and lookup(decisions, "entity_mapping", d["dataset_id"], d["entity_slug"]) != "rejected":
            entity_of[d["dataset_id"]] = (d["entity_slug"], d["entity_label"])
            for c in (x for x in columns if x["dataset_id"] == d["dataset_id"]):
                if c.get("term_label") and c.get("term_status") == "generated":
                    term_entities.setdefault(c["term_slug"], set()).add(d["entity_slug"])

    term_slugs: set[str] = set()
    for d in datasets:
        dnode = dataset_node(d)
        if d.get("physical_ref"):
            phys.add((dnode, MESH.physicalRef, _lit(d["physical_ref"])))
        ent = None
        if d.get("entity_label"):
            status = lookup(decisions, "entity_mapping", d["dataset_id"], d["entity_slug"])
            node = _entity(sem, d["entity_label"], d["entity_slug"])
            em = uri("entity-mapping", d["dataset_id"])
            mp.add((em, RDF.type, MESH.EntityMapping))
            mp.add((em, MESH.dataset, dnode))
            mp.add((em, MESH.entity, node))
            mp.add((em, MESH.validationStatus, _lit(status)))
            if d.get("entity_method"):
                mp.add((em, MESH.extractionMethod, _lit(d["entity_method"])))
            if d.get("entity_run_id"):
                mp.add((em, PROV.wasGeneratedBy, uri("term-run", d["entity_run_id"])))
            if status != "rejected":
                ent = node
                mp.add((dnode, MESH.representsEntity, ent))
                mp.add((ent, PROV.wasDerivedFrom, dnode))
        for c in (x for x in columns if x["dataset_id"] == d["dataset_id"]):
            cnode = column_node(c, d)
            if not (c.get("term_label") and c.get("term_status") == "generated"):
                continue
            status = lookup(decisions, "term_mapping", c["column_id"], c["term_slug"])
            term = _term(sem, c["term_label"], c["term_slug"])
            term_slugs.add(c["term_slug"])
            prop = property_uri(c["term_slug"])  # the attribute meaning, separate from the SKOS concept
            sem.add((prop, RDF.type, OWL.DatatypeProperty))
            sem.add((prop, RDFS.label, _lit(c["term_label"])))
            sem.add((term, MESH.termProperty, prop))
            if c.get("term_description"):
                sem.add((term, SKOS.definition, _lit(c["term_description"])))
            if c.get("term_sensitive") is not None:
                sem.add((term, MESH.sensitive, _lit(bool(c["term_sensitive"]))))
            if ent is not None and status != "rejected":
                sem.add((term, MESH.termOfEntity, ent))
                if len(term_entities.get(c["term_slug"], ())) == 1:  # several entities: no single OWL domain
                    sem.add((prop, RDFS.domain, ent))
            binding = uri("term-binding", c["column_id"])
            mp.add((binding, RDF.type, MESH.TermBinding))
            mp.add((binding, MESH["term"], term))  # MESH.term is Namespace.term(), not a URI
            mp.add((binding, MESH.column, cnode))
            mp.add((binding, MESH.validationStatus, _lit(status)))
            mp.add((binding, PROV.wasGeneratedBy, uri("term-run", c.get("term_run_id") or "unknown")))
            if c.get("term_description"):
                mp.add((binding, SKOS.definition, _lit(c["term_description"])))
            if c.get("term_explanation"):
                mp.add((binding, MESH.explanation, _lit(c["term_explanation"])))
            if status != "rejected":
                mp.add((term, MESH.boundToColumn, cnode))
                mp.add((cnode, MESH.hasBusinessTerm, term))

    for row in db.query("SELECT term_slug, synonym FROM term_synonyms WHERE source_id = ? AND active = 1",
                        (source_id,)):
        if row["term_slug"] in term_slugs:
            sem.add((term_uri(row["term_slug"]), SKOS.altLabel, _lit(row["synonym"])))

    # -- keys (physical) and FK-derived navigational edges (mapping) ------------------------------
    runs: set[URIRef] = set()
    fk_nodes: dict[str, tuple[URIRef, str, dict[str, Any], dict[str, Any]]] = {}  # key_id -> (node, status, dep, ref)
    weak_fks: set[str] = set()  # inferred FKs below min_fk_confidence: no automatic join mapping
    declared_sig = {(k["dataset_id"], k["columns_json"], k["referenced_dataset_id"], k["referenced_columns_json"])
                    for k in keys if k["key_type"] == "foreign" and k["origin"] == "declared"}
    for k in keys:
        d = ds_by_id[k["dataset_id"]]
        cols = loads(k["columns_json"], [])
        member_cols = [col_by_ds_name.get((d["dataset_id"], name)) for name in cols]
        if not cols or any(c is None for c in member_cols):
            continue  # key on a retired column: skip rather than publish a dangling constraint
        ref_ds, ref_col_rows, ref_cols = None, {}, []
        if k["key_type"] == "foreign":
            ref_cols = loads(k["referenced_columns_json"], [])
            ref_ds = db.query_one(datasets_select(db) + " WHERE d.dataset_id = ? AND d.active = 1",
                                  (k["referenced_dataset_id"],))
            if ref_ds is None:
                continue  # referenced dataset not cataloged (or retired)
            ref_col_rows = {r["name"]: r for r in db.query(
                "SELECT * FROM columns WHERE dataset_id = ? AND active = 1", (ref_ds["dataset_id"],))}
            if len(ref_cols) != len(cols) or any(rc not in ref_col_rows for rc in ref_cols):
                continue
        if k["origin"] == "declared":
            status = "verified"
        else:
            status = lookup(decisions, "foreign_key", k["key_id"]) if k["key_type"] == "foreign" else "inferred"
        knode = uri("key", k["key_id"])
        cls = {"primary": MESH.PrimaryKey, "unique": MESH.UniqueKey, "foreign": MESH.ForeignKey}[k["key_type"]]
        phys.add((knode, RDF.type, MESH.KeyConstraint))
        phys.add((knode, RDF.type, cls))
        phys.add((knode, MESH.onDataset, dataset_node(d)))
        phys.add((knode, MESH.origin, _lit(k["origin"])))
        phys.add((knode, MESH.validationStatus, _lit(status)))
        phys.add((knode, MESH.primaryKeyCandidate, _lit(bool(k["is_pk_candidate"]))))
        if k.get("constraint_name"):
            phys.add((knode, MESH.constraintName, _lit(k["constraint_name"])))
        for pos, c in enumerate(member_cols, start=1):
            m = URIRef(f"{knode}:column:{pos}")
            phys.add((m, RDF.type, MESH.KeyColumn))
            phys.add((m, MESH.position, _lit(pos)))
            phys.add((m, MESH.column, column_node(c, d)))
            phys.add((knode, MESH.keyColumn, m))
        if k["origin"] == "inferred":
            confidence = assess_foreign_key(db, k, fk_rules)  # FKs only; from the terms current at build time
            if confidence and _LEVELS[confidence["level"]] < _LEVELS[min_fk_confidence]:
                weak_fks.add(k["key_id"])
            if (k["dataset_id"], k["columns_json"], k["referenced_dataset_id"],
                    k["referenced_columns_json"]) in declared_sig:
                weak_fks.add(k["key_id"])  # profiling confirmed a declared key: same join, mapped once
            if confidence:
                phys.add((knode, MESH.fkConfidence, _lit(confidence["level"])))
                phys.add((knode, MESH.fkConfidenceBasis, _lit(confidence["basis"])))
                phys.add((knode, MESH.confidence, _lit(confidence["level"])))
            run = uri("profiling-run", k["profiling_run_id"])
            phys.add((knode, MESH.profilingRun, run))
            phys.add((knode, MESH.profilingScope, _lit(k["scope_json"] or "{}")))
            if run not in runs:
                runs.add(run)
                phys.add((run, RDF.type, MESH.ProfilingRun))
                phys.add((run, MESH.runId, _lit(k["profiling_run_id"])))
        if ref_ds is None:
            continue
        phys.add((knode, MESH.referencesDataset, dataset_node(ref_ds)))
        for pos, (fc, tc) in enumerate(zip(member_cols, ref_cols), start=1):
            mpos = URIRef(f"{knode}:mapping:{pos}")
            phys.add((mpos, RDF.type, MESH.ColumnMapping))
            phys.add((mpos, MESH.position, _lit(pos)))
            phys.add((mpos, MESH.fromColumn, column_node(fc, d)))
            phys.add((mpos, MESH.toColumn, column_node(ref_col_rows[tc], ref_ds)))
            phys.add((knode, MESH.columnMapping, mpos))
        fk_nodes[k["key_id"]] = (knode, status, d, ref_ds)
        # Schema-level navigational edge backed by the key (no business semantics; spec 10.5).
        dep_ent = entity_of.get(d["dataset_id"])
        ref_ent = entity_of.get(ref_ds["dataset_id"]) if ref_ds["dataset_id"] in ds_by_id else (
            (ref_ds["entity_slug"], ref_ds["entity_label"]) if ref_ds.get("entity_label") and lookup(
                decisions, "entity_mapping", ref_ds["dataset_id"], ref_ds.get("entity_slug")) != "rejected" else None)
        if dep_ent and ref_ent and dep_ent[0] != ref_ent[0]:
            rel = uri("relationship", f"fk-{k['key_id']}")
            mp.add((rel, RDF.type, MESH.Relationship))
            mp.add((rel, RDF.type, MESH.NavigationalEdge))
            mp.add((rel, MESH.sourceEntity, _entity(sem, dep_ent[1], dep_ent[0])))
            mp.add((rel, MESH.targetEntity, _entity(sem, ref_ent[1], ref_ent[0])))
            mp.add((rel, MESH.relationshipType, _lit("references")))
            mp.add((rel, RDFS.label, _lit(f"{dep_ent[1]} references {ref_ent[1]}")))
            mp.add((rel, PROV.wasDerivedFrom, knode))
            mp.add((rel, MESH.extractionMethod, _lit(f"{k['origin']}_foreign_key")))
            mp.add((rel, MESH.validationStatus, _lit(status)))

    # -- named relationships proposed by the relationship stage (semantic) + join mappings --------
    def entity_slug_of(dataset: dict[str, Any]) -> str | None:
        if dataset["dataset_id"] in ds_by_id:
            ent = entity_of.get(dataset["dataset_id"])
            return ent[0] if ent else None
        if dataset.get("entity_slug") and lookup(decisions, "entity_mapping", dataset["dataset_id"],
                                                 dataset["entity_slug"]) != "rejected":
            return dataset["entity_slug"]
        return None

    proposals: dict[str, list[dict[str, Any]]] = {}
    for r in db.query("SELECT * FROM semantic_relationships WHERE source_id = ? AND active = 1 "
                      "ORDER BY relationship_id, key_id", (source_id,)):
        proposals.setdefault(r["relationship_id"], []).append(r)
    for rid, rows in proposals.items():
        explicit = [r for r in rows if r["key_id"] in fk_nodes]
        if not explicit:
            continue  # every FK it was proposed for is gone: nothing supports it any more
        r0 = rows[0]
        status = lookup(decisions, "relationship", rid)
        node = uri("relationship", f"sem-{rid}")
        src_slug, tgt_slug = r0["source_entity_slug"], r0["target_entity_slug"]
        src_label = entity_labels.get(src_slug, r0["source_entity_label"])
        tgt_label = entity_labels.get(tgt_slug, r0["target_entity_label"])
        src, tgt = _entity(sem, src_label, src_slug), _entity(sem, tgt_label, tgt_slug)
        sem.add((node, RDF.type, MESH.Relationship))
        sem.add((node, RDF.type, MESH.SemanticRelationship))
        sem.add((node, MESH.sourceEntity, src))
        sem.add((node, MESH.targetEntity, tgt))
        sem.add((node, MESH.relationshipType, _lit(r0["name"])))
        sem.add((node, RDFS.label, _lit(f"{src_label} {r0['name']} {tgt_label}")))
        sem.add((node, MESH.relationshipId, _lit(rid)))
        sem.add((node, MESH.validationStatus, _lit(status)))
        sem.add((node, MESH.extractionMethod, _lit("llm_proposal")))
        if r0.get("definition"):
            sem.add((node, SKOS.definition, _lit(r0["definition"])))
        for syn in loads(r0.get("synonyms_json"), []) or []:
            sem.add((node, SKOS.altLabel, _lit(syn)))
        if r0.get("inverse_name"):
            sem.add((node, MESH.inverseName, _lit(r0["inverse_name"])))
        if r0.get("cardinality"):
            sem.add((node, MESH.cardinality, _lit(r0["cardinality"])))
        run = uri("relationship-run", r0["run_id"] or "unknown")
        sem.add((run, RDF.type, MESH.RelationshipRun))
        sem.add((run, MESH.runId, _lit(r0["run_id"] or "unknown")))
        sem.add((node, PROV.wasGeneratedBy, run))
        prop = predicate_uri(rid)
        sem.add((prop, RDF.type, OWL.ObjectProperty))
        sem.add((prop, RDFS.label, _lit(r0["name"])))
        sem.add((prop, RDFS.domain, src))
        sem.add((prop, RDFS.range, tgt))
        sem.add((node, MESH.objectProperty, prop))
        explicit_keys = set()
        for r in explicit:
            knode, key_status = fk_nodes[r["key_id"]][:2]
            explicit_keys.add(r["key_id"])
            sem.add((node, PROV.wasDerivedFrom, knode))
            _join_mapping(mp, node, rid, r["key_id"], knode, "explicit", r["key_direction"],
                          _join_status(status, key_status, "explicit"))
        for key_id, (knode, key_status, dep, ref) in fk_nodes.items():
            if key_id in explicit_keys or key_id in weak_fks:
                continue
            dep_slug, ref_slug = entity_slug_of(dep), entity_slug_of(ref)
            if dep_slug and ref_slug and dep_slug != ref_slug and {dep_slug, ref_slug} == {src_slug, tgt_slug}:
                direction = "forward" if dep_slug == src_slug else "reverse"
                _join_mapping(mp, node, rid, key_id, knode, "automatic", direction,
                              _join_status(status, key_status, "automatic"))

    # -- term relations (semantic, descriptive) ----------------------------------------------------
    seen_relations: set[str] = set()
    for tr in db.query("SELECT * FROM term_relations WHERE source_id = ? AND active = 1 ORDER BY relation_id",
                       (source_id,)):
        if tr["relation_id"] in seen_relations:
            continue
        seen_relations.add(tr["relation_id"])
        node = uri("term-relation", tr["relation_id"])
        subj_label = term_labels.get(tr["subject_slug"], tr["subject_label"])
        obj_label = term_labels.get(tr["object_slug"], tr["object_label"])
        subj, obj = _term(sem, subj_label, tr["subject_slug"]), _term(sem, obj_label, tr["object_slug"])
        sem.add((node, RDF.type, MESH.TermRelation))
        sem.add((node, MESH.subjectTerm, subj))
        sem.add((node, MESH.objectTerm, obj))
        sem.add((node, MESH.predicate, term_predicate(tr["predicate"])))
        sem.add((node, MESH.relationId, _lit(tr["relation_id"])))
        sem.add((node, RDFS.label, _lit(f"{subj_label} {tr['predicate']} {obj_label}")))
        sem.add((node, MESH.validationStatus, _lit(lookup(decisions, "term_relation", tr["relation_id"]))))
        sem.add((node, MESH.extractionMethod, _lit("llm_proposal")))
        sem.add((node, MESH.assertionLevel, _lit("descriptive")))
        if tr.get("definition"):
            sem.add((node, SKOS.definition, _lit(tr["definition"])))
        run = uri("relationship-run", tr["run_id"] or "unknown")
        sem.add((run, RDF.type, MESH.RelationshipRun))
        sem.add((run, MESH.runId, _lit(tr["run_id"] or "unknown")))
        sem.add((node, PROV.wasGeneratedBy, run))
    return layers


def _join_mapping(g: Graph, rel: URIRef, rid: str, key_id: str, knode: URIRef, method: str, direction: str,
                  status: str) -> None:
    node = uri("join-mapping", stable_hash(rid, key_id))
    g.add((node, RDF.type, MESH.JoinMapping))
    g.add((node, MESH.relationship, rel))
    g.add((node, MESH.foreignKey, knode))
    g.add((node, MESH.mappingMethod, _lit(method)))
    g.add((node, MESH.keyDirection, _lit(direction)))
    g.add((node, MESH.validationStatus, _lit(status)))
    g.add((rel, MESH.joinMapping, node))


def build_source_context(db: Database, source_id: str, fk_rules: FKRules | None = None) -> Graph:
    """All layers of a source in one graph (validation, export)."""
    g = _g()
    for layer in build_source_layers(db, source_id, fk_rules).values():
        for triple in layer:
            g.add(triple)
    return g


# -- business glossary -------------------------------------------------------------------
def build_glossary_context(db: Database) -> Graph:
    """Every active business term and entity of the business glossary with its label, description and
    (terms) sensitive flag, including terms bound to no column yet."""
    g = _g()
    for t in db.query("SELECT slug, label, description, sensitive, origin FROM business_terms WHERE deleted = 0"):
        node = _term(g, t["label"], t["slug"])
        if t["description"]:
            g.add((node, SKOS.definition, _lit(t["description"])))
        if t["sensitive"] is not None:
            g.add((node, MESH.sensitive, _lit(bool(t["sensitive"]))))
        g.add((node, MESH.origin, _lit(t["origin"])))
    for e in db.query("SELECT slug, label, description, origin FROM business_entities WHERE deleted = 0"):
        node = _entity(g, e["label"], e["slug"])
        if e["description"]:
            g.add((node, SKOS.definition, _lit(e["description"])))
        g.add((node, MESH.origin, _lit(e["origin"])))
    return g


# -- definitions -------------------------------------------------------------------------
def build_definition_context(definition_id: str, name: str, text: str, author: str | None,
                             document_id: str | None, location: str | None, terms: list[tuple[str, str]],
                             created_at: str) -> Graph:
    """A business definition: it describes business terms (``terms``: (slug, label)) and, optionally, is
    defined in a policy document (``mesh:definedIn``, optionally at ``location``: page or section)."""
    g = _g()
    node = uri("definition", definition_id)
    g.add((node, RDF.type, MESH.BusinessDefinition))
    g.add((node, RDFS.label, _lit(name)))
    g.add((node, MESH.definitionText, _lit(text)))
    if author:
        g.add((node, MESH.author, _lit(author)))
    if document_id:
        g.add((node, MESH.definedIn, uri("document", document_id)))
    if location:
        g.add((node, MESH.location, _lit(location)))
    g.add((node, DCTERMS.created, _lit(created_at)))
    for slug, label in terms:
        g.add((node, MESH.describes, _term(g, label, slug)))
    return g


__all__ = ["build_document_context", "build_source_layers", "build_source_context", "build_definition_context",
           "build_glossary_context", "dumps"]

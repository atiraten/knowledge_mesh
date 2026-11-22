"""RDF knowledge graph model: catalog provisioning, layers, OWL/SKOS, statuses and reviews (AT-17, AT-18)."""

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL

from knowledge_mesh.common.errors import GraphValidationError, InvalidArgument
from knowledge_mesh.common.ids import dataset_id_for
from knowledge_mesh.graph.builders import build_source_context
from knowledge_mesh.graph.ontology import (GLOSSARY_SCHEME, MESH, ONTOLOGY_CONTEXT, PROV, RDF, RDFS, SKOS, context_uri,
                                           layer_context_uri)
from knowledge_mesh.structured.glossary_store import columns_select, datasets_select
from knowledge_mesh.structured.relationships import RelationshipGenerationService
from tests.kg_support import KG_OVERRIDES, shop_ids, shop_mesh, shop_source  # noqa: F401

E = lambda slug: URIRef(f"urn:mesh:entity:{slug}")  # noqa: E731
T = lambda slug: URIRef(f"urn:mesh:term:{slug}")  # noqa: E731


def _semantic(mesh, name):
    ds = mesh.graph.dataset
    return next(r for r in ds.subjects(RDF.type, MESH.SemanticRelationship)
                if str(ds.value(r, MESH.relationshipType)) == name)


def test_catalog_links_every_table_to_an_entity_and_every_column_to_a_term(mesh_factory, config_factory,
                                                                           sqlite_source, parquet_source,
                                                                           fake_term_llm):
    """AT-17: the configured sources are scanned; tables -> entities, columns -> glossary terms, shared ones too."""
    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source, parquet_source], **KG_OVERRIDES}),
                        term_llm=fake_term_llm)
    for source in ("crm", "export"):
        assert mesh.catalog_source(source, wait_seconds=120)["status"] == "succeeded"
    datasets = mesh.db.query(datasets_select(mesh.db) + " WHERE d.active = 1")  # entity from the glossary
    assert {d["source_id"] for d in datasets} == {"crm", "export"}
    assert all(d["entity_label"] for d in datasets)
    columns = mesh.db.query(columns_select(mesh.db) + " WHERE c.active = 1")  # term from the glossary
    assert columns and all(c["term_status"] == "generated" and c["term_label"] for c in columns)
    ds = mesh.graph.dataset
    # two tables of two sources share the Customer entity
    customer_tables = {str(ds.value(t, MESH.datasetId)) for t in ds.subjects(MESH.representsEntity, E("customer"))}
    cust = mesh.db.query_one("SELECT dataset_id FROM datasets WHERE source_id = 'export' AND name = 'cust'")
    assert {dataset_id_for("crm", "main", "customers"), cust["dataset_id"]} <= customer_tables
    # several columns share the Customer Identifier term
    bound = {str(ds.value(c, RDFS.label)) for c in ds.objects(T("customer-identifier"), MESH.boundToColumn)}
    assert {"customer_id", "cust_id"} <= bound
    for d in datasets:  # every table is linked in the graph
        assert (URIRef(f"urn:mesh:dataset:{d['dataset_id']}"), MESH.representsEntity, None) in ds


def test_graph_provisioning_references_the_catalog(shop_mesh):
    """AT-18: tables, columns, keys (primary, foreign, unique, composite), entities and terms, with the links."""
    mesh = shop_mesh()
    ds, ids = mesh.graph.dataset, shop_ids()
    for layer in ("physical", "semantic", "mapping"):
        assert mesh.graph.context_triples(layer_context_uri("shop", layer)) > 0
    assert len(ds.graph(context_uri("source", "shop"))) == 0  # layered, not the single context
    assert mesh.graph.context_triples(context_uri("source", "shop")) > 0  # the source's layers together
    for table, dataset_id in ids.items():
        node = URIRef(f"urn:mesh:dataset:{dataset_id}")
        assert (node, RDF.type, MESH.Dataset) in ds
        row = mesh.db.query_one("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        assert (node, MESH.representsEntity, E(row["entity_slug"])) in ds
        for c in mesh.db.query("SELECT * FROM columns WHERE dataset_id = ? AND active = 1", (dataset_id,)):
            cnode = URIRef(f"urn:mesh:column:{c['column_id']}")
            assert (node, MESH.hasColumn, cnode) in ds and (cnode, MESH.inDataset, node) in ds
            assert (cnode, MESH.hasBusinessTerm, T(c["term_slug"])) in ds
            assert (cnode, MESH.dataType, None) in ds
    keys = mesh.db.query("SELECT * FROM keys WHERE active = 1")
    for k in keys:
        assert (URIRef(f"urn:mesh:key:{k['key_id']}"), RDF.type, MESH.KeyConstraint) in ds
    kinds = {(k["key_type"], k["origin"]) for k in keys}
    assert {("primary", "declared"), ("foreign", "declared")} <= kinds
    # composite FK shipments(order_id, line_no) -> order_lines(order_id, line_no): two ordered column pairs
    fk = next(k for k in keys if k["dataset_id"] == ids["shipments"] and k["key_type"] == "foreign"
              and k["origin"] == "declared")
    knode = URIRef(f"urn:mesh:key:{fk['key_id']}")
    pairs = sorted((int(ds.value(m, MESH.position)), str(ds.value(ds.value(m, MESH.fromColumn), RDFS.label)),
                    str(ds.value(ds.value(m, MESH.toColumn), RDFS.label))) for m in ds.objects(knode, MESH.columnMapping))
    assert pairs == [(1, "order_id", "order_id"), (2, "line_no", "line_no")]
    assert str(ds.value(knode, MESH.validationStatus)) == "verified"


def test_owl_skos_model_keeps_concepts_properties_and_relationships_apart(shop_mesh):
    mesh = shop_mesh()
    ds = mesh.graph.dataset
    assert (E("customer"), RDF.type, OWL.Class) in ds and (E("customer"), RDF.type, MESH.BusinessEntity) in ds
    term = T("birth-date")
    assert (term, RDF.type, SKOS.Concept) in ds and (term, SKOS.inScheme, GLOSSARY_SCHEME) in ds
    assert str(ds.value(term, SKOS.prefLabel)) == "Birth Date"
    assert {str(x) for x in ds.objects(term, SKOS.altLabel)} == {"date of birth", "DOB"}
    assert (term, MESH.termOfEntity, E("customer")) in ds
    prop = ds.value(term, MESH.termProperty)
    assert prop != term and (prop, RDF.type, OWL.DatatypeProperty) in ds and (prop, RDFS.domain, E("customer")) in ds
    assert (term, RDF.type, OWL.DatatypeProperty) not in ds  # the SKOS term is not the OWL property
    rel = _semantic(mesh, "placesOrder")
    assert ds.value(rel, MESH.sourceEntity) == E("customer") and ds.value(rel, MESH.targetEntity) == E("order")
    assert str(ds.value(rel, MESH.validationStatus)) == "inferred"
    assert {str(x) for x in ds.objects(rel, SKOS.altLabel)} >= {"bought", "purchases"}
    obj = ds.value(rel, MESH.objectProperty)
    assert (obj, RDF.type, OWL.ObjectProperty) in ds and ds.value(obj, RDFS.domain) == E("customer")
    assert (rel, PROV.wasGeneratedBy, None) in ds and (rel, PROV.wasDerivedFrom, None) in ds
    joins = list(ds.objects(rel, MESH.joinMapping))
    assert joins and {str(ds.value(j, MESH.mappingMethod)) for j in joins} == {"explicit"}
    nav = [r for r in ds.subjects(RDF.type, MESH.NavigationalEdge)]
    assert nav and all(str(ds.value(r, MESH.relationshipType)) == "references" for r in nav)
    tr = next(ds.subjects(RDF.type, MESH.TermRelation))
    assert ds.value(tr, MESH.predicate) == MESH.expectedAfter
    assert ds.value(tr, MESH.subjectTerm) == T("settlement-date") and ds.value(tr, MESH.objectTerm) == T("transaction-date")
    assert str(ds.value(tr, MESH.assertionLevel)) == "descriptive"
    assert str(ds.value(tr, MESH.validationStatus)) == "inferred"


def test_earlier_single_context_layout_is_migrated_and_vocabulary_published(shop_mesh):
    mesh = shop_mesh()
    mesh.graph.remove_source("shop")
    mesh.graph.publish(context_uri("source", "shop"), build_source_context(mesh.db, "shop"))  # earlier layout
    assert mesh.graph.legacy_source_contexts() == ["shop"]
    out = mesh.provision_graph()
    assert out["sources"]["shop"]["status"] == "published" and out["ontology"]["status"] == "published"
    assert mesh.graph.legacy_source_contexts() == []
    assert len(mesh.graph.dataset.graph(context_uri("source", "shop"))) == 0
    assert mesh.graph.context_triples(layer_context_uri("shop", "semantic")) > 0
    assert (MESH.SemanticRelationship, RDFS.subClassOf, MESH.Relationship) in mesh.graph.dataset.graph(ONTOLOGY_CONTEXT)
    assert mesh.provision_graph()["ontology"]["status"] == "unchanged"


def test_invalid_layers_are_rejected_and_the_previous_graph_kept(shop_mesh):
    mesh = shop_mesh()
    before = mesh.graph.context_triples(context_uri("source", "shop"))
    bad = Graph()
    jm = URIRef("urn:mesh:join-mapping:bad")
    bad.add((jm, RDF.type, MESH.JoinMapping))
    bad.add((jm, MESH.mappingMethod, Literal("guessed")))
    with pytest.raises(GraphValidationError) as err:
        mesh.graph.publish_layers("shop", {"mapping": bad})
    assert any(v["focus_node"] == str(jm) for v in err.value.details["violations"])
    assert mesh.graph.context_triples(context_uri("source", "shop")) == before


def test_reviews_change_status_survive_reruns_and_rejected_items_leave_traversal(shop_mesh):
    mesh = shop_mesh()
    ds = mesh.graph.dataset
    rel_id = mesh.db.query_one("SELECT relationship_id FROM semantic_relationships WHERE name = 'placesOrder' "
                               "AND active = 1")["relationship_id"]
    out = mesh.review("relationship", rel_id, "verified", "steward")
    assert out["graph"]["status"] == "published"
    assert str(ds.value(_semantic(mesh, "placesOrder"), MESH.validationStatus)) == "verified"
    # the decision survives a new proposal run (stable relationship id)
    ids = shop_ids()
    assert mesh.generate_relationships([ids["orders"]], wait_seconds=60)["status"] == "succeeded"
    assert str(mesh.graph.dataset.value(_semantic(mesh, "placesOrder"), MESH.validationStatus)) == "verified"
    # rejected: kept in the graph with its status, never traversed
    mesh.review("relationship", rel_id, "rejected", "steward", "wrong direction")
    res = mesh.query_knowledge_graph({"intent": "customer bought product", "retrieval_queries": ["customer bought product"],
                                      "top_k": 20})
    assert all(e["predicate"] != "placesOrder" for p in res["paths"] for e in p["edges"])
    mesh.review("relationship", rel_id, "inferred", "steward")
    assert str(mesh.graph.dataset.value(_semantic(mesh, "placesOrder"), MESH.validationStatus)) == "inferred"
    # mapping reviews: a rejected column -> term link is no longer asserted
    col = mesh.db.query_one("SELECT column_id FROM columns WHERE dataset_id = ? AND name = 'birth_date'",
                            (ids["customers"],))["column_id"]
    mesh.review("term_mapping", col, "rejected", "steward")
    cnode = URIRef(f"urn:mesh:column:{col}")
    assert (cnode, MESH.hasBusinessTerm, None) not in mesh.graph.dataset
    binding = URIRef(f"urn:mesh:term-binding:{col}")
    assert str(mesh.graph.dataset.value(binding, MESH.validationStatus)) == "rejected"
    assert mesh.get_dataset(ids["customers"])["columns"][2]["business_term"]["status"] == "rejected"
    mesh.review("entity_mapping", ids["trades"], "verified", "steward")
    assert mesh.get_dataset(ids["trades"])["business_entity"]["status"] == "verified"
    audit = mesh.db.query("SELECT * FROM review_decisions")
    assert {(a["object_type"], a["status"]) for a in audit} >= {("term_mapping", "rejected"),
                                                                 ("entity_mapping", "verified")}


def test_declared_keys_cannot_be_reviewed_and_inferred_keys_can(shop_mesh):
    mesh = shop_mesh()
    declared = mesh.db.query_one("SELECT key_id FROM keys WHERE key_type = 'foreign' AND origin = 'declared'")
    with pytest.raises(InvalidArgument):
        mesh.review("foreign_key", declared["key_id"], "rejected", "steward")
    inferred = mesh.db.query_one("SELECT key_id FROM keys WHERE key_type = 'foreign' AND origin = 'inferred' "
                                 "AND active = 1")
    assert inferred, "BINDER found no inferred FK in the shop fixture"
    mesh.review("foreign_key", inferred["key_id"], "rejected", "steward")
    node = URIRef(f"urn:mesh:key:{inferred['key_id']}")
    assert str(mesh.graph.dataset.value(node, MESH.validationStatus)) == "rejected"
    assert all(e.get("status") != "rejected" for e in mesh.graph.evidence_for([E("order-line")], 100))


def test_relationship_proposals_are_validated(shop_mesh):
    mesh = shop_mesh(catalog=True)
    svc: RelationshipGenerationService = mesh.relationships
    ctx = svc._context(shop_ids()["support_tickets"])
    assert len(ctx["fks"]) == 2
    fk = next(f["ref"] for f in ctx["fks"] if f["ref_entity"] == "Customer")
    text = """{"relationships": [
        {"fk": "FK9", "source_entity": "Customer", "name": "x", "target_entity": "Support Ticket"},
        {"fk": "@", "source_entity": "Order", "name": "wrongEntities", "target_entity": "Support Ticket"},
        {"fk": "@", "source_entity": "Customer", "name": "DROP TABLE; --", "target_entity": "Support Ticket"},
        {"fk": "@", "source_entity": "Customer", "name": "opensTicket", "target_entity": "Support Ticket",
         "synonyms": ["complains", "complains", "opensTicket"], "cardinality": "lots"}],
      "term_relations": [{"subject": "Ticket Opened Date", "predicate": "before", "object": "Support Ticket Identifier"},
                         {"subject": "Ticket Opened Date", "predicate": "dependsOn", "object": "Birth Date"}],
      "term_synonyms": [{"term": "Unknown Term", "synonyms": ["x"]}]}"""
    proposals, issues = svc.parse(text.replace('"@"', f'"{fk}"'), ctx)
    assert [r["name"] for r in proposals["relationships"]] == ["opensTicket"]
    r = proposals["relationships"][0]
    assert r["synonyms"] == ["complains"] and r["cardinality"] is None and r["key_direction"] == "reverse"
    assert proposals["term_relations"] == [] and proposals["term_synonyms"] == []
    assert len(issues) == 6
    with pytest.raises(ValueError):
        svc.parse("not json", ctx)

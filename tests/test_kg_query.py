"""query_knowledge_graph: resolution, multi-hop paths, ranking, mappings, limits and scope (AT-05..AT-13, AT-16)."""

import json

import pytest

from knowledge_mesh.common.errors import InvalidArgument
from knowledge_mesh.graph.builders import build_document_context
from knowledge_mesh.graph.ontology import context_uri
from knowledge_mesh.unstructured.entities import Mention, Relationship
from tests.kg_support import CUSTOMERS, shop_ids, shop_mesh, shop_source  # noqa: F401

PURCHASE = ["urn:mesh:entity:customer", "urn:mesh:entity:order", "urn:mesh:entity:order-line",
            "urn:mesh:entity:product"]


def q(mesh, *queries, **kw):
    return mesh.query_knowledge_graph({"intent_id": "I1", "intent": queries[0], "retrieval_queries": list(queries), **kw})


@pytest.fixture
def shop(shop_mesh):
    return shop_mesh()


def test_entity_lookup_returns_stable_iri_and_mappings(shop):
    """AT-05."""
    res = q(shop, "customer")
    assert res["status"] == "ok" and res["intent_id"] == "I1"
    ent = res["business_entities"][0]
    assert ent["iri"] == "urn:mesh:entity:customer" and ent["name"] == "Customer" and ent["role"] == "matched"
    assert "owl:Class" in ent["classification"]
    assert [t["name"] for t in ent["tables"]] == ["customers"]
    assert {t["name"] for t in res["physical_mappings"]["tables"]} == {"customers"}
    assert any(c["type"] == "primary_key" and c["table"] == "customers" for c in res["constraints"])
    # direct lookup by IRI and case-insensitive labels
    assert q(shop, "urn:mesh:entity:customer")["business_entities"][0]["match_type"] == "iri"
    assert q(shop, "CUSTOMERS")["business_entities"][0]["iri"] == "urn:mesh:entity:customer"


def test_term_lookup_resolves_owner_columns_and_synonyms(shop):
    """AT-06 / spec 6.2: 'customer date of birth' -> Birth Date of Customer, its column and type; synonyms work."""
    res = q(shop, "customer date of birth")
    term = next(t for t in res["business_terms"] if t["role"] == "matched")
    assert term["iri"] == "urn:mesh:term:birth-date" and term["entity_iri"] == "urn:mesh:entity:customer"
    assert term["match_type"] in ("alias", "token_set")
    assert [(c["table"], c["name"], c["data_type"]) for c in term["columns"]] == [("customers", "birth_date", "TEXT")]
    assert term["property_iri"] == "urn:mesh:property:birth-date"
    assert set(term["synonyms"]) == {"date of birth", "DOB"}
    assert q(shop, "DOB of our customers")["business_terms"][0]["iri"] == "urn:mesh:term:birth-date"
    # the planner's formulations keep the constraint; the graph returns metadata, never rows
    res = q(shop, "list customer born before 1990", "customers birth date before 1990")
    assert res["matched_queries"] == ["list customer born before 1990", "customers birth date before 1990"]
    exact = [t["iri"] for t in res["business_terms"] if t["role"] == "matched" and t["match_type"] != "semantic"]
    assert exact == ["urn:mesh:term:birth-date"]  # approximate (semantic) matches may come in addition
    text = json.dumps(res)
    assert not any(name in text for _, name, _ in CUSTOMERS) and "1985-03-02" not in text


def test_multi_hop_path_through_bridge_entities(shop):
    """AT-07: 'customer bought product' -> Customer -placesOrder-> Order -containsOrderLine-> Order Line
    -referencesProduct-> Product, with predicates and intermediate nodes."""
    res = q(shop, "customer bought product")
    top = res["paths"][0]
    assert top["nodes"] == PURCHASE and top["hops"] == 3
    assert [e["predicate"] for e in top["edges"]] == ["placesOrder", "containsOrderLine", "referencesProduct"]
    assert all(e["direction"] == "forward" and e["kind"] == "semantic" for e in top["edges"])
    assert top["executable"] and top["missing_mappings"] == []
    assert top["score_explanation"]["calibrated"] is False and 0 < top["relevance_score"] <= 1
    assert {"urn:mesh:entity:order", "urn:mesh:entity:order-line"} <= {e["iri"] for e in res["business_entities"]
                                                                      if e["role"] == "path"}
    rels = {r["iri"]: r for r in res["relationships"]}
    assert all(e["relationship_iri"] in rels for e in top["edges"])
    assert rels[top["edges"][0]["relationship_iri"]]["predicate_iri"].startswith("urn:mesh:predicate:")
    joins = {j["join_id"]: j for j in res["physical_mappings"]["joins"]}
    first = joins[top["edges"][0]["join_refs"][0]]
    assert first["join_predicate"] == "main.orders.customer_id = main.customers.customer_id"
    assert first["origin"] == "declared" and first["method"] == "explicit"


def test_purchase_path_outranks_support_ticket_path(shop_mesh):
    """AT-08: for 'bought', the purchase route ranks above the shorter support-ticket route."""
    shop = shop_mesh({"knowledge_graph": {"max_response_chars": 500000}})
    res = q(shop, "customer bought product", top_k=20)
    labels = [[e["predicate"] for e in p["edges"]] for p in res["paths"]]
    purchase = labels.index(["placesOrder", "containsOrderLine", "referencesProduct"])
    support = labels.index(["opensTicket", "concernsProduct"])
    assert purchase < support
    assert res["paths"][purchase]["hops"] > res["paths"][support]["hops"]  # not the shortest path blindly
    assert res["paths"][purchase]["relevance_score"] > res["paths"][support]["relevance_score"]


def test_composite_fk_is_returned_as_ordered_column_pairs(shop):
    """AT-09."""
    res = q(shop, "shipment of an order line")
    join = next(j for j in res["physical_mappings"]["joins"] if j["relationship"] == "shipsOrderLine"
                and j["origin"] == "declared")
    assert [(p["position"], p["from_column"], p["to_column"]) for p in join["column_pairs"]] == [
        (1, "order_id", "order_id"), (2, "line_no", "line_no")]
    assert join["join_predicate"] == ("main.shipments.order_id = main.order_lines.order_id AND "
                                      "main.shipments.line_no = main.order_lines.line_no")


def test_missing_join_mapping_gives_partial_evidence_and_a_warning(shop):
    """AT-10: a document relationship has no join: the path is returned, not executable, with a warning."""
    doc = {"document_id": "doc-reviews", "source_uri": "file://reviews.md", "title": "Reviews"}
    rel = Relationship("rel-doc-1", "chunk-1", "Customer", "customer", "reviews", "Product", "product",
                       "A customer reviews a product after delivery.", "lexicon")
    graph = build_document_context(doc, {"chunk-1": {}}, [Mention("chunk-1", "Customer", "customer")], [rel])
    shop.graph.publish(context_uri("document", "doc-reviews"), graph)
    res = q(shop, "customer reviews product", top_k=20)
    path = next(p for p in res["paths"] if [e["predicate"] for e in p["edges"]] == ["reviews"])
    assert path["executable"] is False and path["missing_mappings"] == [path["edges"][0]["relationship_iri"]]
    assert path["edges"][0]["join_refs"] == [] and path["edges"][0]["kind"] == "document"
    assert any("not executable for SQL" in w and "reviews" in w for w in res["warnings"])
    joins = res["physical_mappings"]["joins"]
    assert all(j["relationship"] != "reviews" for j in joins)  # no invented join


def test_schema_paths_are_not_record_level_facts(shop):
    """AT-11."""
    res = q(shop, "which customer bought which product")
    assert res["assertion_level"] == "schema" and "execute_sql" in res["note"]
    text = json.dumps(res)
    assert "Ann Lee" not in text and "Phone" not in text and "rows" not in res


def test_unknown_concept_is_no_match(shop):
    """AT-12."""
    res = q(shop, "zebra migration patterns")
    assert res["status"] == "no_match"
    assert res["business_entities"] == res["business_terms"] == res["paths"] == []
    assert res["physical_mappings"] == {"tables": [], "columns": [], "joins": []}
    assert any("zebra migration patterns" in w for w in res["warnings"])


def test_limits_and_scope_are_enforced(shop_mesh):
    """AT-13: max_hops, top_k, candidate/expansion/time limits, response size and the access scope."""
    mesh = shop_mesh({"knowledge_graph": {"max_response_chars": 4000}})
    short = q(mesh, "customer bought product", max_hops=2, top_k=20)
    assert all(p["hops"] <= 2 for p in short["paths"])
    assert all(p["nodes"] != PURCHASE for p in short["paths"])
    assert len(q(mesh, "customer bought product", top_k=1)["paths"]) == 1
    with pytest.raises(InvalidArgument):
        q(mesh, "customer bought product", max_hops=9)
    big = q(mesh, "customer bought product", top_k=5)
    assert len(json.dumps(big)) <= 4000 and big["truncated"] and big["paths"]
    # scope: another source / another dataset only
    assert q(mesh, "customer", **{}) and mesh.query_knowledge_graph(
        {"intent": "customer"}, structured_source_ids=["crm"])["status"] == "no_match"
    scoped = mesh.query_knowledge_graph({"intent": "customer bought product", "top_k": 20},
                                        dataset_ids=[shop_ids()["customers"], shop_ids()["orders"]])
    assert {t["name"] for t in scoped["physical_mappings"]["tables"]} <= {"customers", "orders"}
    assert all("main.order_lines" not in j["join_predicate"] for j in scoped["physical_mappings"]["joins"])

    tight = shop_mesh({"knowledge_graph": {"max_candidate_paths": 1, "max_expansions": 3}})
    res = q(tight, "customer bought product")
    assert any("max_candidate_paths" in w or "max_expansions" in w for w in res["warnings"])
    slow = shop_mesh({"knowledge_graph": {"query_timeout_seconds": 0.000001}})
    assert any("timeout" in w for w in q(slow, "customer bought product")["warnings"])


def test_term_rule_is_represented_without_asserting_records(shop):
    """AT-16: 'Settlement Date expected after Transaction Date' is descriptive, not a validated fact."""
    res = q(shop, "settlement date and transaction date of trades")
    rule = next(c for c in res["constraints"] if c["type"] == "term_relation")
    assert (rule["subject"], rule["predicate"], rule["object"]) == ("Settlement Date", "expectedAfter",
                                                                     "Transaction Date")
    assert rule["assertion_level"] == "descriptive" and rule["status"] == "inferred"
    assert "not validated against records" in rule["note"]
    term = next(t for t in res["business_terms"] if t["name"] == "Settlement Date")
    assert term["related_terms"][0]["predicate"] == "expectedAfter"
    from rdflib.namespace import SH
    assert (None, None, SH.ValidationResult) not in shop.graph.dataset  # no rule engine ran on data


def test_results_merge_across_queries_and_keep_the_matching_queries(shop):
    res = q(shop, "customer orders", "orders placed by customers")
    ent = {e["iri"]: e for e in res["business_entities"]}
    assert ent["urn:mesh:entity:customer"]["matched_queries"] == ["customer orders", "orders placed by customers"]
    assert len([e for e in res["business_entities"] if e["iri"] == "urn:mesh:entity:customer"]) == 1


def test_inferred_relationships_can_be_excluded(shop_mesh):
    mesh = shop_mesh({"knowledge_graph": {"include_inferred_relationships": False}})
    res = q(mesh, "customer bought product", top_k=20)
    assert res["paths"] and all(e["status"] == "verified" for p in res["paths"] for e in p["edges"])
    assert all(e["kind"] == "navigational" for p in res["paths"] for e in p["edges"])
    for name in ("placesOrder", "containsOrderLine", "referencesProduct"):
        rid = mesh.db.query_one("SELECT relationship_id FROM semantic_relationships WHERE name = ? AND active = 1",
                                (name,))["relationship_id"]
        mesh.review("relationship", rid, "verified", "steward")
    top = q(mesh, "customer bought product")["paths"][0]
    assert top["nodes"] == PURCHASE and top["validation_status"] == "verified"


def test_retrieve_context_lists_the_new_relationships_with_their_status(shop):
    res = shop.retrieve_context("customer places order", evidence_types=["graph", "glossary"])
    rels = [g for g in res["graph_evidence"] if g["type"] == "relationship" and "placesOrder" in g["label"]]
    assert rels and rels[0]["status"] == "inferred" and "status inferred" in rels[0]["text"] and rels[0]["inferred"]
    res = shop.retrieve_context("settlement date", evidence_types=["graph", "glossary"])
    assert any(g["type"] == "term_relation" and "expectedAfter" in g["label"] for g in res["graph_evidence"])

"""Business glossary (Definitions menu): business terms and entities stored once, edited in place.

Terms and entities have their own rows (``business_terms`` / ``business_entities``); columns and datasets
reference them by slug. Edits keep the slug, are republished to the graph and the glossary index, and survive
term generation; deletes follow the rules of ``structured/glossary_store.py``.
"""

import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.api.app import create_app
from knowledge_mesh.common.errors import Conflict, InvalidArgument, NotFound
from knowledge_mesh.graph.ontology import MESH, RDFS, SKOS, entity_uri, term_uri
from knowledge_mesh.storage.db import Database
from knowledge_mesh.structured.glossary_store import GlossaryStore, columns_select

POLICY = ("# Customer policy\nEach customer places orders through the web store.\n"
          "## Orders\nAn order belongs to one customer.\n")


@pytest.fixture
def crm(mesh_factory, config_factory, sqlite_source, fake_term_llm):
    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source]}), term_llm=fake_term_llm)
    assert mesh.catalog_source("crm", wait_seconds=120)["status"] == "succeeded"
    return mesh


def labels(mesh, node) -> set[str]:
    return {str(o) for o in mesh.graph.dataset.objects(node, RDFS.label)}


def dataset_id(mesh, name: str) -> str:
    return mesh.db.query_one("SELECT dataset_id FROM datasets WHERE name = ?", (name,))["dataset_id"]


def test_fill_from_the_per_column_copies(tmp_path):
    db = Database(tmp_path / "m.sqlite")
    db.execute("INSERT INTO datasets(dataset_id, source_id, name, catalog_ref, entity_label, entity_slug) VALUES"
               " ('d1','s','customers','c:1','Customer','customer'), ('d2','s','clients','c:2','Customer','customer')")
    db.execute("INSERT INTO columns(column_id, dataset_id, name, catalog_ref, term_label, term_slug, term_description,"
               " term_status, term_sensitive) VALUES"
               " ('c1','d1','customer_id','c:1/a','Customer Identifier','customer-identifier','Id of a customer','generated',1),"
               " ('c2','d2','cust','c:2/b','Customer Identifier','customer-identifier','Id of a customer','generated',NULL),"
               " ('c3','d2','cid','c:2/c','Customer Identifier','customer-identifier','Other text','generated',0)")
    db.execute("INSERT INTO doc_entity_mentions(document_id, chunk_id, entity_slug, entity_label)"
               " VALUES ('doc','ch','policy','Policy')")
    store = GlossaryStore(db)
    assert store.fill_if_empty() == {"terms": 1, "entities": 2, "label_conflicts": 1}
    term = store.term("customer-identifier")  # most common description; sensitive if any column says so
    assert term["description"] == "Id of a customer" and term["sensitive"] == 1 and term["origin"] == "generated"
    assert store.entity("customer")["origin"] == "generated" and store.entity("policy")["origin"] == "document"
    assert store.fill_if_empty() == {"terms": 0, "entities": 0, "label_conflicts": 0}  # only once
    col = db.query_one(columns_select(db) + " WHERE c.column_id = 'c2'")
    assert col["term_label"] == "Customer Identifier" and col["term_sensitive"] == 1


def test_term_crud_keeps_the_slug_and_republishes(crm):
    t = crm.update_business_term("customer-identifier", "Client Number", "The number a client is known by", True,
                                 actor="steward")
    assert t["slug"] == "customer-identifier" and t["label"] == "Client Number" and t["origin"] == "edited"
    assert t["sensitive"] is True and t["updated_by"] == "steward"
    node = term_uri("customer-identifier")
    assert labels(crm, node) == {"Client Number"}  # every context republished: no stale label left
    assert (node, SKOS.definition, None) in crm.graph.dataset
    view = {c["name"]: c for c in crm.get_dataset(dataset_id(crm, "customers"))["columns"]}
    assert view["customer_id"]["business_term"]["label"] == "Client Number"
    res = crm.retrieve_documents(["client number"], evidence_types=["glossary"])
    assert res["glossary_results"][0]["label"] == "Client Number"

    # term generation keeps the manual edit, even when forced
    crm.generate_terms([dataset_id(crm, "customers")], force=True, wait_seconds=60)
    assert crm.get_business_term("customer-identifier")["label"] == "Client Number"

    with pytest.raises(Conflict):
        crm.create_business_term("Client Number")  # same label
    with pytest.raises(Conflict):
        crm.update_business_term("order-identifier", "client number")  # label of another term
    with pytest.raises(InvalidArgument):
        crm.create_business_term("   ")


def test_manual_term_without_columns_is_in_the_graph_and_the_glossary(crm):
    t = crm.create_business_term("Lifetime Value", "Total margin expected from a customer", None)
    assert t["slug"] == "lifetime-value" and t["origin"] == "manual" and t["columns"] == []
    assert labels(crm, term_uri("lifetime-value")) == {"Lifetime Value"}
    listed = {x["slug"]: x for x in crm.list_business_terms()["terms"]}
    assert listed["lifetime-value"]["column_count"] == 0
    assert crm.list_business_terms("lifetime")["total"] == 1
    res = crm.retrieve_documents(["expected margin of a customer"], evidence_types=["glossary"])
    hit = next(g for g in res["glossary_results"] if g["label"] == "Lifetime Value")
    assert hit["source_ref"] is None and hit["column_id"] is None


def test_term_delete_unbinds_columns_and_generation_does_not_recreate_it(crm):
    ds = dataset_id(crm, "customers")
    out = crm.delete_business_term("customer-email-address")
    assert out["deleted"] is True and ds in out["dataset_ids"]
    with pytest.raises(NotFound):
        crm.get_business_term("customer-email-address")
    col = crm.db.query_one("SELECT term_slug, term_status FROM columns WHERE dataset_id = ? AND name = 'email'", (ds,))
    assert col == {"term_slug": None, "term_status": "removed"}
    assert not labels(crm, term_uri("customer-email-address"))
    crm.generate_terms([ds], force=True, wait_seconds=60)
    col = crm.db.query_one("SELECT term_slug, term_status FROM columns WHERE dataset_id = ? AND name = 'email'", (ds,))
    assert col == {"term_slug": None, "term_status": "removed"}
    with pytest.raises(NotFound):
        crm.get_business_term("customer-email-address")


def test_entity_edit_keeps_the_slug_everywhere(crm):
    doc = crm.ingest_document(content=POLICY.encode(), filename="policy.md", wait_seconds=30)
    e = crm.update_business_entity("customer", "Client", "A person or company that buys from us", actor="steward")
    assert e["slug"] == "customer" and e["label"] == "Client" and e["description"].startswith("A person")
    assert labels(crm, entity_uri("customer")) == {"Client"}
    assert crm.get_dataset(dataset_id(crm, "customers"))["business_entity"]["label"] == "Client"
    # re-indexing the policy finds "customer" (lexicon) and "client" (glossary label): both are the same entity
    crm.index_documents([doc["document_id"]], force=True, wait_seconds=60)
    mentions = crm.db.query("SELECT DISTINCT entity_slug, entity_label FROM doc_entity_mentions WHERE document_id = ?",
                            (doc["document_id"],))
    assert {"entity_slug": "customer", "entity_label": "Client"} in mentions
    assert not any(m["entity_slug"] == "client" for m in mentions)
    with pytest.raises(Conflict):
        crm.update_business_entity("order", "client")


def test_entity_delete_removes_it_everywhere_and_is_not_recreated(crm):
    doc = crm.ingest_document(content=POLICY.encode(), filename="policy.md", wait_seconds=30)
    ds = dataset_id(crm, "customers")
    assert crm.db.scalar("SELECT COUNT(*) FROM doc_entity_mentions WHERE entity_slug = 'customer'") > 0
    out = crm.delete_business_entity("customer", actor="steward")
    assert out["removed"]["datasets"] == 1 and out["removed"]["document_mentions"] > 0
    assert crm.get_dataset(ds)["business_entity"] is None
    assert crm.db.scalar("SELECT COUNT(*) FROM doc_entity_mentions WHERE entity_slug = 'customer'") == 0
    assert (entity_uri("customer"), None, None) not in crm.graph.dataset
    assert not any(e["slug"] == "customer" for e in crm.list_business_entities()["entities"])
    # neither term generation nor document indexing brings it back
    crm.generate_terms([ds], force=True, wait_seconds=60)
    assert crm.get_dataset(ds)["business_entity"] is None
    crm.index_documents([doc["document_id"]], force=True, wait_seconds=60)
    assert crm.db.scalar("SELECT COUNT(*) FROM doc_entity_mentions WHERE entity_slug = 'customer'") == 0
    with pytest.raises(NotFound):
        crm.get_business_entity("customer")


def test_term_rename_is_shown_in_its_definitions(crm):
    doc = crm.ingest_document(content=POLICY.encode(), filename="policy.md", wait_seconds=30)
    crm.create_definition("Active Customer", "Ordered in the past 90 days.", ["customer-identifier"],
                          doc["document_id"], "Orders")
    crm.update_business_term("customer-identifier", "Client Number")
    assert crm.definitions.get("active-customer")["terms"][0]["label"] == "Client Number"
    assert labels(crm, term_uri("customer-identifier")) == {"Client Number"}  # definition context republished
    assert (None, MESH.definedIn, None) in crm.graph.dataset


def test_glossary_api(crm):
    with TestClient(create_app(crm)) as client:
        created = client.post("/api/business-terms", json={"label": "Churn Risk", "description": "Likelihood to leave",
                                                           "sensitive": False}, headers={"X-User": "ann"})
        assert created.status_code == 201 and created.json()["updated_by"] == "ann"
        assert client.post("/api/business-terms", json={"label": "Churn Risk"}).status_code == 409
        terms = client.get("/api/business-terms", params={"q": "churn"}).json()
        assert [t["slug"] for t in terms["terms"]] == ["churn-risk"] and terms["terms"][0]["sensitive"] is False
        edited = client.put("/api/business-terms/churn-risk", json={"label": "Churn Probability"}).json()
        assert edited["slug"] == "churn-risk" and edited["label"] == "Churn Probability"
        assert client.delete("/api/business-terms/churn-risk").status_code == 200
        assert client.get("/api/business-terms/churn-risk").status_code == 404

        entities = client.get("/api/business-entities").json()["entities"]
        order = next(e for e in entities if e["slug"] == "order")
        assert order["dataset_count"] == 1
        put = client.put("/api/business-entities/order", json={"label": "Sales Order", "description": "A purchase"})
        assert put.status_code == 200 and put.json()["description"] == "A purchase"
        assert client.delete("/api/business-entities/order").json()["removed"]["datasets"] == 1
        assert client.get("/api/business-entities/order").status_code == 404

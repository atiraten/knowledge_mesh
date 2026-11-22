import asyncio
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.api.app import create_app
from knowledge_mesh.mcp_server import build_server
from tests.test_crawler import resolver, site_handler
from tests.test_loaders_chunking import make_pdf

SPEC_TOOLS = {"crawl_website", "ingest_document", "index_documents", "read_document",
              "extract_structured_metadata", "discover_keys", "retrieve_documents"}


@pytest.fixture
def client(mesh_factory, config_factory, sqlite_source, fake_term_llm):
    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source]}), term_llm=fake_term_llm,
                        http_transport=httpx.MockTransport(site_handler), resolver=resolver)
    with TestClient(create_app(mesh)) as c:
        c.mesh = mesh
        yield c


def wait_job(client, job_id):
    client.mesh.runner.wait(job_id, timeout=120)
    return client.get(f"/api/jobs/{job_id}").json()


def test_health_and_trace_header(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert len(r.headers["x-trace-id"]) == 32
    assert r.json()["checks"]["document_index"]["embedding_model"]


def test_browser_ui_served_with_csp(client):
    r = client.get("/")
    assert r.status_code == 200 and "Knowledge Mesh" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200


def test_upload_search_and_follow_citations(client):
    pdf = make_pdf(["Customers are identified by a customer number.", "Each order belongs to one customer."])
    r = client.post("/api/documents", files={"file": ("guide.pdf", pdf, "application/pdf")})
    assert r.status_code == 201, r.text
    doc_id = r.json()["document_id"]
    assert wait_job(client, r.json()["job_id"])["status"] == "succeeded"
    assert client.get(f"/api/documents/{doc_id}").json()["status"] == "indexed"
    assert any(d["document_id"] == doc_id for d in client.get("/api/documents").json()["documents"])
    s = client.post("/api/search", json={"query": "which customer does an order belong to"}).json()
    hit = s["results"][0]
    assert hit["page"] == 2 and s["trace_id"]
    chunk = client.get(hit["citation"]["chunk_ref"]).json()
    assert chunk["document_id"] == doc_id and chunk["page"] == 2
    original = client.get(hit["citation"]["original_ref"])
    assert original.status_code == 200 and original.headers["content-type"] == "application/pdf"
    assert original.headers["content-security-policy"] == "sandbox"
    text = client.get(f"/api/documents/{doc_id}/text", params={"page": 2}).json()
    assert "order" in text["text"].lower()


def test_upload_rejections_use_structured_errors(client):
    r = client.post("/api/documents", files={"file": ("run.sh", b"#!/bin/sh", "text/x-sh")})
    assert r.status_code == 415
    body = r.json()
    assert body["error"]["code"] == "unsupported_media_type" and body["trace_id"]
    r = client.post("/api/documents", files={"file": ("x.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "extraction_failed"
    assert r.json()["error"]["details"]["document_id"].startswith("doc-")
    assert client.get("/api/documents/doc-missing").status_code == 404
    bad = client.post("/api/search", json={"query": "x", "top_k": 99})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_argument"
    assert client.post("/api/search", json={"nope": 1}).status_code == 422
    assert client.post("/api/chat", json={"query": "anything"}).status_code == 503


def test_crawl_submission_and_blocked_url(client):
    r = client.post("/api/crawl", json={"url": "https://site.test/", "max_depth": 2, "max_pages": 8})
    assert r.status_code == 202
    job = wait_job(client, r.json()["job_id"])
    assert job["status"] in {"succeeded", "partial"} and job["item_count"] >= 8
    page = client.get(f"/api/jobs/{job['job_id']}/items", params={"limit": 3}).json()
    assert len(page["items"]) == 3 and page["next_offset"] == 3
    blocked = client.post("/api/crawl", json={"url": "http://169.254.169.254/latest/meta-data/"})
    assert blocked.status_code == 400 and blocked.json()["error"]["code"] == "blocked_url"
    assert client.post("/api/crawl", json={"url": "https://site.test/", "max_depth": 0}).status_code == 422


def test_structured_catalog_definitions_and_mixed_search(client):
    r = client.get("/api/structured-sources").json()
    assert r["sources"][0]["source_id"] == "crm"
    assert "TEST_CRM_URL" in json.dumps(r) and "sqlite:///" not in json.dumps(r)  # env var name only
    job = client.post("/api/structured-sources/crm/catalog", json={}).json()
    assert wait_job(client, job["job_id"])["status"] == "succeeded"
    tree = client.get("/api/catalog").json()["sources"][0]
    ds = next(d for s in tree["schemas"] for d in s["datasets"] if d["name"] == "customers")
    detail = client.get(f"/api/datasets/{ds['dataset_id']}").json()
    term = detail["columns"][0]["business_term"]
    assert client.get(term["ref"]).status_code == 200
    doc = client.mesh.ingest_document(content=b"# Customer policy\nAn active customer ordered in 90 days.",
                                      filename="policy.md", wait_seconds=30)
    body = {"name": "Active Customer", "definition": "Ordered in 90 days.", "terms": [term["slug"]],
            "document_id": doc["document_id"], "location": "section 1"}
    d = client.post("/api/business-definitions", json=body, headers={"X-User": "alice"})
    assert d.status_code == 201 and d.json()["author"] == "alice"
    got = client.get("/api/business-definitions/active-customer").json()
    assert got["name"] == "Active Customer" and [t["slug"] for t in got["terms"]] == [term["slug"]]
    assert got["document"]["document_id"] == doc["document_id"] and got["location"] == "section 1"
    # terms only: entities are not accepted; a definition needs an existing term and its policy document
    assert client.post("/api/business-definitions", json={**body, "name": "Orphan", "entities": ["Customer"]}
                       ).status_code == 422
    unknown = client.post("/api/business-definitions", json={**body, "name": "Orphan", "terms": ["nope"]})
    assert unknown.status_code == 400 and unknown.json()["error"]["details"]["unknown_terms"] == ["nope"]
    unknown_doc = client.post("/api/business-definitions", json={**body, "name": "Orphan", "document_id": "doc-missing"})
    assert unknown_doc.status_code == 400
    # the policy document is optional (a location needs one)
    no_doc = client.post("/api/business-definitions", json={"name": "Prospect", "definition": "Never ordered.",
                                                            "terms": [term["slug"]]})
    assert no_doc.status_code == 201 and no_doc.json()["document"] is None and no_doc.json()["location"] is None
    assert client.post("/api/business-definitions", json={"name": "Lead", "definition": "x", "terms": [term["slug"]],
                                                          "location": "page 2"}).status_code == 400
    assert client.post("/api/business-definitions", json=body).status_code == 409  # same name again
    edited = client.put("/api/business-definitions/active-customer", json={**body, "name": "Active Buyer"}).json()
    assert edited["definition_id"] == "active-customer" and edited["name"] == "Active Buyer"
    blocked = client.delete(term["ref"])
    assert blocked.status_code == 409 and blocked.json()["error"]["details"]["definitions"][0]["definition_id"] \
        == "active-customer"
    client.put("/api/business-definitions/active-customer", json=body)
    s = client.post("/api/search", json={"query": "active customer identifier"}).json()
    assert s["glossary_results"] and any(g["type"] == "business_definition" for g in s["graph_evidence"])


def test_api_key_enforced_when_configured(mesh_factory, config_factory, monkeypatch):
    monkeypatch.setenv("TEST_MESH_KEY", "s3cret")
    mesh = mesh_factory(config_factory({"api": {"api_key_env": "TEST_MESH_KEY"}}))
    with TestClient(create_app(mesh)) as c:
        assert c.get("/api/documents").status_code == 401
        assert c.get("/api/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/api/health").status_code == 200


def test_metrics_endpoint_exposes_application_metrics(client):
    client.post("/api/documents", files={"file": ("n.txt", b"Customer orders note.", "text/plain")})
    client.get("/api/documents/doc-missing")
    text = client.get("/metrics").text
    assert "mesh_http_requests" in text and "mesh_documents_ingested" in text


def test_mcp_tools_registered_and_bounded(client):
    server = build_server(client.mesh)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert SPEC_TOOLS <= set(tools)

    def call(name, args):
        result = asyncio.run(server.call_tool(name, args))
        return json.loads(result.content[0].text)

    content = base64.b64encode(b"# Accounts\nAn account holds payments for a customer.").decode()
    out = call("ingest_document", {"content_base64": content, "filename": "acct.md", "wait_seconds": 30})
    assert out["ok"] and out["document_id"].startswith("doc-") and out["job_id"] and out["trace_id"]
    read = call("read_document", {"document_id": out["document_id"], "max_chars": 200})
    assert read["ok"] and len(read["text"]) <= 200 and read["document_ref"]
    idx = call("index_documents", {"document_ids": [out["document_id"]], "force": True, "wait_seconds": 30})
    assert idx["ok"] and idx["index_version"].startswith("documents-v") and "counts" in idx
    crawl = call("crawl_website", {"url": "https://site.test/", "max_depth": 3, "max_pages": 30, "wait_seconds": 60})
    assert crawl["ok"] and crawl["manifest_ref"].startswith("/api/jobs/") and len(json.dumps(crawl)) < 5000
    ext = call("extract_structured_metadata", {"source_id": "crm", "wait_seconds": 60})
    assert ext["ok"] and ext["result"]["extract"]["catalog_refs"]
    ids = [r.rsplit("/", 1)[-1] for r in ext["result"]["extract"]["catalog_refs"]]
    keys = call("discover_keys", {"dataset_ids": ids, "wait_seconds": 60})
    assert keys["ok"] and keys["result"]["profiling_ref"] and keys["result"]["inferred_unique_keys"] > 0
    assert "most_common_values" not in json.dumps(keys)  # raw profiling values stay out of tool results
    res = call("retrieve_documents", {"queries": ["account payments customer"], "top_k": 3})
    assert res["ok"] and len(res["results"]) <= 3
    err = call("read_document", {"document_id": "doc-nope"})
    assert err["ok"] is False and err["error"]["code"] == "not_found"
    blocked = call("crawl_website", {"url": "http://127.0.0.1:8080/"})
    assert blocked["error"]["code"] == "blocked_url"
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    traced = call("retrieve_documents", {"queries": ["account"], "traceparent": tp})
    assert traced["trace_id"] == "0af7651916cd43dd8448eb211c80319c"

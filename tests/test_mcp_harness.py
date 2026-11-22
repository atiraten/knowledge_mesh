"""The search agent harness uses MCP tools only (config ``mcp``): tool list from list_tools (read-only allowlist,
parameters the harness fills in hidden), search scope added to every call, evidence markers numbered in the
ledger, generate_sql from the evidence passed only, and both connection modes (in process; streamable HTTP at /mcp
in the same app, with the API key)."""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn

from knowledge_mesh.agent import EvidenceLedger, McpToolBox, SearchScope
from knowledge_mesh.api.app import create_app
from knowledge_mesh.common.ids import dataset_id_for
from knowledge_mesh.config import MCP_AGENT_TOOLS, ConfigError, load_config
from knowledge_mesh.llm.client import FakeLLM
from knowledge_mesh.mcp_server import build_server
from tests.conftest import REPO
from tests.test_agent import CLV_DOC, PROMPTS, SINGLE, Script, call, complete, tools

WRITE_TOOLS = {"crawl_website", "ingest_document", "index_documents", "extract_structured_metadata",
               "discover_keys", "get_job"}


def agent_settings():
    return {"enabled": True, "provider": "fake", "model": "fake", **PROMPTS,
            "sql_generation_prompt_path": str(REPO / "prompts" / "sql_generation.txt")}


@pytest.fixture
def crm_mesh(mesh_factory, config_factory, sqlite_source, fake_term_llm):
    def make(llm=None, **overrides):
        mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source], "agent": agent_settings(),
                                            **overrides}),
                            term_llm=fake_term_llm, agent_llm=llm or FakeLLM(lambda *a: "{}"))
        mesh.ingest_document(content=CLV_DOC.encode(), filename="clv.md", wait_seconds=30)
        mesh.catalog_source("crm", wait_seconds=120)
        return mesh
    return make


def toolbox(mesh, scope=None):
    return McpToolBox(mesh.mcp_tools, mesh.config, EvidenceLedger(), scope)


def test_config_mcp_section_is_validated():
    cfg = load_config(REPO / "config" / "mesh.yaml")
    assert cfg.mcp.mode == "in_process" and cfg.mcp.url == "http://127.0.0.1:9801/mcp"
    assert cfg.mcp.tools == MCP_AGENT_TOOLS and not WRITE_TOOLS & set(cfg.mcp.tools)


@pytest.mark.parametrize("mcp, message", [({"mode": "stdio"}, "mode"), ({"url": "127.0.0.1:9801/mcp"}, "http"),
                                          ({"tools": []}, "tools"), ({"port": 1}, "port")])
def test_bad_mcp_settings_are_rejected(config_factory, mcp, message):
    with pytest.raises((ConfigError, ValueError), match=message):
        config_factory({"mcp": mcp})


def test_executor_gets_the_allowlisted_mcp_tools_without_hidden_parameters(crm_mesh):
    mesh = crm_mesh()
    specs = {s["name"]: s for s in toolbox(mesh).specs()}
    published = {t.name for t in asyncio.run(build_server(mesh).list_tools())}
    assert set(specs) == set(MCP_AGENT_TOOLS) and WRITE_TOOLS <= published - set(specs)
    params = {name: set(s["parameters"]["properties"]) for name, s in specs.items()}
    assert params["retrieve_documents"] == {"queries", "top_k", "evidence_types", "document_ids", "dataset_ids"}
    assert params["execute_sql"] == {"source_id", "sql", "max_rows"}
    assert params["generate_sql"] == {"source_id", "intent", "tables", "joins", "facts"}
    assert not any(p & {"traceparent", "caller", "view", "structured_source_ids"} for p in params.values())
    assert "max_chars" not in params["read_document"] and "document_ids" not in params["query_knowledge_graph"]

    mesh.config.mcp.tools = ["retrieve_documents", "crawl_website"]  # the allowlist decides
    assert {s["name"] for s in toolbox(mesh).specs()} == {"retrieve_documents", "crawl_website"}
    mesh.config.sql_execution.enabled = False
    mesh.config.mcp.tools = list(MCP_AGENT_TOOLS)
    assert not {"generate_sql", "execute_sql"} & {s["name"] for s in toolbox(mesh).specs()}


def test_search_scope_is_added_to_every_call_and_only_narrowed(crm_mesh, monkeypatch):
    mesh = crm_mesh()
    seen, real = [], mesh.retrieve_documents

    def spy(queries, top_k=None, evidence_types=None, document_ids=None, source_types=None,
            structured_source_ids=None, dataset_ids=None, expand_parents=None, rerank=None):
        seen.append({"top_k": top_k, "evidence_types": evidence_types, "source_types": source_types,
                     "structured_source_ids": structured_source_ids, "expand_parents": expand_parents})
        return real(queries, top_k, evidence_types, document_ids, source_types, structured_source_ids, dataset_ids,
                    expand_parents, rerank)
    monkeypatch.setattr(mesh, "retrieve_documents", spy)
    scope = SearchScope(top_k=3, evidence_types=["documents", "glossary"], source_types=["upload"],
                        structured_source_ids=["crm"], expand_parents=False, caller="local:test")
    box = toolbox(mesh, scope)
    ok = box.call("retrieve_documents", {"queries": ["customer lifetime value"], "top_k": 10,
                                         "evidence_types": ["documents", "graph"], "source_types": ["web"]})
    assert ok.status == "ok" and seen[-1] == {"top_k": 3, "evidence_types": ["documents"], "source_types": ["upload"],
                                              "structured_source_ids": ["crm"], "expand_parents": False}
    out = box.call("retrieve_documents", {"queries": ["x"], "evidence_types": ["graph"]})
    assert out.status == "error" and out.observation["error"]["code"] == "invalid_argument" and len(seen) == 1

    rows = box.call("execute_sql", {"source_id": "crm", "sql": "SELECT count(*) FROM customers", "caller": "spoofed"})
    assert rows.status == "ok" and rows.observation["rows"] == [[12]]
    audit = mesh.db.query("SELECT caller FROM audit_events WHERE action = 'execute_sql'")
    assert audit[-1]["caller"] == "local:test"  # the model cannot set hidden parameters


def test_evidence_items_are_numbered_once_and_details_replace_them(crm_mesh):
    mesh = crm_mesh()
    box = toolbox(mesh)
    first = box.call("retrieve_documents", {"queries": ["customer lifetime value"]})
    again = box.call("retrieve_documents", {"queries": ["Customer Lifetime Value"]})
    assert first.evidence_ids and first.evidence_ids[0] == again.evidence_ids[0] == "D1"
    doc = first.observation["documents"][0]
    assert doc["id"] == "D1" and not {"evidence_ref", "evidence_kind", "evidence_data", "parents"} & set(doc)
    assert box.ledger.item("D1").data["citation"]["chunk_ref"].startswith("/api/chunks/")  # full data for synthesis

    listed = box.call("find_datasets", {"name_contains": "orders"})
    tid = listed.observation["datasets"][0]["id"]
    assert tid.startswith("T") and "columns" not in box.ledger.item(tid).data
    orders = dataset_id_for("crm", "main", "orders")
    detail = box.call("get_dataset", {"dataset_id": orders})
    assert detail.evidence_ids == [tid] and detail.observation["id"] == tid
    assert {c["name"] for c in box.ledger.item(tid).data["columns"]} >= {"order_id", "amount"}
    assert "most_common_values" not in str(detail.observation)

    seen = box.call("retrieve_documents", {"queries": ["customer lifetime value"]}, frozenset(first.evidence_ids))
    stub = seen.observation["documents"][0]
    assert seen.compacted and stub["seen_before"] is True and "text" not in stub


def test_generate_sql_writes_the_query_from_the_evidence_passed_only(crm_mesh):
    llm = FakeLLM(lambda prompt, system, purpose: '{"sql": "SELECT customer_id, sum(amount) FROM main.orders '
                                                  'GROUP BY customer_id", "explanation": "orders per customer"}')
    mesh = crm_mesh(llm)
    server = build_server(mesh)

    def call_mcp(args):
        import json
        return json.loads(asyncio.run(server.call_tool("generate_sql", args)).content[0].text)

    out = call_mcp({"source_id": "crm", "intent": "total order amount per customer",
                    "tables": [{"name": "main.orders", "columns": [{"name": "customer_id", "data_type": "TEXT"},
                                                                   {"name": "amount", "data_type": "REAL",
                                                                    "business_term": "Order Amount"}],
                                "keys": [{"type": "primary", "columns": ["order_id"], "origin": "declared"}]}],
                    "facts": ["Amounts are in euros."]})
    assert out["ok"] and out["status"] == "ok" and out["tables"] == ["catalog:crm/main/orders"]
    prompt = next(c for c in llm.calls if c["purpose"] == "sql_generation")["prompt"]
    assert "table main.orders: columns: customer_id TEXT; amount REAL (term: Order Amount)" in prompt
    assert "Amounts are in euros." in prompt and "SQL dialect: sqlite" in prompt
    assert "customers" not in prompt and "stores" not in prompt  # nothing looked up in the catalog

    missing = call_mcp({"source_id": "crm", "intent": "x", "tables": []})
    assert missing["ok"] is False and missing["error"]["code"] == "invalid_argument"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served():
    """Run a FastAPI app on a free port (uvicorn in a thread); yields the base URL."""
    servers = []

    def start(app, port):
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 20
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"
    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(10)


def test_streamable_http_mode_mounts_mcp_in_the_app_for_the_harness_and_outside_clients(
        mesh_factory, config_factory, served, monkeypatch):
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    monkeypatch.setenv("TEST_MCP_KEY", "k3y")
    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    script = Script(SINGLE, {"I1": [tools(call("retrieve_documents", queries=["customer lifetime value"])),
                                    lambda p: complete("CLV is the total revenue of a customer [D1].")]})
    mesh = mesh_factory(config_factory({"agent": agent_settings(), "api": {"api_key_env": "TEST_MCP_KEY"},
                                        "mcp": {"mode": "streamable_http", "url": url}}), agent_llm=FakeLLM(script))
    mesh.ingest_document(content=CLV_DOC.encode(), filename="clv.md", wait_seconds=30)
    served(create_app(mesh), port)

    res = mesh.search("customer lifetime value")  # the harness connects to mcp.url with the API key
    assert res["mode"] == "agent" and res["outcome"] == "answered" and res["citations"][0]["id"] == "D1"

    async def outside(headers):
        transport = streamable_http_client(url, http_client=create_mcp_http_client(headers=headers))
        async with Client(transport) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            out = await client.call_tool("retrieve_documents", {"queries": ["customer lifetime value"]})
            return names, out
    names, out = asyncio.run(outside({"X-API-Key": "k3y"}))
    assert WRITE_TOOLS <= names and not out.is_error and "customer" in out.content[0].text.lower()
    with pytest.raises(Exception):
        asyncio.run(outside(None))  # /mcp requires the same key as /api


def test_unreachable_mcp_server_falls_back_to_direct_retrieval(mesh_factory, config_factory):
    url = f"http://127.0.0.1:{free_port()}/mcp"  # nothing listens there
    script = Script(SINGLE, {"I1": [complete("x")]})
    mesh = mesh_factory(config_factory({"agent": agent_settings(), "mcp": {"mode": "streamable_http", "url": url}}),
                        agent_llm=FakeLLM(script))
    mesh.ingest_document(content=CLV_DOC.encode(), filename="clv.md", wait_seconds=30)
    res = mesh.search("customer lifetime value")
    assert res["mode"] == "retrieval" and "MCP server unavailable" in res["agent"]["reason"] and res["results"]

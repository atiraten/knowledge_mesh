"""MCP publication of query_knowledge_graph, execute_sql and generate_html (AT-01); retrieve_documents
keeps its signature and behaviour (AT-14)."""

import asyncio
import json

from knowledge_mesh.mcp_server import build_server
from tests.kg_support import shop_mesh, shop_source  # noqa: F401

RETRIEVE_DOCUMENTS_PARAMS = {"queries", "top_k", "evidence_types", "document_ids", "source_types",
                           "structured_source_ids", "dataset_ids", "expand_parents", "rerank", "view", "traceparent"}


def test_tools_are_discoverable_with_documented_schemas(shop_mesh):
    server = build_server(shop_mesh(catalog=False))
    tools = {t.name: t.model_dump(exclude_none=True) for t in asyncio.run(server.list_tools())}
    assert {"query_knowledge_graph", "execute_sql", "generate_html", "retrieve_documents"} <= set(tools)
    kg = tools["query_knowledge_graph"]
    assert {"intent_id", "intent", "retrieval_queries", "max_hops", "top_k", "include_physical_mappings",
            "include_related_terms"} <= set(kg["input_schema"]["properties"])
    sql = tools["execute_sql"]
    assert set(sql["input_schema"]["required"]) == {"source_id", "sql"}
    html = tools["generate_html"]
    block = json.dumps(html["input_schema"])
    assert all(t in block for t in ("heading", "paragraph", "list", "table", "citations"))
    for name in ("query_knowledge_graph", "execute_sql", "generate_html"):
        t = tools[name]
        assert all(word in t["description"] for word in ("Input:", "Purpose:", "Output:")), name
        assert t["output_schema"]["properties"]["ok"]["type"] == "boolean"
    assert {"paths", "business_entities", "business_terms", "physical_mappings", "warnings"} <= set(
        tools["query_knowledge_graph"]["output_schema"]["properties"])
    assert {"status", "columns", "rows", "row_count"} <= set(tools["execute_sql"]["output_schema"]["properties"])
    assert set(tools["retrieve_documents"]["input_schema"]["properties"]) == RETRIEVE_DOCUMENTS_PARAMS


def test_tools_run_through_mcp(shop_mesh):
    mesh = shop_mesh()
    server = build_server(mesh)

    def call(name, args):
        result = asyncio.run(server.call_tool(name, args))
        return json.loads(result.content[0].text)

    kg = call("query_knowledge_graph", {"intent_id": "I1", "intent": "customer bought product",
                                        "retrieval_queries": ["customer bought product"], "top_k": 2})
    assert kg["ok"] and kg["status"] == "ok" and kg["intent_id"] == "I1" and kg["trace_id"]
    assert [e["predicate"] for e in kg["paths"][0]["edges"]] == ["placesOrder", "containsOrderLine",
                                                                 "referencesProduct"]
    assert call("query_knowledge_graph", {"retrieval_queries": ["zebra"]})["status"] == "no_match"
    bad = call("query_knowledge_graph", {"max_hops": 3})
    assert bad["ok"] is False and bad["error"]["code"] == "invalid_argument"
    rows = call("execute_sql", {"source_id": "shop", "sql": "SELECT product_name FROM products ORDER BY 1"})
    assert rows["ok"] and rows["status"] == "ok" and rows["rows"] == [["Cable"], ["Laptop"], ["Phone"]]
    denied = call("execute_sql", {"source_id": "shop", "sql": "DROP TABLE products"})
    assert denied["status"] == "error" and denied["error"]["code"] == "not_read_only"
    html = call("generate_html", {"title": "Products", "blocks": [
        {"type": "table", "columns": [c["name"] for c in rows["columns"]], "rows": rows["rows"]}]})
    assert html["ok"] and "<td>Laptop</td>" in html["html"]
    assert call("generate_html", {"blocks": []})["ok"] is False


def test_retrieve_documents_behaves_as_before(shop_mesh):
    """AT-14: same tool, same result fields (queries is a list); it now also lists the proposed relationships."""
    mesh = shop_mesh()
    server = build_server(mesh)
    result = asyncio.run(server.call_tool("retrieve_documents", {"queries": ["customer birth date"], "top_k": 3}))
    out = json.loads(result.content[0].text)
    assert out["ok"] and {"results", "glossary_results", "graph_evidence", "insufficient_evidence"} <= set(out)
    assert any(g["label"] == "Birth Date" for g in out["glossary_results"])
    direct = mesh.retrieve_documents(["customer birth date"], top_k=3)
    assert [g["graph_ref"] for g in direct["glossary_results"]] == [g["graph_ref"] for g in out["glossary_results"]]

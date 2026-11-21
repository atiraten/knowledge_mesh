"""Catalog navigation (AT-19) and the knowledge graph view of a database, schema or table (AT-20), API side;
the catalog grids (database -> schemas, schema -> tables + business entity, table -> columns) and the data they
show. The browser interaction itself (clicks and right-click in the tree) is checked live, not here."""

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.api.app import STATIC_DIR, create_app
from knowledge_mesh.common.html_render import HtmlRequest, render_html
from knowledge_mesh.storage.db import Database
from tests.kg_support import SHOP_TABLES, shop_ids, shop_mesh, shop_source  # noqa: F401


@pytest.fixture
def client(shop_mesh):
    return TestClient(create_app(shop_mesh()))


def test_catalog_tree_nests_tables_under_schemas_under_databases(client):
    """AT-19 data: Databases (sources) -> Schemas -> Tables."""
    sources = client.get("/api/catalog").json()["sources"]
    shop = next(s for s in sources if s["source_id"] == "shop")
    assert [sc["schema"] for sc in shop["schemas"]] == ["main"]
    assert sorted(d["name"] for d in shop["schemas"][0]["datasets"]) == sorted(SHOP_TABLES)
    js = (STATIC_DIR / "app.js").read_text()
    for marker in ("catalogFacet", 'role: "tree"', '"contextmenu"', "F10",
                   "Knowledge graph of", "renderKnowledgeGraph", "/api/graph/view"):
        assert marker in js, marker
    assert client.get("/static/vendor/cytoscape.min.js").status_code == 200
    assert (STATIC_DIR / "vendor" / "cytoscape.LICENSE").read_text().startswith("Copyright")


def test_graph_view_is_scoped_to_the_selection(client):
    """AT-20 data: right-click database / schema / table -> nodes and edges for that selection."""
    whole = client.get("/api/graph/view", params={"source_id": "shop"}).json()
    types = {n["type"] for n in whole["nodes"]}
    assert {"table", "column", "primary_key", "foreign_key", "entity", "term"} <= types
    edge_types = {e["type"] for e in whole["edges"]}
    assert {"has_column", "has_key", "references", "represents", "has_term", "semantic_relationship",
            "term_relation"} <= edge_types
    assert len([n for n in whole["nodes"] if n["type"] == "table"]) == len(SHOP_TABLES)
    schema = client.get("/api/graph/view", params={"source_id": "shop", "schema": "main"}).json()
    assert {n["id"] for n in schema["nodes"]} == {n["id"] for n in whole["nodes"]}
    ids = shop_ids()
    table = client.get("/api/graph/view", params={"source_id": "shop", "dataset_id": ids["orders"]}).json()
    tables = {n["label"] for n in table["nodes"] if n["type"] == "table"}
    assert tables == {"orders", "customers"}  # the table and the table its foreign key references
    columns = {n["label"] for n in table["nodes"] if n["type"] == "column"}
    assert columns == {"order_id", "customer_id", "order_date", "amount"}
    assert any(e["type"] == "references" and "customer_id→customer_id" in e["label"] for e in table["edges"])
    assert client.get("/api/graph/view", params={"source_id": "shop", "dataset_id": "ds-missing"}).status_code == 404
    assert client.get("/api/graph/view", params={"source_id": "nope"}).status_code == 404


def test_graph_view_respects_the_node_budget(shop_mesh):
    client = TestClient(create_app(shop_mesh({"knowledge_graph": {"view_max_nodes": 12}})))
    out = client.get("/api/graph/view", params={"source_id": "shop"}).json()
    assert len(out["nodes"]) == 12 and out["truncated"] is True


def test_catalog_grids_have_their_data(client):
    """Database grid: schemas + description; schema grid: tables + linked business entity; table grid: column name,
    data type (generic, from profiling), column data type (the source's type), content type, data format, term."""
    shop = next(s for s in client.get("/api/catalog").json()["sources"] if s["source_id"] == "shop")
    assert "description" in shop
    tables = {d["name"]: d for d in shop["schemas"][0]["datasets"]}
    assert tables["customers"]["entity_label"] and tables["customers"]["entity_slug"]
    assert client.get(f"/api/business-entities/{tables['customers']['entity_slug']}").status_code == 200
    d = client.get(f"/api/datasets/{shop_ids()['orders']}").json()
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["order_id"]["data_type"].upper() == "INTEGER" and cols["order_id"]["profile"]["type_category"] == "integer"
    assert cols["amount"]["profile"]["type_category"] in ("float", "decimal")
    assert cols["order_date"]["data_type"].upper() == "TEXT" and cols["order_date"]["profile"]["type_category"] == "string"
    assert cols["order_date"]["profile"]["content_type"] and cols["order_date"]["profile"]["format_patterns"]
    assert cols["customer_id"]["business_term"]["label"]
    js = (STATIC_DIR / "app.js").read_text()
    for marker in ('["Schema", "Description"]', '["Table", "Business entity"]',
                   '["Column name", "Data type", "Column data type", "Content type", "Data format", "Business term"]',
                   "catalogShell", "CATALOG_ROUTES", "p.type_category", '"km-grid"'):
        assert marker in js, marker
    css = (STATIC_DIR / "style.css").read_text()
    assert ".facet { position: sticky" in css and "table.km-grid" in css and "table.grid" in css


def test_profiles_saved_before_the_generic_type_existed_are_migrated(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE column_profiles (run_id TEXT NOT NULL, column_id TEXT NOT NULL, dataset_id TEXT NOT NULL,"
                 " row_count INTEGER, distinct_count INTEGER, empty_count INTEGER, null_count INTEGER,"
                 " format_patterns_json TEXT, content_type TEXT, most_common_values_json TEXT, created_at TEXT,"
                 " PRIMARY KEY (run_id, column_id))")
    conn.execute("INSERT INTO column_profiles(run_id, column_id, dataset_id, content_type) VALUES ('r1', 'c1', 'd1', 'text')")
    conn.commit()
    conn.close()
    db = Database(path)
    row = db.query_one("SELECT * FROM column_profiles WHERE column_id = 'c1'")
    assert row["content_type"] == "text" and row["type_category"] is None  # existing rows are kept as they were
    Database(path)  # idempotent


def test_sql_result_html_is_a_grid():
    out = render_html(HtmlRequest(blocks=[{"type": "table", "columns": ["state", "customer_count"],
                                           "rows": [["CA", 1], ["NY", 13]]}]))
    assert '<table class="km-grid"><thead><tr><th>state</th><th>customer_count</th></tr></thead>' in out["html"]


@pytest.mark.skipif(shutil.which("osascript") is None, reason="JavaScript syntax check uses macOS JavaScriptCore")
def test_app_js_parses():
    src = Path(STATIC_DIR / "app.js")
    script = (f'const s = $.NSString.stringWithContentsOfFileEncodingError("{src}", $.NSUTF8StringEncoding, null).js;'
              ' new Function(s); "ok"')
    out = subprocess.run(["osascript", "-l", "JavaScript", "-e", script], capture_output=True, text=True, timeout=30)
    assert out.stdout.strip() == "ok", out.stderr

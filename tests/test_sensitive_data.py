"""Sensitive data: business terms flagged by the term model when they can identify a person (PII / PHI /
GDPR), the per-column ``sensitive`` flag of execute_sql results (lineage through aliases, expressions, CTEs,
subqueries and UNION; COUNT excepted) and the red marking of those columns in generate_html grids."""

import asyncio
import json
import sqlite3

import pytest

from knowledge_mesh.api.app import STATIC_DIR
from knowledge_mesh.common.html_render import render_html
from knowledge_mesh.mcp_server import build_server
from knowledge_mesh.storage.db import Database
from knowledge_mesh.structured import sql_exec
from knowledge_mesh.structured.terms import parse_term_output
from tests.conftest import REPO
from tests.kg_support import shop_ids, shop_mesh, shop_source  # noqa: F401


def test_term_output_carries_the_sensitive_flag():
    text = "\n".join([
        "<table name>,<column name>,<business term>,<sensitive>,<detailed business description>,<comments>",
        "Customer,fstnm,Customer First Name,Y,The given name of the customer, as on file,cust=customer fst=first nm=name",
        "Customer,cust_id,Customer Identifier,N,The unique identifier of a customer,cust=customer id=identifier",
        "Customer,seg,Customer Segment Code,The marketing segment of the customer,seg=segment",  # older prompt: no flag
    ])
    records, issues = parse_term_output(text, "Customer", ["fstnm", "cust_id", "seg"])
    assert not issues
    assert records["fstnm"].sensitive is True and records["fstnm"].term == "Customer First Name"
    assert records["fstnm"].description == "The given name of the customer, as on file"
    assert records["fstnm"].explanation == "cust=customer fst=first nm=name"
    assert records["cust_id"].sensitive is False and records["cust_id"].description.startswith("The unique")
    assert records["seg"].sensitive is None and records["seg"].description == "The marketing segment of the customer"


def test_prompt_asks_for_the_flag():
    prompt = (REPO / "prompts" / "term_generation.txt").read_text()
    assert "<table name>,<column name>,<business term>,<sensitive>,<detailed business description>,<comments>" in prompt
    for item in ("identify a physical person", "Date Of Birth", "Social Security Number", "Medical Record Number",
                 "Medicaid Number", "company (not a person) is N"):
        assert item in prompt, item


def test_generated_terms_store_the_flag(shop_mesh):
    mesh = shop_mesh()
    cols = {c["name"]: c for c in mesh.get_dataset(shop_ids()["customers"])["columns"]}
    assert cols["full_name"]["business_term"]["sensitive"] is True
    assert cols["birth_date"]["business_term"]["sensitive"] is True
    assert cols["customer_id"]["business_term"]["sensitive"] is False
    assert mesh.get_business_term("birth-date")["sensitive"] is True
    assert mesh.get_business_term("customer-identifier")["sensitive"] is False


def test_terms_saved_before_the_flag_existed_are_migrated(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE columns (column_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, name TEXT NOT NULL,"
                 " ordinal INTEGER, data_type TEXT, nullable INTEGER, catalog_ref TEXT NOT NULL,"
                 " active INTEGER NOT NULL DEFAULT 1, term_label TEXT, term_slug TEXT, term_description TEXT,"
                 " term_explanation TEXT, term_run_id TEXT, term_prompt_version TEXT, term_status TEXT,"
                 " term_error TEXT, first_seen_at TEXT, last_seen_at TEXT, retired_at TEXT)")
    conn.execute("INSERT INTO columns(column_id, dataset_id, name, catalog_ref, term_label, term_status) "
                 "VALUES ('c1', 'd1', 'dob', 'x', 'Customer Date Of Birth', 'generated')")
    conn.commit()
    conn.close()
    row = Database(path).query_one("SELECT * FROM columns WHERE column_id = 'c1'")
    assert row["term_label"] == "Customer Date Of Birth" and row["term_sensitive"] is None  # not assessed


def _flags(out):
    assert out["status"] == "ok", out
    return [(c["name"], c["sensitive"], c["sensitive_terms"]) for c in out["columns"]]


def test_execute_sql_marks_columns_computed_from_sensitive_terms(shop_mesh):
    mesh = shop_mesh()
    out = mesh.execute_sql("shop", "SELECT customer_id, full_name AS who, UPPER(full_name) AS up, "
                                   "MAX(birth_date) AS latest, COUNT(birth_date) AS n, "
                                   "COUNT(DISTINCT full_name) AS names FROM customers")
    assert _flags(out) == [("customer_id", False, []), ("who", True, ["Customer Full Name"]),
                           ("up", True, ["Customer Full Name"]), ("latest", True, ["Birth Date"]),
                           ("n", False, []), ("names", False, [])]
    assert [f[1] for f in _flags(mesh.execute_sql("shop", "SELECT * FROM customers"))] == [False, True, True]
    cte = mesh.execute_sql("shop", "WITH x AS (SELECT full_name AS person FROM customers) SELECT person FROM x")
    assert _flags(cte) == [("person", True, ["Customer Full Name"])]
    derived = mesh.execute_sql("shop", "SELECT t.d FROM (SELECT birth_date AS d FROM main.customers) t")
    assert _flags(derived) == [("d", True, ["Birth Date"])]
    sub = mesh.execute_sql("shop", "SELECT o.order_id, (SELECT c.full_name FROM customers c "
                                   "WHERE c.customer_id = o.customer_id) AS buyer FROM orders o")
    assert [f[1] for f in _flags(sub)] == [False, True]
    counted = mesh.execute_sql("shop", "SELECT c.customer_id, (SELECT COUNT(*) FROM orders o "
                                       "WHERE o.customer_id = c.customer_id) AS orders FROM customers c")
    assert [f[1] for f in _flags(counted)] == [False, False]
    union = mesh.execute_sql("shop", "SELECT product_name FROM products UNION SELECT full_name FROM customers")
    assert _flags(union) == [("product_name", True, ["Customer Full Name"])]
    joined = mesh.execute_sql("shop", "SELECT c.full_name || ' ' || o.order_date AS label, SUM(o.amount) AS total "
                                      "FROM customers c JOIN orders o ON o.customer_id = c.customer_id GROUP BY 1")
    assert [f[1] for f in _flags(joined)] == [True, False]


def test_unassessed_terms_and_unreadable_lineage(shop_mesh, monkeypatch):
    mesh = shop_mesh()
    mesh.db.execute("UPDATE business_terms SET sensitive = NULL WHERE slug IN"
                    " (SELECT term_slug FROM columns WHERE name = 'amount')")
    assert _flags(mesh.execute_sql("shop", "SELECT amount, order_id FROM orders")) == [
        ("amount", None, []), ("order_id", False, [])]

    def broken(*args, **kwargs):
        raise RuntimeError("lineage unavailable")

    monkeypatch.setattr(sql_exec, "_lineage", broken)  # falls back to the output column names
    assert _flags(mesh.execute_sql("shop", "SELECT full_name, customer_id, UPPER(full_name) AS x FROM customers")) == [
        ("full_name", True, ["Customer Full Name"]), ("customer_id", False, []), ("x", None, [])]


def test_html_grid_marks_sensitive_columns_in_red():
    out = render_html({"blocks": [{"type": "table", "columns": ["customer_id", "full_name"],
                                   "rows": [["C1", "Ann Lee"]], "sensitive_columns": ["full_name"]}]})
    html = out["html"]
    assert '<thead><tr><th>customer_id</th><th class="km-sensitive">full_name</th></tr></thead>' in html
    assert '<tr><td>C1</td><td class="km-sensitive">Ann Lee</td></tr>' in html
    assert '<p class="km-note">Sensitive data (can identify a person) in red: full_name.</p>' in html
    plain = render_html({"blocks": [{"type": "table", "columns": ["n"], "rows": [[1]]}]})["html"]
    assert "km-sensitive" not in plain and "Sensitive data" not in plain
    with pytest.raises(ValueError, match="sensitive_columns not in the table"):
        render_html({"blocks": [{"type": "table", "columns": ["a"], "rows": [], "sensitive_columns": ["b"]}]})
    js = (STATIC_DIR / "app.js").read_text()
    css = (STATIC_DIR / "style.css").read_text()
    assert '"km-sensitive"' in js and "sensitiveTag(c.business_term.sensitive)" in js
    assert "table.km-grid .km-sensitive { color: var(--bad)" in css


def test_mcp_clients_get_the_flags(shop_mesh):
    server = build_server(shop_mesh())
    tools = {t.name: t.model_dump(exclude_none=True) for t in asyncio.run(server.list_tools())}
    assert "sensitive" in json.dumps(tools["execute_sql"]["output_schema"])
    assert "sensitive_columns" in json.dumps(tools["generate_html"]["input_schema"])

    def call(name, args):
        return json.loads(asyncio.run(server.call_tool(name, args)).content[0].text)

    rows = call("execute_sql", {"source_id": "shop", "sql": "SELECT customer_id, birth_date FROM customers "
                                                             "ORDER BY 1 LIMIT 1"})
    assert [(c["name"], c["sensitive"]) for c in rows["columns"]] == [("customer_id", False), ("birth_date", True)]
    html = call("generate_html", {"blocks": [{"type": "table", "columns": [c["name"] for c in rows["columns"]],
                                              "rows": rows["rows"],
                                              "sensitive_columns": [c["name"] for c in rows["columns"] if c["sensitive"]]}]})
    assert html["ok"] and '<td class="km-sensitive">' in html["html"]
    bad = call("generate_html", {"blocks": [{"type": "table", "columns": ["a"], "rows": [], "sensitive_columns": ["z"]}]})
    assert bad["ok"] is False and bad["error"]["code"] == "invalid_argument"

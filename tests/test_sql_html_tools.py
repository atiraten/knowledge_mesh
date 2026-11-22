"""execute_sql, generate_sql and generate_html, alone and through the agent (AT-21, AT-22)."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.api.app import create_app
from knowledge_mesh.common.errors import InvalidArgument
from knowledge_mesh.llm.client import FakeLLM
from tests.conftest import REPO
from tests.kg_support import shop_ids, shop_mesh, shop_source  # noqa: F401
from tests.test_agent import PROMPTS, Script, call, complete, last_result, tools

BORN = "SELECT customer_id, full_name, birth_date FROM customers WHERE birth_date < '1990-01-01' ORDER BY customer_id"


class ShopScript(Script):
    """Script that also answers the SQL-generation prompt (the agent model writes the SQL)."""

    def __init__(self, plan, actions, sql=None, **kw):
        super().__init__(plan, actions, **kw)
        self.sql = sql

    def __call__(self, prompt, system, purpose):
        if purpose == "sql_generation":
            return self._dump(self.sql, prompt)
        return super().__call__(prompt, system, purpose)


def plan(intent, queries):
    return {"original_query": intent, "decompose": False, "reasoning": "One objective.", "execution": "single",
            "intents": [{"id": "I1", "intent": intent, "retrieval_queries": queries, "depends_on": []}]}


def cite_synthesis(prompt):
    ids = re.findall(r"^\[(R\d+)\]", prompt, re.M) or re.findall(r"^\[([A-Z]\d+)\]", prompt, re.M)
    return f"Answer based on the records [{ids[0]}]." if ids else "No evidence."


@pytest.fixture
def agent_shop(shop_mesh):
    def make(script, agent=None):
        settings = {"enabled": True, "provider": "fake", "model": "fake", **PROMPTS,
                    "sql_generation_prompt_path": str(REPO / "prompts" / "sql_generation.txt"),
                    "clarification_enabled": False, **(agent or {})}
        llm = FakeLLM(script)
        mesh = shop_mesh({"agent": settings}, agent_llm=llm)
        llm.calls.clear()
        return mesh, llm
    return make


# -- execute_sql ---------------------------------------------------------------------------------------
def test_execute_sql_returns_rows_columns_and_status(shop_mesh):
    mesh = shop_mesh()
    out = mesh.execute_sql("shop", BORN)
    assert out["status"] == "ok" and out["dialect"] == "sqlite"
    assert [c["name"] for c in out["columns"]] == ["customer_id", "full_name", "birth_date"]
    assert out["rows"] == [["C1", "Ann Lee", "1985-03-02"], ["C3", "Cid Moe", "1979-11-30"]]
    assert out["row_count"] == 2 and out["truncated"] is False
    assert [t["name"] for t in out["tables"]] == ["customers"]
    cte = mesh.execute_sql("shop", "WITH c AS (SELECT * FROM main.customers) SELECT count(*) AS n FROM c")
    assert cte["status"] == "ok" and cte["rows"] == [[4]]
    join = mesh.execute_sql("shop", "SELECT o.order_id, c.full_name FROM orders o JOIN customers c "
                                    "ON o.customer_id = c.customer_id ORDER BY o.order_id")
    assert join["rows"][0] == [1, "Ann Lee"]
    audit = mesh.db.query("SELECT * FROM audit_events WHERE action = 'execute_sql'")
    assert audit and audit[0]["decision"] == "allowed"


def test_execute_sql_empty_result_and_errors(shop_mesh, parquet_source):
    mesh = shop_mesh()
    empty = mesh.execute_sql("shop", "SELECT * FROM customers WHERE birth_date < '1900-01-01'")
    assert empty["status"] == "ok" and empty["rows"] == [] and empty["row_count"] == 0
    failed = mesh.execute_sql("shop", "SELECT no_such_column FROM customers")
    assert failed["status"] == "error" and failed["error"]["code"] == "execution_failed"
    assert "no_such_column" in failed["error"]["message"] and failed["rows"] == []
    capped = mesh.execute_sql("shop", "SELECT * FROM orders", max_rows=1)
    assert capped["row_count"] == 1 and capped["truncated"] is True
    cases = {
        "DELETE FROM customers": "not_read_only",
        "SELECT * FROM customers; DROP TABLE customers": "not_read_only",
        "UPDATE customers SET full_name = 'x'": "not_read_only",
        "PRAGMA table_info(customers)": "not_read_only",
        "SELECT * INTO copy FROM customers": "not_read_only",
        "SELECT * FROM sqlite_master": "uncataloged_table",
        "SELECT * FROM nowhere.customers": "uncataloged_table",
        "SELEC * FROM": "invalid_sql",
    }
    for sql, code in cases.items():
        res = mesh.execute_sql("shop", sql)
        assert res["status"] == "error" and res["error"]["code"] == code, (sql, res)
    assert mesh.execute_sql("missing", "SELECT 1")["error"]["code"] == "unknown_source"
    denied = mesh.execute_sql("shop", BORN, structured_source_ids=["crm"])
    assert denied["error"]["code"] == "out_of_scope"
    denied = mesh.execute_sql("shop", BORN, dataset_ids=[shop_ids()["orders"]])
    assert denied["error"]["code"] == "out_of_scope"
    assert mesh.db.query("SELECT * FROM audit_events WHERE action = 'execute_sql' AND decision = 'denied'")


def test_execute_sql_supports_database_sources_only(mesh_factory, config_factory, parquet_source, fake_term_llm):
    mesh = mesh_factory(config_factory({"structured_sources": [parquet_source]}), term_llm=fake_term_llm)
    mesh.catalog_source("export", wait_seconds=60)
    res = mesh.execute_sql("export", "SELECT * FROM cust")
    assert res["status"] == "error" and res["error"]["code"] == "unsupported_source"


# -- generate_html -------------------------------------------------------------------------------------
def test_generate_html_escapes_values(shop_mesh):
    mesh = shop_mesh(catalog=False)
    out = mesh.generate_html({"title": "Customers <script>", "blocks": [
        {"type": "paragraph", "text": "<img src=x onerror=alert(1)> & co"},
        {"type": "table", "columns": ["id", "name"], "rows": [["C1", "<b>Ann</b>"], ["C2", None]], "caption": "born"},
        {"type": "list", "items": ["one", "two"], "ordered": True}]})
    html = out["html"]
    assert out["status"] == "ok" and html.startswith('<div class="km-html">')
    assert "<script>" not in html and "&lt;script&gt;" in html and "&lt;img src=x onerror=alert(1)&gt; &amp; co" in html
    assert "<td>&lt;b&gt;Ann&lt;/b&gt;</td>" in html and "<td></td>" in html and "<ol><li>one</li>" in html
    assert set(re.findall(r"</?([a-z0-9]+)", html)) <= set(out["allowed_tags"])
    many = mesh.generate_html({"blocks": [{"type": "table", "columns": ["n"], "rows": [[i] for i in range(600)]}]})
    assert many["truncated"] and "100 more row(s) not shown" in many["html"]
    with pytest.raises(InvalidArgument):
        mesh.generate_html({"blocks": [{"type": "table", "result_ref": "R1"}]})


# -- through the agent --------------------------------------------------------------------------------
def born_script(sql, answer):
    def table(p):  # the executor copies the table from get_dataset: generate_sql gets ALL the evidence it uses
        d = last_result(p)
        return {"name": f"{d['schema']}.{d['name']}", "keys": d["keys"],
                "columns": [{"name": c["name"], "data_type": c.get("data_type"),
                             "business_term": c.get("business_term")} for c in d["columns"]]}

    return ShopScript(plan("list customers born before 1990", ["customers born before 1990",
                                                               "customer birth date before 1990"]), {
        "I1": [tools(call("query_knowledge_graph", intent="list customers born before 1990")),
               tools(call("find_datasets", name_contains="customers")),
               lambda p: tools(call("get_dataset", dataset_id=last_result(p)["datasets"][0]["dataset_id"])),
               lambda p: tools(call("generate_sql", source_id="shop", intent="customers born before 1990",
                                    tables=[table(p)], facts=["Birth Date is the customer's date of birth."])),
               lambda p: tools(call("execute_sql", source_id="shop", sql=last_result(p)["sql"])),
               lambda p: complete(answer(last_result(p)))]},
        sql={"sql": sql, "explanation": "customers.birth_date compared with 1990-01-01"}, synthesis=cite_synthesis)


def test_agent_generates_sql_with_an_llm_runs_it_and_cites_the_rows(agent_shop):
    """AT-21: KG evidence -> generate_sql (LLM) -> execute_sql -> rows used in the answer."""
    mesh, llm = agent_shop(born_script(BORN, lambda r: f"Ann Lee and Cid Moe were born before 1990 [{r['id']}]."))
    res = mesh.search("list customers born before 1990")
    assert res["outcome"] == "answered"
    steps = [c["tool"] for s in res["tasks"][0]["steps"] for c in s["tool_calls"]]
    assert steps == ["query_knowledge_graph", "find_datasets", "get_dataset", "generate_sql", "execute_sql"]
    first = res["tasks"][0]["steps"][0]["tool_calls"][0]
    assert first["arguments"]["intent_id"] == "I1" and "retrieval_queries" in first["arguments"]
    assert "intent fields added by the harness" in first["note"]
    sql_calls = [c for c in llm.calls if c["purpose"] == "sql_generation"]
    assert len(sql_calls) == 1 and "SQL dialect: sqlite" in sql_calls[0]["prompt"]
    assert "table main.customers" in sql_calls[0]["prompt"] and "birth_date TEXT (term: Birth Date)" in sql_calls[0]["prompt"]
    assert "Birth Date is the customer's date of birth." in sql_calls[0]["prompt"]  # only the evidence passed
    assert sql_calls[0]["system"] == (REPO / "prompts" / "sql_generation.txt").read_text().strip()
    gen = res["tasks"][0]["steps"][3]["tool_calls"][0]
    assert gen["status"] == "ok"
    cited = {c["id"]: c for c in res["citations"]}
    rid = next(i for i in cited if i.startswith("R"))
    assert cited[rid]["type"] == "query_result" and cited[rid]["row_count"] == 2
    assert "Ann Lee and Cid Moe" in res["tasks"][0]["answer"]


def test_agent_sql_empty_result_and_execution_error(agent_shop):
    mesh, _ = agent_shop(born_script(BORN.replace("1990", "1900"),
                                     lambda r: f"No customer was born before 1900 [{r['id']}]."))
    res = mesh.search("list customers born before 1900")
    exec_call = res["tasks"][0]["steps"][4]["tool_calls"][0]
    assert exec_call["status"] == "ok" and exec_call["summary"] == "0 rows"
    assert res["outcome"] == "answered" and "No customer" in res["tasks"][0]["answer"]

    mesh, _ = agent_shop(born_script("SELECT missing_column FROM customers",
                                     lambda r: "The query failed: " + r["error"]["message"]))
    res = mesh.search("list customers born before 1990")
    exec_call = res["tasks"][0]["steps"][4]["tool_calls"][0]
    assert exec_call["summary"].startswith("error:") and "missing_column" in exec_call["summary"]


def test_generate_sql_rejects_write_statements(agent_shop):
    mesh, _ = agent_shop(born_script("DELETE FROM customers", lambda r: "done"))
    res = mesh.search("list customers born before 1990")
    gen = res["tasks"][0]["steps"][3]["tool_calls"][0]
    assert "generate_sql invalid" in gen["summary"]
    assert mesh.execute_sql("shop", "SELECT count(*) FROM customers")["rows"] == [[4]]  # nothing was deleted


def test_html_output_is_generated_and_returned_for_display(agent_shop):
    """AT-22: HTML requested -> generate_html renders the rows -> the response carries the fragment."""
    def html_step(p):
        rid = last_result(p)["id"]
        return tools(call("generate_html", title="Customers born before 1990",
                          blocks=[{"type": "table", "result_ref": rid}, {"type": "paragraph", "text": "<b>2</b> rows"}]))

    script = born_script(BORN, lambda r: "Two customers [R1].")
    script.actions["I1"].insert(5, html_step)
    mesh, llm = agent_shop(script)
    res = mesh.search("list customers born before 1990", output_format="html")
    assert any("Output format requested: HTML" in c["prompt"] for c in llm.calls if c["purpose"] == "agent_step")
    assert res["output_format"] == "html" and len(res["html"]) == 1
    html = res["html"][0]["html"]
    assert '<table class="km-grid">' in html and "&lt;b&gt;2&lt;/b&gt; rows" in html
    # full_name and birth_date have business terms flagged sensitive: marked from the R1 result automatically
    assert ('<th>customer_id</th><th class="km-sensitive">full_name</th><th class="km-sensitive">birth_date</th>'
            in html)
    assert '<td>C1</td><td class="km-sensitive">Ann Lee</td><td class="km-sensitive">1985-03-02</td>' in html
    assert "Sensitive data (can identify a person) in red: full_name, birth_date." in html
    assert res["html"][0]["id"].startswith("H")

    client = TestClient(create_app(mesh))
    js = client.get("/static/app.js").text
    assert "sanitizeFragment" in js and "renderHtmlOutput" in js and ".innerHTML" not in js
    with pytest.raises(Exception):
        mesh.search("x", output_format="pdf")

import re
import sqlite3

import pyarrow as pa
import pytest

from knowledge_mesh.common.ids import dataset_id_for
from knowledge_mesh.structured.connectors.databricks import DatabricksConnector
from knowledge_mesh.structured.profiling import format_pattern, infer_content_type
from knowledge_mesh.structured.terms import heuristic_entity_label, parse_term_output, render_prompt
from tests.conftest import REPO


def ds_id(source, schema, name):
    return dataset_id_for(source, schema, name)


@pytest.fixture
def crm_mesh(mesh_factory, config_factory, sqlite_source, parquet_source, fake_term_llm):
    def make(term_llm=fake_term_llm, extra=None):
        cfg = config_factory({"structured_sources": [sqlite_source, parquet_source], **(extra or {})})
        return mesh_factory(cfg, term_llm=term_llm)
    return make


def keys_of(mesh, dataset_id):
    return mesh.get_dataset(dataset_id)["keys"]


def test_format_patterns_and_content_types():
    assert format_pattern("1999-12-22") == "NNNN-NN-NN"
    assert format_pattern("PAUL MERSON") == "TTTT TTTTTT"
    assert infer_content_type(["a@b.com", "c@d.org"], "string", 0.9) == "email"
    assert infer_content_type(["2024-01-02", "2023-05-06"], "string", 0.9) == "date"
    assert infer_content_type(["123", "456"], "string", 0.9) == "integer"
    assert infer_content_type(["x"], "integer", 0.9) == "integer"


def test_catalog_extraction_declared_keys_and_stable_ids(crm_mesh):
    mesh = crm_mesh()
    job = mesh.extract_structured_metadata("crm", wait_seconds=60)
    assert job["status"] == "succeeded", job
    cust = mesh.get_dataset(ds_id("crm", "main", "customers"))
    assert cust["catalog_ref"] == "catalog:crm/main/customers"
    assert [c["name"] for c in cust["columns"]] == ["customer_id", "email", "first_name", "region"]
    assert cust["columns"][0]["catalog_ref"] == "catalog:crm/main/customers/customer_id"
    declared = {(k["key_type"], tuple(k["columns"])) for k in cust["keys"] if k["origin"] == "declared"}
    assert ("primary", ("customer_id",)) in declared and ("unique", ("email",)) in declared
    orders = mesh.get_dataset(ds_id("crm", "main", "orders"))
    fk = next(k for k in orders["keys"] if k["key_type"] == "foreign" and k["origin"] == "declared")
    assert fk["mapping"] == [{"from": "customer_id", "to": "customer_id"}]
    assert fk["referenced_dataset_id"] == ds_id("crm", "main", "customers")
    # parquet source
    job = mesh.extract_structured_metadata("export", wait_seconds=60)
    pq_ds = mesh.get_dataset(ds_id("export", "files", "cust"))
    assert pq_ds["row_count"] == 12 and len(pq_ds["columns"]) == 3


def test_key_discovery_unique_composite_and_foreign_keys(crm_mesh):
    mesh = crm_mesh()
    mesh.extract_structured_metadata("crm", wait_seconds=60)
    ids = [ds_id("crm", "main", t) for t in ("customers", "orders", "stores", "sales")]
    # sales(cust, ctry, store) are identifiers only by their business terms (Customer Identifier,
    # Country Code, Store Code): generate the terms first, as catalog_source does.
    assert mesh.generate_terms(ids, wait_seconds=120)["status"] == "succeeded"
    job = mesh.discover_keys(ids, wait_seconds=120)
    assert job["status"] == "succeeded", job
    result = job["result"]
    assert result["datasets_profiled"] == 4 and result["inferred_foreign_keys"] >= 2

    sales = keys_of(mesh, ids[3])
    inferred_unique = {tuple(k["columns"]) for k in sales if k["key_type"] == "unique" and k["origin"] == "inferred"}
    assert ("sale_id",) in inferred_unique
    stores = keys_of(mesh, ids[2])
    store_unique = {tuple(k["columns"]) for k in stores if k["key_type"] == "unique"}
    assert ("country", "store_code") in store_unique and ("store_name",) in store_unique
    assert ("country",) not in store_unique  # composite only: no single-column key on country
    composite = next(k for k in stores if tuple(k["columns"]) == ("country", "store_code"))
    assert composite["origin"] == "inferred" and composite["primary_key_candidate"] is True
    assert composite["scope"]["rows_evaluated"] == 9 and composite["profiling_run_id"] == result["profiling_run_id"]

    fks = {(tuple(k["columns"]), k["referenced_dataset_id"], tuple(k["referenced_columns"])): k
           for k in sales if k["key_type"] == "foreign"}
    assert (("cust",), ids[0], ("customer_id",)) in fks  # single-column FK candidate
    comp = fks[(("ctry", "store"), ids[2], ("country", "store_code"))]  # composite FK candidate
    assert comp["origin"] == "inferred"
    assert comp["mapping"] == [{"from": "ctry", "to": "country"}, {"from": "store", "to": "store_code"}]

    orders = keys_of(mesh, ids[1])
    origins = {k["origin"] for k in orders if k["key_type"] == "foreign" and k["columns"] == ["customer_id"]}
    assert origins == {"declared", "inferred"}  # declared FK and the matching inferred candidate stay distinct

    cust = mesh.get_dataset(ids[0])
    prof = next(c for c in cust["columns"] if c["name"] == "email")["profile"]
    assert prof["distinct_count"] == 12 and prof["null_count"] == 0 and prof["content_type"] == "email"
    assert prof["most_common_values"] and prof["format_patterns"]


def test_sampled_scope_is_labeled(crm_mesh):
    mesh = crm_mesh()
    mesh.extract_structured_metadata("crm", wait_seconds=60)
    sales = ds_id("crm", "main", "sales")
    mesh.discover_keys([sales], max_rows=10, wait_seconds=60)
    key = next(k for k in keys_of(mesh, sales) if k["origin"] == "inferred")
    assert key["scope"]["sampled"] is True and key["scope"]["rows_evaluated"] == 10 and key["scope"]["total_rows"] == 30


def test_term_generation_glossary_and_graph(crm_mesh):
    mesh = crm_mesh()
    job = mesh.catalog_source("crm", wait_seconds=120)
    assert job["status"] == "succeeded", job
    cust = mesh.get_dataset(ds_id("crm", "main", "customers"))
    assert cust["business_entity"]["label"] == "Customer" and cust["business_entity"]["method"] == "llm_decoded_table"
    col = next(c for c in cust["columns"] if c["name"] == "customer_id")
    assert col["business_term"]["label"] == "Customer Identifier"
    assert col["business_term"]["description"] == "The unique identifier assigned to each customer"
    assert col["business_term"]["explanation"] == "customer_id=customer identifier"
    assert all(c["business_term"] for c in cust["columns"])

    res = mesh.retrieve_documents(["customer identifier"], evidence_types=["glossary"])
    top = res["glossary_results"][0]
    assert top["label"] == "Customer Identifier" and top["type"] == "business_term"
    assert top["catalog_ref"].startswith("catalog:crm/main/") or top["catalog_ref"].startswith("catalog:")
    assert top["score_type"] == "cosine_similarity"
    ent = mesh.retrieve_documents(["customer business entity"], evidence_types=["glossary"])["glossary_results"]
    assert any(g["type"] == "business_entity" and g["label"] == "Customer" for g in ent)
    term = mesh.get_business_term("customer-identifier")
    assert {c["catalog_ref"] for c in term["columns"]} >= {"catalog:crm/main/customers/customer_id"}
    stats = mesh.graph.stats()
    assert stats["terms"] > 0 and stats["keys"] > 0


def test_catalog_works_without_term_model(mesh_factory, config_factory, sqlite_source):
    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source]}))
    job = mesh.catalog_source("crm", wait_seconds=120)
    assert job["status"] == "succeeded"
    items = mesh.job_items(job["job_id"], limit=200)["items"]
    assert any(i["ref"] == "term_generation" and i["status"] == "skipped" for i in items)
    assert keys_of(mesh, ds_id("crm", "main", "customers"))


def test_term_failure_reported_against_dataset(crm_mesh):
    from knowledge_mesh.llm.client import FakeLLM

    def broken(prompt, system, purpose):
        raise RuntimeError("model exploded")

    mesh = crm_mesh(term_llm=FakeLLM(broken))
    job = mesh.catalog_source("crm", wait_seconds=120)
    assert job["status"] == "partial"
    failed = mesh.job_items(job["job_id"], status="failed", limit=50)["items"]
    assert {i["ref"] for i in failed} >= {ds_id("crm", "main", "customers")}
    assert all("term generation failed" in i["reason"] for i in failed)


def test_structured_refresh_is_idempotent_and_retires_stale(crm_mesh, tmp_path):
    mesh = crm_mesh()
    mesh.catalog_source("crm", wait_seconds=120)
    snapshot = lambda: (mesh.db.scalar("SELECT COUNT(*) FROM columns WHERE active=1"),  # noqa: E731
                        mesh.db.scalar("SELECT COUNT(*) FROM keys WHERE active=1"),
                        mesh.glossary_index.stats()["embedded_nodes"], mesh.graph.stats()["triples"])
    before = snapshot()
    calls_before = len(mesh.terms.llm().calls)
    mesh.catalog_source("crm", wait_seconds=120)
    assert snapshot() == before
    assert len(mesh.terms.llm().calls) == calls_before  # unchanged schema: no regeneration

    db_path = tmp_path / "crm.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript("ALTER TABLE customers DROP COLUMN region; DROP TABLE stores;")
    conn.close()
    mesh.catalog_source("crm", wait_seconds=120)
    cust = mesh.get_dataset(ds_id("crm", "main", "customers"))
    assert "region" not in [c["name"] for c in cust["columns"]]
    assert mesh.get_dataset(ds_id("crm", "main", "stores"))["active"] is False
    labels = {g["column"] for g in mesh.retrieve_documents(["customer region code"], top_k=20,
                                                          evidence_types=["glossary"])["glossary_results"]}
    assert "region" not in labels
    sales_keys = keys_of(mesh, ds_id("crm", "main", "sales"))
    assert not any(k.get("referenced_dataset_id") == ds_id("crm", "main", "stores") for k in sales_keys)


def test_prompt_rendering_and_output_validation():
    template = (REPO / "prompts" / "term_generation.txt").read_text()
    from knowledge_mesh.config import TermGenerationSettings

    settings = TermGenerationSettings(industry_name="Retail", class_words=["Identifier", "Name"])
    prompt = render_prompt(template, "cust", ["cust_id", "fst_nm"], settings, {"cust": "customer"})
    # the prompt file may write its labels as plain text or markdown (*Table:*)
    assert re.search(r"\*{0,2}Table:\*{0,2} cust\n", prompt) and re.search(r"\*{0,2}Columns:\*{0,2} cust_id, fst_nm", prompt)
    assert "cust=customer" in prompt
    assert "{" not in prompt.replace("{industry", "")  # every placeholder filled
    text = ("Customer,cust_id,Customer Identifier,The id, unique per customer,cust=customer | id=identifier\n"
            "Customer,unknown_col,Ghost,x,y\n"
            "Customer,fst_nm,Customer First Name,Given name, cust=customer | fst=first | nm=name\n")
    records, issues = parse_term_output(text, "cust", ["cust_id", "fst_nm"])
    assert records["cust_id"].description == "The id, unique per customer"
    assert records["cust_id"].explanation == "cust=customer | id=identifier"
    assert records["fst_nm"].description == "Given name"
    assert any("unknown column" in i for i in issues)
    assert heuristic_entity_label("tbl_customer_orders", {}) == "Customer Order"


def test_databricks_connector_with_fake_driver(monkeypatch):
    from knowledge_mesh.config import StructuredSourceSettings

    monkeypatch.setenv("DBX_HOST", "adb.example")
    monkeypatch.setenv("DBX_PATH", "/sql/1.0/warehouses/x")
    monkeypatch.setenv("DBX_TOKEN", "secret")
    responses = {
        "information_schema.tables": [("customers", "MANAGED")],
        "information_schema.columns": [("customer_id", "STRING", "NO", 1), ("name", "STRING", "YES", 2)],
        "information_schema.table_constraints": [("pk_c", "PRIMARY KEY")],
        "information_schema.key_column_usage": [("customer_id", 1)],
        "COUNT(*)": [(3,)],
    }
    seen_kwargs = {}

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql): self.sql = sql
        def fetchall(self): return next(v for k, v in responses.items() if k in self.sql)
        def fetchall_arrow(self): return pa.table({"customer_id": ["a", "b"], "name": ["x", None]})

    class Conn:
        def cursor(self): return Cursor()
        def close(self): pass

    def connect(**kwargs):
        seen_kwargs.update(kwargs)
        return Conn()

    settings = StructuredSourceSettings(id="orders", kind="databricks", server_hostname_env="DBX_HOST",
                                        http_path_env="DBX_PATH", token_env="DBX_TOKEN", catalog="sales", schema="public")
    conn = DatabricksConnector(settings, None, connect=connect)
    assert seen_kwargs["access_token"] == "secret"
    refs = conn.list_datasets()
    schema = conn.describe(refs[0])
    assert [c.name for c in schema.columns] == ["customer_id", "name"]
    assert schema.keys[0].key_type == "primary" and schema.keys[0].columns == ["customer_id"]
    assert schema.row_count == 3 and schema.physical_ref == "databricks:sales.public.customers"
    assert conn.read_sample(refs[0], ["customer_id", "name"], 10).num_rows == 2


def test_missing_credentials_report_unavailable_source(mesh_factory, config_factory):
    cfg = config_factory({"structured_sources": [{"id": "crm", "kind": "database", "connection_url_env": "NOPE_URL"}]})
    mesh = mesh_factory(cfg)
    job = mesh.extract_structured_metadata("crm", wait_seconds=30)
    assert job["status"] == "failed" and job["error"]["code"] == "source_unavailable"
    assert "NOPE_URL" in job["error"]["message"]

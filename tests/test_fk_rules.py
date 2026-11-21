"""Foreign keys are identifiers and share the business term of the key they reference."""

import uuid

import pyarrow as pa
import pytest

from knowledge_mesh.common.ids import dataset_id_for
from knowledge_mesh.structured.fk_rules import FKRules
from knowledge_mesh.structured.keys.binder import Binder, DatasetValues, normalize_value
from knowledge_mesh.structured.profiling import profile_column, type_category
from knowledge_mesh.config import ProfilingSettings


@pytest.fixture
def crm_mesh_factory(mesh_factory, config_factory, sqlite_source, parquet_source, fake_term_llm):
    def make():
        cfg = config_factory({"structured_sources": [sqlite_source, parquet_source]})
        return mesh_factory(cfg, term_llm=fake_term_llm)
    return make


def col(name, term=None, data_type="TEXT"):
    return {"name": name, "data_type": data_type, "term_label": term, "term_status": "generated" if term else None}


def test_identifier_basis():
    rules = FKRules.from_settings(ProfilingSettings())
    assert rules.identifier_basis(col("appraisal_date"), declared=True) == "declared_key"
    assert rules.identifier_basis(col("loan_ref", data_type="UUID")) == "uuid"
    assert rules.identifier_basis(col("x"), content_type="uuid") == "uuid"
    assert rules.identifier_basis(col("cust", "Customer Identifier")) == "business_term"
    assert rules.identifier_basis(col("prd_cd", "Product Code")) == "business_term"
    assert rules.identifier_basis(col("enforceable_flg", "Netting Agreement Enforceable Indicator")) == "business_term"
    assert rules.identifier_basis(col("report_id", "Report Date")) is None  # the term decides over the name
    assert rules.identifier_basis(col("appraised_value", "Appraised Value Amount")) is None
    for name in ("loan_id", "prd_cd", "loanId", "ln_no", "active_flg", "risk_ind", "natural_key"):
        assert rules.identifier_basis(col(name)) == "column_name", name
    for name in ("appraisal_date", "premium", "address1", "cust"):
        assert rules.identifier_basis(col(name)) is None, name
    narrow = FKRules.from_settings(ProfilingSettings(fk_identifier_class_words=["Identifier"]))
    assert narrow.identifier_basis(col("prd_cd")) is None and narrow.identifier_basis(col("loan_id"))


def test_term_and_name_matching():
    assert FKRules.term_match("Loan Identifier", "loan identifier")
    assert FKRules.term_match("Parent Counterparty Identifier", "Counterparty Identifier")  # role qualifier
    assert FKRules.term_match("Facility Identifier", "Loan Facility Identifier")
    assert not FKRules.term_match("Loan Identifier", "Identifier")  # a class word alone says nothing
    assert not FKRules.term_match("Loan Identifier", "Pool Identifier")
    assert FKRules.name_match("loan_id", "loan_id")
    assert FKRules.name_match("parent_cpty_id", "cpty_id")
    assert FKRules.name_match("customer_id", "id", "customers")  # <table>_id -> id
    assert not FKRules.name_match("order_id", "id", "customers")
    assert not FKRules.name_match("premium", "insurance_amount")


def test_confidence_levels():
    rules = FKRules()
    same_term = rules.assess([(col("loan_id", "Loan Identifier"), col("loan_id", "Loan Identifier"))], "loan", set(), set())
    assert same_term["level"] == "high" and same_term["basis"] == "business_term"
    role = rules.assess([(col("parent_cpty_id", "Parent Counterparty Identifier"), col("cpty_id", "Counterparty Identifier"))],
                        "lgl_entity", set(), set())
    assert role["level"] == "high"
    no_terms = rules.assess([(col("loan_id"), col("loan_id"))], "loan", set(), set())
    assert no_terms["level"] == "medium" and "not generated yet" in no_terms["reasons"][0]
    values_only = rules.assess([(col("cust_ref"), col("customer_id"))], "customers", set(), set())
    assert values_only["level"] == "low" and values_only["basis"] == "values_only"
    demoted = rules.assess([(col("issue_id", "Issue Date"), col("issue_id", "Issue Date"))], "pool", set(), set())
    assert demoted["level"] == "low" and "not an identifier" in demoted["reasons"][0]
    # composite: the weakest pair decides
    mixed = rules.assess([(col("ctry", "Country Code"), col("country", "Store Country Code")),
                          (col("store"), col("store_code"))], "stores", set(), set())
    assert mixed["level"] == "low" and len(mixed["reasons"]) == 2


def test_uuid_columns_are_profiled_and_matched():
    ids = [uuid.uuid4() for _ in range(4)]
    arr = pa.array(ids)
    assert type_category(arr.type) == "uuid"
    prof = profile_column(arr, ProfilingSettings())
    assert prof["category"] == "uuid" and prof["content_type"] == "uuid" and prof["distinct_count"] == 4
    assert normalize_value(ids[0]) == str(ids[0])

    loan = DatasetValues("loan", {"loan_id": [str(i) for i in ids], "opened": ["2024-01-01", "2024-01-02",
                                                                          "2024-01-03", "2024-01-04"]},
                         {"loan_id": "uuid", "opened": "date"})
    # payments.loan_id holds the same UUIDs as text; paid_on dates are a subset of loan.opened
    payments = DatasetValues("payments", {"loan_id": [str(ids[0]), str(ids[1]).upper().lower(), str(ids[1])],
                                          "paid_on": ["2024-01-01", "2024-01-02", "2024-01-02"]},
                             {"loan_id": "string", "paid_on": "date"})
    data = {"loan": loan, "payments": payments}
    keys = {"loan": [("loan_id",), ("opened",)], "payments": []}
    everything = Binder().discover(data, keys, 2)
    assert {(i.dep_dataset, i.dep_columns, i.ref_columns) for i in everything} == {
        ("payments", ("loan_id",), ("loan_id",)), ("payments", ("paid_on",), ("opened",))}
    identifiers = {("loan", "loan_id"), ("payments", "loan_id")}
    binder = Binder()
    only_ids = binder.discover(data, keys, 2, eligible=lambda a: a in identifiers)
    assert [(i.dep_columns, i.ref_columns) for i in only_ids] == [(("loan_id",), ("loan_id",))]
    assert binder.stats["excluded_columns"] == 2


def test_identifier_fks_and_confidence_follow_terms(crm_mesh_factory):
    mesh = crm_mesh_factory()
    mesh.extract_structured_metadata("crm", wait_seconds=60)
    ids = {t: dataset_id_for("crm", "main", t) for t in ("customers", "orders", "stores", "sales")}

    def inferred_fks(table):
        return {(tuple(k["columns"]), tuple(k["referenced_columns"])): k
                for k in mesh.get_dataset(ids[table])["keys"] if k["key_type"] == "foreign" and k["origin"] == "inferred"}

    # 1. No business terms yet: identifiers come from names and declared keys only.
    job = mesh.discover_keys(list(ids.values()), wait_seconds=120)
    assert job["status"] == "succeeded", job
    run = mesh.keys.get_run(job["result"]["profiling_run_id"])
    assert run["algorithm"]["fk_identifiers_only"] is True
    orders = inferred_fks("orders")
    fk = orders[(("customer_id",), ("customer_id",))]
    assert fk["confidence"]["level"] == "medium" and fk["confidence"]["basis"] == "column_name"
    assert fk["metrics"]["identifier_basis"] == {"dependent": ["declared_key"], "referenced": ["declared_key"]}
    assert not inferred_fks("sales")  # cust / ctry / store do not look like identifiers by name

    # 2. Terms generated later: the stored FK is re-graded on read, without re-running discovery.
    assert mesh.generate_terms(list(ids.values()), wait_seconds=120)["status"] == "succeeded"
    fk = inferred_fks("orders")[(("customer_id",), ("customer_id",))]
    assert fk["confidence"]["level"] == "high"
    assert "same business term 'Customer Identifier'" in fk["confidence"]["reasons"][0]
    from knowledge_mesh.graph.ontology import MESH, RDF  # graph rebuilt by generate_terms: same grading
    graph_fks = [mesh.graph._key(n) for n in set(mesh.graph.dataset.subjects(RDF.type, MESH.ForeignKey))]
    graph_fks = [k for k in graph_fks if k and k["inferred"] and "orders(customer_id)" in k["label"]]
    assert graph_fks and graph_fks[0]["provenance"]["confidence"] == "high"
    assert "confidence high (business_term)" in graph_fks[0]["text"]

    # 3. Re-running discovery now finds the FKs whose columns are identifiers by their terms.
    assert mesh.discover_keys(list(ids.values()), wait_seconds=120)["status"] == "succeeded"
    sales = inferred_fks("sales")
    assert sales[(("cust",), ("customer_id",))]["confidence"]["level"] == "high"
    composite = sales[(("ctry", "store"), ("country", "store_code"))]
    assert composite["confidence"]["level"] == "high"  # Country Code qualifies Store Country Code; Store Code
    for table in ids:  # never a date, amount, quantity or name
        for cols, _ in inferred_fks(table):
            assert not set(cols) & {"order_date", "amount", "qty", "first_name", "store_name", "email"}, cols


def test_agent_dataset_tool_reports_fk_confidence(crm_mesh_factory):
    from knowledge_mesh.agent.ledger import EvidenceLedger
    from knowledge_mesh.agent.tools import build_search_tools

    mesh = crm_mesh_factory()
    assert mesh.catalog_source("crm", wait_seconds=120)["status"] == "succeeded"
    tools = build_search_tools(mesh, EvidenceLedger(), None, 20_000)
    out = tools.call("get_dataset", {"dataset_id": dataset_id_for("crm", "main", "sales")}).observation
    fks = [k for k in out["keys"] if k["type"] == "foreign" and k["origin"] == "inferred"]
    assert fks and all(k["confidence"] in {"high", "medium", "low"} for k in fks)


def test_fks_reference_primary_keys_not_sibling_columns(mesh_factory, config_factory, tmp_path, monkeypatch):
    """insurance.loan_id is unique in the sample but is not insurance's key: escrow.loan_id -> loan only."""
    import sqlite3

    path = tmp_path / "lend.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE loan (loan_id TEXT PRIMARY KEY, opened TEXT, amount REAL);
        CREATE TABLE insurance (insurance_id TEXT PRIMARY KEY, loan_id TEXT, premium REAL);
        CREATE TABLE escrow (escrow_id TEXT PRIMARY KEY, loan_id TEXT, insurance_amount REAL);
    """)
    conn.executemany("INSERT INTO loan VALUES (?,?,?)", [(f"L{i}", f"2024-01-0{i}", 100.0 * i) for i in range(1, 7)])
    conn.executemany("INSERT INTO insurance VALUES (?,?,?)", [(f"I{i}", f"L{i}", 10.0 * i) for i in range(1, 6)])
    conn.executemany("INSERT INTO escrow VALUES (?,?,?)", [(f"E{n}", f"L{i}", 10.0 * i) for n, i in enumerate((1, 2, 2, 3))])
    conn.commit()
    conn.close()
    monkeypatch.setenv("TEST_LEND_URL", f"sqlite:///{path}")

    def fks(settings):
        cfg = config_factory({"structured_sources": [{"id": "lend", "kind": "database",
                                                       "connection_url_env": "TEST_LEND_URL"}],
                              "profiling": settings})
        mesh = mesh_factory(cfg)
        mesh.extract_structured_metadata("lend", wait_seconds=60)
        ids = [dataset_id_for("lend", "main", t) for t in ("loan", "insurance", "escrow")]
        assert mesh.discover_keys(ids, wait_seconds=120)["status"] == "succeeded"
        names = {i: t for i, t in zip(ids, ("loan", "insurance", "escrow"))}
        return {(names[k["dataset_id"]], tuple(k["columns"]), names[k["referenced_dataset_id"]])
                for i in ids for k in mesh.get_dataset(i)["keys"] if k["key_type"] == "foreign" and k["origin"] == "inferred"}

    assert fks({}) == {("escrow", ("loan_id",), "loan"), ("insurance", ("loan_id",), "loan")}
    # the old behaviour, any unique column as target, also links the siblings
    assert ("escrow", ("loan_id",), "insurance") in fks({"fk_reference_primary_keys": False})

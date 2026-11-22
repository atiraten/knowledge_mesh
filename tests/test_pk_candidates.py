"""Primary key candidates: a declared primary key needs no inferred candidate (and, when FKs target primary
keys, no unique-key search); otherwise the unique key referenced by other tables (declared FKs, inferred FKs of
high/medium confidence) is the PK candidate, the most referenced one when several are; with no reference,
every null-free unique key stays a candidate."""

import sqlite3

import pytest

from knowledge_mesh.common.ids import dataset_id_for
from knowledge_mesh.structured import key_discovery
from knowledge_mesh.structured.catalog import CatalogService
from knowledge_mesh.structured.key_discovery import choose_pk_candidates

TABLES = """
    CREATE TABLE account (account_id TEXT PRIMARY KEY, email TEXT, nickname TEXT);
    CREATE TABLE branch (branch_id TEXT, branch_code TEXT, branch_name TEXT);
    CREATE TABLE teller (teller_id TEXT PRIMARY KEY, branch_id TEXT);
    CREATE TABLE region (region_id TEXT, region_code TEXT);
    CREATE TABLE vendor (vendor_id TEXT, vendor_code TEXT);
    CREATE TABLE invoice (invoice_id TEXT PRIMARY KEY, vendor_id TEXT);
    CREATE TABLE shipment (shipment_id TEXT PRIMARY KEY, vendor_id TEXT);
    CREATE TABLE contract (contract_id TEXT PRIMARY KEY, vendor_code TEXT);
    CREATE TABLE product (product_id TEXT, product_code TEXT UNIQUE);
    CREATE TABLE sale (sale_id TEXT PRIMARY KEY, product_code TEXT REFERENCES product(product_code));
"""
NAMES = ("account", "branch", "teller", "region", "vendor", "invoice", "shipment", "contract", "product", "sale")


@pytest.fixture
def bank(tmp_path, monkeypatch):
    path = tmp_path / "bank.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(TABLES)
    six = range(1, 7)
    conn.executemany("INSERT INTO account VALUES (?,?,?)", [(f"A{i}", f"a{i}@x.org", f"nick{i}") for i in six])
    conn.executemany("INSERT INTO branch VALUES (?,?,?)", [(f"B{i}", f"BC{i}", f"Branch {i}") for i in six])
    conn.executemany("INSERT INTO teller VALUES (?,?)", [(f"T{i}", f"B{1 + i % 4}") for i in range(8)])
    conn.executemany("INSERT INTO region VALUES (?,?)", [(f"G{i}", f"RC{i}") for i in six])
    conn.executemany("INSERT INTO vendor VALUES (?,?)", [(f"V{i}", f"VC{i}") for i in six])
    conn.executemany("INSERT INTO invoice VALUES (?,?)", [(f"I{i}", f"V{1 + i % 3}") for i in range(7)])
    conn.executemany("INSERT INTO shipment VALUES (?,?)", [(f"S{i}", f"V{1 + i % 5}") for i in range(7)])
    conn.executemany("INSERT INTO contract VALUES (?,?)", [(f"K{i}", f"VC{1 + i % 4}") for i in range(7)])
    conn.executemany("INSERT INTO product VALUES (?,?)", [(f"P{i}", f"PC{i}") for i in six])
    conn.executemany("INSERT INTO sale VALUES (?,?)", [(f"X{i}", f"PC{1 + i % 4}") for i in range(7)])
    conn.commit()
    conn.close()
    monkeypatch.setenv("TEST_BANK_URL", f"sqlite:///{path}")
    return {"id": "bank", "kind": "database", "connection_url_env": "TEST_BANK_URL"}


def discover(mesh_factory, config_factory, source, profiling=None):
    mesh = mesh_factory(config_factory({"structured_sources": [source], "profiling": profiling or {}}))
    mesh.extract_structured_metadata("bank", wait_seconds=60)
    ids = {t: dataset_id_for("bank", "main", t) for t in NAMES}
    result = mesh.discover_keys(list(ids.values()), wait_seconds=120)
    assert result["status"] == "succeeded", result
    return mesh, ids, result


def unique_keys(mesh, dataset_id):
    """{columns: is PK candidate} of the table's active inferred unique keys."""
    return {tuple(k["columns"]): k["primary_key_candidate"] for k in mesh.get_dataset(dataset_id)["keys"]
            if k["key_type"] == "unique" and k["origin"] == "inferred"}


def test_choose_pk_candidates():
    keys = {("a",): True, ("b",): True, ("c",): False, ("d", "e"): True}
    assert choose_pk_candidates(keys, {}) == {("a",), ("b",), ("d", "e")}  # nothing referenced: today's rule
    assert choose_pk_candidates(keys, {frozenset({"b"}): 1}) == {("b",)}  # the referenced one only
    assert choose_pk_candidates(keys, {frozenset({"a"}): 1, frozenset({"b"}): 3}) == {("b",)}  # most referenced
    assert choose_pk_candidates(keys, {frozenset({"b"}): 2, frozenset({"e", "d"}): 2}) == {("b",)}  # tie: fewer columns
    assert choose_pk_candidates(keys, {frozenset({"c"}): 5}) == {("a",), ("b",), ("d", "e")}  # nulls: never a PK


def test_declared_primary_key_needs_no_inferred_candidate(mesh_factory, config_factory, bank):
    mesh, ids, result = discover(mesh_factory, config_factory, bank)
    account = mesh.get_dataset(ids["account"])
    assert unique_keys(mesh, ids["account"]) == {}  # no unique-key search: email and nickname are not listed
    assert [k["columns"] for k in account["keys"] if k["key_type"] == "primary"] == [["account_id"]]
    run = mesh.get_profiling_run(result["result"]["profiling_run_id"])
    assert run["scope"]["datasets"][ids["account"]]["ucc_search_skipped"] == "declared primary key"


def test_declared_pk_table_keeps_unique_keys_when_fks_may_reference_any(mesh_factory, config_factory, bank):
    mesh, ids, _ = discover(mesh_factory, config_factory, bank, {"fk_reference_primary_keys": False})
    keys = unique_keys(mesh, ids["account"])
    assert ("email",) in keys and not any(keys.values())  # searched (FK targets), but never a PK candidate


def test_primary_key_declared_after_discovery_clears_candidates(mesh_factory, config_factory, bank, tmp_path):
    mesh, ids, _ = discover(mesh_factory, config_factory, bank)
    assert unique_keys(mesh, ids["region"]) == {("region_id",): True, ("region_code",): True}
    conn = sqlite3.connect(tmp_path / "bank.sqlite")  # the source now declares region's primary key
    conn.executescript("""
        ALTER TABLE region RENAME TO region_old;
        CREATE TABLE region (region_id TEXT PRIMARY KEY, region_code TEXT);
        INSERT INTO region SELECT * FROM region_old;
        DROP TABLE region_old;
    """)
    conn.close()
    mesh.extract_structured_metadata("bank", wait_seconds=60)  # catalog refresh, no new key discovery
    keys = unique_keys(mesh, ids["region"])
    assert keys and not any(keys.values())


def test_stored_candidates_of_declared_pk_tables_are_cleared_at_startup(mesh_factory, config_factory, bank):
    mesh, ids, _ = discover(mesh_factory, config_factory, bank, {"fk_reference_primary_keys": False})
    mesh.db.execute("UPDATE keys SET is_pk_candidate = 1 WHERE dataset_id = ? AND origin = 'inferred'",
                    (ids["account"],))  # as stored by an older version
    assert any(unique_keys(mesh, ids["account"]).values())
    CatalogService(mesh.config, mesh.db, mesh.sources)
    keys = unique_keys(mesh, ids["account"])
    assert keys and not any(keys.values())
    assert unique_keys(mesh, ids["region"]) == {("region_id",): True, ("region_code",): True}  # no declared PK


def test_referenced_unique_key_is_the_pk_candidate(mesh_factory, config_factory, bank):
    mesh, ids, result = discover(mesh_factory, config_factory, bank)
    # teller.branch_id -> branch.branch_id (inferred, same name): branch_code and branch_name are only unique
    assert unique_keys(mesh, ids["branch"]) == {("branch_id",): True, ("branch_code",): False,
                                                ("branch_name",): False}
    # nothing references region: both null-free unique keys stay candidates (today's rule)
    assert unique_keys(mesh, ids["region"]) == {("region_id",): True, ("region_code",): True}
    # vendor_id is referenced by invoice and shipment, vendor_code by contract only: the most referenced wins
    assert unique_keys(mesh, ids["vendor"]) == {("vendor_id",): True, ("vendor_code",): False}
    # sale.product_code REFERENCES product(product_code) (declared): product_id is only unique
    assert unique_keys(mesh, ids["product"]) == {("product_code",): True, ("product_id",): False}
    assert result["result"]["primary_key_candidates"] == 1 + 2 + 1 + 1  # branch, region, vendor, product


def test_low_confidence_references_do_not_count(mesh_factory, config_factory, bank, monkeypatch):
    real = key_discovery.assess_foreign_key

    def assess(db, key, rules=None):  # make every inferred FK into vendor value-only (low confidence)
        out = real(db, key, rules)
        return {**(out or {}), "level": "low"} if key.get("referenced_dataset_id") == vendor else out

    vendor = dataset_id_for("bank", "main", "vendor")

    monkeypatch.setattr(key_discovery, "assess_foreign_key", assess)
    mesh, ids, _ = discover(mesh_factory, config_factory, bank)
    assert unique_keys(mesh, ids["vendor"]) == {("vendor_id",): True, ("vendor_code",): True}  # as if unreferenced

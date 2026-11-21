import json

import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.api.app import create_app
from knowledge_mesh.common.errors import Conflict, Forbidden, InvalidArgument
from knowledge_mesh.common.secrets import MemorySecretStore, mask_secrets
from tests.conftest import build_parquet, build_sqlite

SECRET = "Sup3r-Secret-Pw"


@pytest.fixture
def store():
    return MemorySecretStore()


@pytest.fixture
def mesh(mesh_factory, store, fake_term_llm):
    return mesh_factory(secret_store=store, term_llm=fake_term_llm)


def sqlite_url(tmp_path):
    db = build_sqlite(tmp_path / "crm.sqlite")
    return f"sqlite:///{db}"


def test_add_database_with_keychain_secret_and_catalog_it(mesh, store, tmp_path):
    url = sqlite_url(tmp_path)
    out = mesh.add_source({"id": "crm", "kind": "database", "connection_url": url, "description": "CRM"})
    assert out["credentials"] == {"store": "secret_store", "reference": "memory: crm:connection_url",
                                  "status": "present"}
    assert store.get("crm", "connection_url") == url
    stored = json.loads(mesh.db.query_one("SELECT settings_json FROM source_configs WHERE source_id = 'crm'")
                        ["settings_json"])
    assert "connection_url" not in stored and stored["credential_store"] == "secret_store"
    assert out["origin"] == "settings" and out["editable"] is True
    test = mesh.test_source(source_id="crm")
    assert test["ok"] and test["dataset_count"] == 4
    job = mesh.catalog_source("crm", wait_seconds=120)
    assert job["status"] == "succeeded", job
    assert any(s["source_id"] == "crm" and s["dataset_count"] == 4 for s in mesh.catalog_tree())


def test_add_database_by_env_var(mesh, tmp_path, monkeypatch):
    monkeypatch.setenv("UI_CRM_URL", sqlite_url(tmp_path))
    out = mesh.add_source({"id": "crm_env", "kind": "database", "connection_url_env": "UI_CRM_URL"})
    assert out["credentials"] == {"store": "env", "reference": "UI_CRM_URL", "status": "present"}
    assert mesh.test_source(source_id="crm_env")["ok"]


def test_parquet_repository_must_be_under_allowed_roots(mesh, tmp_path):
    root = mesh.config.storage.allowed_source_roots[0]
    build_parquet(root / "exports")
    out = mesh.add_source({"id": "exports", "kind": "parquet", "path": "exports"})
    assert out["target"] == str((root / "exports").resolve())
    assert mesh.test_source(source_id="exports")["datasets"] == ["files.cust"]
    outside = build_parquet(tmp_path / "elsewhere")
    with pytest.raises(Forbidden) as err:
        mesh.add_source({"id": "outside", "kind": "parquet", "path": str(outside)})
    assert "allowed_roots" in err.value.details


def test_validation_and_conflicts(mesh, tmp_path):
    with pytest.raises(InvalidArgument) as err:
        mesh.add_source({"id": "dbx", "kind": "databricks", "catalog": "sales"})
    assert {"server_hostname", "http_path"} <= set(err.value.details["fields"])
    with pytest.raises(InvalidArgument):
        mesh.add_source({"id": "bad id!", "kind": "database", "connection_url_env": "X"})
    with pytest.raises(InvalidArgument):
        mesh.add_source({"id": "nourl", "kind": "database"})
    with pytest.raises(InvalidArgument):
        mesh.add_source({"id": "junk", "kind": "database", "connection_url": "not a url"})
    mesh.add_source({"id": "dbx", "kind": "databricks", "server_hostname": "adb-1.azuredatabricks.net",
                     "http_path": "/sql/1.0/warehouses/x", "catalog": "sales", "schema": "public", "token": "dapiX"})
    with pytest.raises(Conflict):
        mesh.add_source({"id": "dbx", "kind": "parquet", "path": "x"})


def test_update_keeps_or_replaces_secret(mesh, store, tmp_path, monkeypatch):
    url = sqlite_url(tmp_path)
    mesh.add_source({"id": "crm", "kind": "database", "connection_url": url})
    out = mesh.update_source("crm", {"description": "renamed"})
    assert out["description"] == "renamed" and store.get("crm", "connection_url") == url
    monkeypatch.setenv("NEW_URL", url)
    out = mesh.update_source("crm", {"connection_url_env": "NEW_URL"})
    assert out["credentials"]["store"] == "env" and store.get("crm", "connection_url") is None


def test_remove_retires_catalog_glossary_and_graph(mesh, store, tmp_path):
    mesh.add_source({"id": "crm", "kind": "database", "connection_url": sqlite_url(tmp_path)})
    mesh.catalog_source("crm", wait_seconds=120)
    assert mesh.glossary_index.stats()["embedded_nodes"] > 0 and mesh.graph.stats()["keys"] > 0
    out = mesh.remove_source("crm")
    assert out["retired_datasets"] == 4
    assert mesh.glossary_index.stats()["embedded_nodes"] == 0
    assert mesh.graph.stats()["keys"] == 0 and mesh.graph.stats()["terms"] == 0
    assert store.get("crm", "connection_url") is None
    assert not any(s["source_id"] == "crm" for s in mesh.list_structured_sources())


def test_config_file_sources_are_read_only(mesh_factory, config_factory, sqlite_source):
    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source]}))
    src = next(s for s in mesh.settings_overview()["sources"] if s["source_id"] == "crm")
    assert src["origin"] == "config_file" and src["editable"] is False
    with pytest.raises(Forbidden):
        mesh.remove_source("crm")
    with pytest.raises(Forbidden):
        mesh.update_source("crm", {"description": "x"})


def test_sources_persist_across_restart(mesh_factory, config_factory, store, tmp_path):
    cfg = config_factory()
    first = mesh_factory(cfg, secret_store=store)
    first.add_source({"id": "crm", "kind": "database", "connection_url": sqlite_url(tmp_path)})
    first.close()
    again = mesh_factory(cfg, secret_store=store)
    assert again.test_source(source_id="crm")["ok"]


def test_connection_test_never_leaks_secrets(mesh, tmp_path):
    r = mesh.test_source({"id": "t", "kind": "database",
                          "connection_url": f"sqlite:///{tmp_path}/missing/{SECRET}.db"})
    assert r["ok"] is False and SECRET not in json.dumps(r)
    assert mask_secrets("postgresql://bob:hunter22@db/x failed") == "postgresql://bob:***@db/x failed"
    assert mask_secrets(f"token {SECRET} rejected", [SECRET]) == "token *** rejected"


def test_settings_api_round_trip_and_local_only_guard(mesh, tmp_path):
    url = f"postgresql+psycopg://bob:{SECRET}@db.internal:5432/crm"
    with TestClient(create_app(mesh)) as c:
        r = c.post("/api/settings/sources", json={"id": "pg", "kind": "database", "connection_url": url})
        assert r.status_code == 201, r.text
        assert SECRET not in r.text and r.json()["target"] == "postgresql+psycopg://bob:***@db.internal:5432/crm"
        overview = c.get("/api/settings")
        assert SECRET not in overview.text and overview.json()["secret_store"]["available"]
        stored = json.dumps(mesh.db.query("SELECT settings_json FROM source_configs"))
        assert SECRET not in stored and "bob:***@" in stored
        t = c.post("/api/settings/sources/test", json={"id": "x", "kind": "parquet", "path": "/etc"})
        assert t.json()["ok"] is False and t.json()["error"]["code"] == "forbidden"
        assert c.delete("/api/settings/sources/pg").status_code == 200
    with TestClient(create_app(mesh), client=("192.168.1.77", 5000)) as remote:
        r = remote.post("/api/settings/sources", json={"id": "pg2", "kind": "database", "connection_url_env": "X"})
        assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
        assert remote.get("/api/settings").status_code == 200  # reading is allowed


def test_settings_page_assets(mesh):
    with TestClient(create_app(mesh)) as c:
        assert 'href="#settings"' in c.get("/").text
        js = c.get("/static/app.js").text
        assert "renderSettings" in js and "/api/settings/sources/test" in js


# -- database described by fields (type, database name, hostname, port, schema) -------------------
def test_build_url_escapes_password_and_uses_default_ports():
    from knowledge_mesh.structured.connectors.database import build_url

    url = build_url("postgresql", "db.example.com", None, "crm", "bob", "p@ss:w/rd")
    assert url.render_as_string(hide_password=False) == "postgresql+psycopg://bob:p%40ss%3Aw%2Frd@db.example.com:5432/crm"
    assert build_url("mysql", "h", None, "d", None, None).port == 3306
    assert build_url("mssql", "h", 1500, "d", "u", None).render_as_string() == "mssql+pymssql://u@h:1500/d"
    oracle = build_url("oracle", "h", None, "ORCLPDB1", "u", "x")
    assert oracle.port == 1521 and oracle.query["service_name"] == "ORCLPDB1" and oracle.database is None


def test_add_database_by_fields_stores_only_password_secret(mesh, store):
    out = mesh.add_source({"id": "pg", "kind": "database", "db_type": "postgresql", "database": "crm",
                           "host": "db.example.com", "port": 6543, "schemas": ["sales"], "username": "bob",
                           "password": SECRET})
    assert out["target"] == "postgresql+psycopg://bob:***@db.example.com:6543/crm"
    assert out["credentials"]["status"] == "present" and out["credentials"]["reference"].endswith("pg:password")
    assert store.get("pg", "password") == SECRET and store.get("pg", "connection_url") is None
    stored = mesh.db.query_one("SELECT settings_json FROM source_configs WHERE source_id='pg'")["settings_json"]
    assert SECRET not in stored and '"schemas": ["sales"]' in stored and '"port": 6543' in stored
    assert out["settings"]["db_type"] == "postgresql" and out["settings"]["schemas"] == ["sales"]


def test_database_fields_validation(mesh):
    with pytest.raises(InvalidArgument) as err:
        mesh.add_source({"id": "x", "kind": "database", "db_type": "postgresql", "database": "crm"})
    assert err.value.message == "Required: Hostname"
    with pytest.raises(InvalidArgument) as err:
        mesh.add_source({"id": "x", "kind": "database"})
    assert err.value.details["fields"] == ["connection_url", "connection_url_env"]  # nothing given: URL mode
    with pytest.raises(InvalidArgument):
        mesh.add_source({"id": "x", "kind": "database", "db_type": "db2", "host": "h", "database": "d"})
    no_pw = mesh.add_source({"id": "trust", "kind": "database", "db_type": "mysql", "host": "h", "database": "d"})
    assert no_pw["credentials"]["status"] == "no_password" and no_pw["target"] == "mysql+pymysql://h:3306/d"


def test_connection_test_by_fields_reports_failure_without_password(mesh):
    r = mesh.test_source({"id": "t", "kind": "database", "db_type": "postgresql", "host": "127.0.0.1",
                          "port": 1, "database": "crm", "username": "bob", "password": SECRET})
    assert r["ok"] is False and SECRET not in json.dumps(r)
    assert "connection failed" in r["error"]["message"]


def test_missing_driver_message_names_the_package(mesh, monkeypatch):
    from knowledge_mesh.structured.connectors import database

    monkeypatch.setitem(database.DB_TYPES, "postgresql",
                        {**database.DB_TYPES["postgresql"], "driver": "postgresql+nosuchdriver"})
    r = mesh.test_source({"id": "t", "kind": "database", "db_type": "postgresql", "host": "h", "database": "d"})
    assert r["ok"] is False and "driver is not installed" in r["error"]["message"]
    assert "pip install 'psycopg[binary]'" in r["error"]["message"]


def test_switching_between_url_and_fields_drops_the_unused_secret(mesh, store, tmp_path):
    mesh.add_source({"id": "crm", "kind": "database", "connection_url": sqlite_url(tmp_path)})
    out = mesh.update_source("crm", {"db_type": "postgresql", "host": "db", "database": "crm", "username": "bob",
                                      "password": "pw1"})
    assert store.get("crm", "connection_url") is None and store.get("crm", "password") == "pw1"
    assert out["target"] == "postgresql+psycopg://bob:***@db:5432/crm"
    kept = mesh.update_source("crm", {"db_type": "postgresql", "host": "db2", "database": "crm", "username": "bob"})
    assert store.get("crm", "password") == "pw1" and "db2" in kept["target"]  # blank password keeps the stored one
    with pytest.raises(InvalidArgument):
        mesh.update_source("crm", {"description": "back to a URL without giving one"})


def test_settings_overview_lists_database_types(mesh):
    types = mesh.settings_overview()["database_types"]
    assert types["postgresql"] == {"label": "PostgreSQL", "port": 5432} and set(types) == {"postgresql", "mysql", "mssql", "oracle"}


# -- catalogs that find nothing must say so --------------------------------------------------------
def test_empty_database_is_reported_not_silent(mesh, tmp_path):
    import sqlite3

    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()
    mesh.add_source({"id": "empty", "kind": "database", "connection_url": f"sqlite:///{empty}"})
    test = mesh.test_source(source_id="empty")
    assert test["ok"] and test["dataset_count"] == 0 and "No tables or views found" in test["warning"]
    job = mesh.catalog_source("empty", wait_seconds=60)
    assert job["status"] in {"partial", "failed"}
    failed = mesh.job_items(job["job_id"], status="failed")["items"]
    assert failed and failed[0]["ref"] == "empty" and "No tables or views found in schema 'main'" in failed[0]["reason"]
    src = next(s for s in mesh.catalog_tree() if s["source_id"] == "empty")
    assert src["status"] == "empty" and "No tables or views found" in src["status_reason"]


class _StubConnector:
    def __init__(self, scanned, overview):
        self._scanned, self._overview = scanned, overview

    def scanned_schemas(self):
        return self._scanned

    def schema_overview(self):
        return self._overview


def test_coverage_note_names_schemas_with_tables():
    from types import SimpleNamespace

    from knowledge_mesh.structured.connectors.base import coverage_note

    s = SimpleNamespace(id="loans", database="loans", datasets=None, schemas=None)
    msg = coverage_note(s, _StubConnector(["public"], [("investment", 16), ("public", 10)]), 0)
    assert "No tables or views found in schema 'public' of database 'loans'" in msg and "investment (16)" in msg
    assert "public (10)" not in msg
    nothing = coverage_note(s, _StubConnector(["public"], []), 0)
    assert "check the database name" in nothing
    unscanned = coverage_note(s, _StubConnector(["public"], [("investment", 16), ("public", 10)]), 10)
    assert unscanned.startswith("Schemas with tables that were not scanned: investment (16)")
    s.schemas = ["public"]  # explicitly chosen: no nagging about other schemas
    assert coverage_note(s, _StubConnector(["public"], [("investment", 16), ("public", 10)]), 10) is None


# -- required fields are named, schema empty = all schemas ------------------------------------------
def test_required_fields_are_named(mesh, tmp_path):
    with pytest.raises(InvalidArgument) as err:
        mesh.add_source({"id": "", "kind": "database", "db_type": "postgresql", "database": "crm", "host": "h"})
    assert err.value.message == "Required: Source ID" and err.value.details["fields"] == ["id"]
    with pytest.raises(InvalidArgument) as err:
        mesh.add_source({"id": "pg", "kind": "database", "db_type": "postgresql"})
    assert err.value.message == "Required: Database name, Hostname"
    assert err.value.details["fields"] == ["database", "host"]
    with pytest.raises(InvalidArgument) as err:
        mesh.add_source({"id": "dbx", "kind": "databricks"})
    assert set(err.value.details["fields"]) >= {"server_hostname", "http_path", "catalog", "token"}
    with pytest.raises(InvalidArgument) as err:
        mesh.add_source({"id": "bad id!", "kind": "parquet", "path": "x"})
    assert err.value.details["fields"] == ["id"]
    mesh.add_source({"id": "pg", "kind": "database", "db_type": "postgresql", "database": "crm", "host": "h"})
    with pytest.raises(Conflict) as err:
        mesh.add_source({"id": "pg", "kind": "database", "db_type": "postgresql", "database": "crm", "host": "h"})
    assert err.value.details["fields"] == ["id"]
    r = mesh.test_source({"kind": "database", "db_type": "postgresql", "database": "crm"})
    assert r["ok"] is False and r["error"]["fields"] == ["host"]


def test_api_errors_carry_field_names(mesh):
    with TestClient(create_app(mesh)) as c:
        r = c.post("/api/settings/sources", json={"id": "", "kind": "parquet", "schemas": []})
        assert r.status_code == 400 and r.json()["error"]["details"]["fields"] == ["id", "path"]
        r = c.post("/api/settings/sources", json={"id": "x", "kind": "database", "port": "abc"})
        assert r.status_code == 422 and r.json()["error"]["details"]["fields"] == ["port"]
        assert r.json()["error"]["message"].startswith("port:")


class _FakeInspector:
    default_schema_name = "public"

    def __init__(self, tables):
        self.tables = tables

    def get_schema_names(self):
        return list(self.tables) + ["information_schema", "pg_catalog", "pg_toast_temp_1"]

    def get_table_names(self, schema=None):
        if self.tables.get(schema) == "denied":
            from sqlalchemy.exc import ProgrammingError

            raise ProgrammingError("select", {}, Exception("permission denied"))
        return self.tables.get(schema, [])

    def get_view_names(self, schema=None):
        return []


def _connector_with(tables, backend="postgresql"):
    from types import SimpleNamespace

    from sqlalchemy.engine import make_url

    from knowledge_mesh.config import StructuredSourceSettings
    from knowledge_mesh.structured.connectors.database import DatabaseConnector

    conn = DatabaseConnector.__new__(DatabaseConnector)
    conn.settings = StructuredSourceSettings(id="t", kind="database", db_type="postgresql", host="h", database="d")
    conn.source_id, conn.secrets, conn._all_schemas = "t", None, None
    conn._inspector = _FakeInspector(tables)
    conn.engine = SimpleNamespace(url=make_url(f"{backend}://h/d"))
    return conn


def test_empty_schema_scans_all_non_system_schemas():
    conn = _connector_with({"public": ["loan"], "investment": ["fund", "trade"], "locked": "denied"})
    names = sorted(r.qualified for r in conn.list_datasets())
    assert names == ["investment.fund", "investment.trade", "public.loan"]  # system schemas and the denied one skipped
    assert set(conn.scanned_schemas()) == {"public", "investment", "locked"}
    mysql = _connector_with({"public": ["loan"], "other_db": ["x"]}, backend="mysql+pymysql")
    assert mysql.scanned_schemas() == ["public"]  # MySQL: schema == database, keep the connected one
    conn.settings.schemas = ["investment"]
    conn._all_schemas = None
    assert [r.qualified for r in conn.list_datasets()] == ["investment.fund", "investment.trade"]

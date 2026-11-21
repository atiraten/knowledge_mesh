from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from knowledge_mesh.common.secrets import MemorySecretStore
from knowledge_mesh.config import _deep_merge, config_from_dict
from knowledge_mesh.llm.client import FakeLLM
from knowledge_mesh.mesh import KnowledgeMesh

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_factory(tmp_path):
    def make(overrides: dict | None = None):
        data = {
            "storage": {"originals_dir": "data/originals", "extracted_dir": "data/extracted",
                        "metadata_db": "data/metadata.sqlite", "allowed_ingest_roots": ["data/inbox"]},
            "vector_store": {"document_persist_dir": "data/index/documents",
                             "glossary_persist_dir": "data/index/glossary"},
            "knowledge_graph": {"data_path": "data/knowledge_graph.trig",
                                "shapes_path": str(REPO / "config" / "knowledge_graph_shapes.ttl")},
            "embedding": {"provider": "hashing"},
            "observability": {"enabled": True, "phoenix_enabled": False, "loki_enabled": False,
                              "prometheus_enabled": True, "log_format": "text"},
            "term_generation": {"prompt_path": str(REPO / "prompts" / "term_generation.txt")},
            "crawler": {"requests_per_second_per_host": 10000, "use_sitemap": False},
            "retrieval": {"min_document_similarity": 0.15, "min_glossary_similarity": 0.15},
            "api": {"api_key_env": None},
        }
        return config_from_dict(_deep_merge(data, overrides or {}), tmp_path)
    return make


@pytest.fixture
def mesh_factory(config_factory):
    created: list[KnowledgeMesh] = []

    def make(config=None, **kwargs) -> KnowledgeMesh:
        kwargs.setdefault("secret_store", MemorySecretStore())  # never touch the real OS keychain in tests
        # never post annotations to a real Phoenix in tests
        kwargs.setdefault("phoenix_transport", httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})))
        mesh = KnowledgeMesh(config or config_factory(), **kwargs)
        created.append(mesh)
        return mesh

    yield make
    for m in created:
        m.close()


@pytest.fixture
def inbox(mesh_factory):
    def _inbox(mesh: KnowledgeMesh) -> Path:
        return mesh.config.storage.allowed_ingest_roots[0]
    return _inbox


# -- structured fixtures ---------------------------------------------------------------------------
CUSTOMERS = [(f"C{i:03d}", f"user{i}@example.com", ["Ann", "Bob", "Cid", "Dee"][i % 4], ["N", "S", "E", "W"][i % 4])
             for i in range(1, 13)]
ORDERS = [(1000 + i, CUSTOMERS[i % 9][0], f"2024-01-{(i % 28) + 1:02d}", 10.5 * i) for i in range(1, 25)]
STORES = [(c, code, f"{c}-{code}-store") for c in ("US", "FR", "DE") for code in ("01", "02", "03")]
SALES = [(5001 + i, STORES[i % 9][0], STORES[i % 9][1], CUSTOMERS[(i * 5) % 12][0], (i % 4) + 1) for i in range(30)]


def build_sqlite(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (customer_id TEXT PRIMARY KEY, email TEXT UNIQUE, first_name TEXT, region TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id),
                             order_date TEXT, amount REAL);
        CREATE TABLE stores (country TEXT, store_code TEXT, store_name TEXT);
        CREATE TABLE sales (sale_id INTEGER, ctry TEXT, store TEXT, cust TEXT, qty INTEGER);
    """)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    conn.executemany("INSERT INTO stores VALUES (?,?,?)", STORES)
    conn.executemany("INSERT INTO sales VALUES (?,?,?,?,?)", SALES)
    conn.commit()
    conn.close()
    return path


def build_parquet(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"cust_id": [c[0] for c in CUSTOMERS], "fst_nm": [c[2] for c in CUSTOMERS],
                             "email_adr": [c[1] for c in CUSTOMERS]}), path / "cust.parquet")
    return path


@pytest.fixture
def sqlite_source(tmp_path, monkeypatch):
    db = build_sqlite(tmp_path / "crm.sqlite")
    monkeypatch.setenv("TEST_CRM_URL", f"sqlite:///{db}")
    return {"id": "crm", "kind": "database", "connection_url_env": "TEST_CRM_URL"}


@pytest.fixture
def parquet_source(tmp_path):
    folder = build_parquet(tmp_path / "parquet")
    return {"id": "export", "kind": "parquet", "path": str(folder)}


TERMS = {
    "customer_id": ("Customer Identifier", "The unique identifier assigned to each customer"),
    "email": ("Customer Email Address", "The email address of the customer"),
    "first_name": ("Customer First Name", "The given name of the customer"),
    "region": ("Customer Region Code", "The sales region of the customer"),
    "order_id": ("Order Identifier", "The unique identifier of the order"),
    "order_date": ("Order Date", "The calendar date the order was placed"),
    "amount": ("Order Amount", "The monetary amount of the order"),
    "cust_id": ("Customer Identifier", "The unique identifier assigned to each customer"),
    "fst_nm": ("Customer First Name", "The given name of the customer"),
    "email_adr": ("Customer Email Address", "The email address of the customer"),
    "country": ("Store Country Code", "Country of the store"),
    "store_code": ("Store Code", "Store code within the country"),
    "store_name": ("Store Name", "Name of the store"),
    "sale_id": ("Sale Identifier", "Identifier of the sale"),
    "ctry": ("Country Code", "Country of the sale"),
    "store": ("Store Code", "Store of the sale"),
    "cust": ("Customer Identifier", "Customer of the sale"),
    "qty": ("Sale Quantity", "Units sold"),
}
DECODED = {"customers": "Customer", "orders": "Order", "cust": "Customer", "stores": "Store", "sales": "Sale"}


def fake_term_responder(prompt: str, system, purpose: str) -> str:
    if purpose != "term_generation":
        return "{}"
    table = re.search(r"^Table: (.+)$", prompt, re.M).group(1).strip()
    columns = [c.strip() for c in re.search(r"^Columns: (.+)$", prompt, re.M).group(1).split(",")]
    lines = ["```csv", "<table name>,<column name>,<business term>,<detailed business description>,<comments>"]
    for col in columns:
        term, desc = TERMS.get(col, (col.replace("_", " ").title(), f"Value of {col}"))
        lines.append(f"{DECODED.get(table, table)},{col},{term},{desc},{col}={term.lower()}")
    lines.append("```")
    return "\n".join(lines)


@pytest.fixture
def fake_term_llm():
    return FakeLLM(fake_term_responder)

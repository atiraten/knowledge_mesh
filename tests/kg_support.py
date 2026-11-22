"""Shared fixtures for the knowledge graph / GraphRAG / SQL tests: a small shop database, a fake term model
and a fake relationship model (the term-generation model also answers the relationship prompt)."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from knowledge_mesh.common.ids import dataset_id_for
from knowledge_mesh.llm.client import FakeLLM
from tests.conftest import REPO, prompt_columns, prompt_table

SHOP_TABLES = ("customers", "orders", "products", "order_lines", "shipments", "support_tickets", "trades")
CUSTOMERS = [("C1", "Ann Lee", "1985-03-02"), ("C2", "Bob Ray", "1992-07-15"), ("C3", "Cid Moe", "1979-11-30"),
             ("C4", "Dee Fox", "1995-01-09")]
PRODUCTS = [("P1", "Phone"), ("P2", "Laptop"), ("P3", "Cable")]
ORDERS = [(1, "C1", "2024-01-05", 120.0), (2, "C2", "2024-02-11", 80.0), (3, "C1", "2024-03-20", 15.5)]
ORDER_LINES = [(1, 1, "P1", 1), (1, 2, "P3", 2), (2, 1, "P2", 1), (3, 1, "P3", 1)]
SHIPMENTS = [(10, 1, 1, "2024-01-06"), (11, 1, 2, "2024-01-06"), (12, 2, 1, "2024-02-12")]
TICKETS = [(100, "C2", "P2", "2024-02-20"), (101, "C3", "P1", "2024-03-01")]
TRADES = [(500, "2024-04-01", "2024-04-03", "C1"), (501, "2024-04-02", "2024-04-04", "C3")]

TERMS = {
    "customer_id": ("Customer Identifier", "The unique identifier of a customer"),
    "full_name": ("Customer Full Name", "The full name of the customer"),
    "birth_date": ("Birth Date", "The date the customer was born"),
    "order_id": ("Order Identifier", "The unique identifier of an order"),
    "order_date": ("Order Date", "The date the order was placed"),
    "amount": ("Order Amount", "The monetary amount of the order"),
    "product_id": ("Product Identifier", "The unique identifier of a product"),
    "product_name": ("Product Name", "The name of the product"),
    "line_no": ("Order Line Number", "The position of the line within its order"),
    "qty": ("Order Line Quantity", "Units of the product on the order line"),
    "shipment_id": ("Shipment Identifier", "The unique identifier of a shipment"),
    "shipped_on": ("Shipment Date", "The date the order line was shipped"),
    "ticket_id": ("Support Ticket Identifier", "The unique identifier of a support ticket"),
    "opened_on": ("Ticket Opened Date", "The date the support ticket was opened"),
    "trade_id": ("Trade Identifier", "The unique identifier of a trade"),
    "transaction_date": ("Transaction Date", "The date the trade was executed"),
    "settlement_date": ("Settlement Date", "The date the trade settled"),
}
SENSITIVE = {"full_name", "birth_date"}  # terms that can identify a person (the model answers Y)
ENTITIES = {"customers": "Customer", "orders": "Order", "products": "Product", "order_lines": "Order Line",
            "shipments": "Shipment", "support_tickets": "Support Ticket", "trades": "Trade"}
# (referencing table, referenced table) -> proposal
RELATIONSHIPS = {
    ("orders", "customers"): {"source_entity": "Customer", "name": "placesOrder", "target_entity": "Order",
                              "definition": "A customer places an order to buy products.",
                              "synonyms": ["buys", "bought", "purchases", "purchased"], "inverse_name": "placedBy",
                              "cardinality": "one-to-many"},
    ("order_lines", "orders"): {"source_entity": "Order", "name": "containsOrderLine", "target_entity": "Order Line",
                                "definition": "An order contains one line per product ordered.",
                                "synonyms": ["has line"], "inverse_name": "partOfOrder", "cardinality": "one-to-many"},
    ("order_lines", "products"): {"source_entity": "Order Line", "name": "referencesProduct",
                                  "target_entity": "Product", "definition": "An order line is for one product.",
                                  "synonyms": ["for product"], "inverse_name": None, "cardinality": "many-to-one"},
    ("shipments", "order_lines"): {"source_entity": "Shipment", "name": "shipsOrderLine",
                                   "target_entity": "Order Line", "definition": "A shipment delivers one order line.",
                                   "synonyms": ["delivers"], "inverse_name": None, "cardinality": "many-to-one"},
    ("support_tickets", "customers"): {"source_entity": "Customer", "name": "opensTicket",
                                       "target_entity": "Support Ticket",
                                       "definition": "A customer opens a support ticket to report a problem.",
                                       "synonyms": ["complains", "reports issue"], "inverse_name": "openedBy",
                                       "cardinality": "one-to-many"},
    ("support_tickets", "products"): {"source_entity": "Support Ticket", "name": "concernsProduct",
                                      "target_entity": "Product", "definition": "A support ticket is about a product.",
                                      "synonyms": ["about product"], "inverse_name": None,
                                      "cardinality": "many-to-one"},
    ("trades", "customers"): {"source_entity": "Customer", "name": "executesTrade", "target_entity": "Trade",
                              "definition": "A customer executes trades.", "synonyms": ["trades"],
                              "inverse_name": "executedBy", "cardinality": "one-to-many"},
}
TERM_RELATIONS = {"trades": [{"subject": "Settlement Date", "predicate": "expectedAfter", "object": "Transaction Date",
                              "definition": "A trade settles on or after the day it is executed."}]}
SYNONYMS = {"customers": [{"term": "Birth Date", "synonyms": ["date of birth", "DOB"]}]}


def build_shop(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (customer_id TEXT PRIMARY KEY, full_name TEXT, birth_date TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id),
                             order_date TEXT, amount REAL);
        CREATE TABLE products (product_id TEXT PRIMARY KEY, product_name TEXT);
        CREATE TABLE order_lines (order_id INTEGER REFERENCES orders(order_id), line_no INTEGER,
                                  product_id TEXT REFERENCES products(product_id), qty INTEGER,
                                  PRIMARY KEY (order_id, line_no));
        CREATE TABLE shipments (shipment_id INTEGER PRIMARY KEY, order_id INTEGER, line_no INTEGER, shipped_on TEXT,
                                FOREIGN KEY (order_id, line_no) REFERENCES order_lines(order_id, line_no));
        CREATE TABLE support_tickets (ticket_id INTEGER PRIMARY KEY,
                                      customer_id TEXT REFERENCES customers(customer_id),
                                      product_id TEXT REFERENCES products(product_id), opened_on TEXT);
        CREATE TABLE trades (trade_id INTEGER PRIMARY KEY, transaction_date TEXT, settlement_date TEXT,
                             customer_id TEXT REFERENCES customers(customer_id));
    """)
    for table, rows in (("customers", CUSTOMERS), ("products", PRODUCTS), ("orders", ORDERS),
                        ("order_lines", ORDER_LINES), ("shipments", SHIPMENTS), ("support_tickets", TICKETS),
                        ("trades", TRADES)):
        conn.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)
    conn.commit()
    conn.close()
    return path


def shop_responder(prompt: str, system, purpose: str) -> str:
    if purpose == "term_generation":
        table = prompt_table(prompt)
        columns = prompt_columns(prompt)
        lines = ["```csv",
                 "<table name>,<column name>,<business term>,<sensitive>,<detailed business description>,<comments>"]
        for col in columns:
            term, desc = TERMS.get(col, (col.replace("_", " ").title(), f"Value of {col}"))
            flag = "Y" if col in SENSITIVE else "N"
            lines.append(f"{ENTITIES.get(table, table)},{col},{term},{flag},{desc},{col}={term.lower()}")
        lines.append("```")
        return "\n".join(lines)
    if purpose == "relationship_generation":
        table = prompt_table(prompt)
        rels = []
        for ref, ref_table in re.findall(r"^- (FK\d+): \w+\([^)]*\) -> (\w+)\(", prompt, re.M):
            proposal = RELATIONSHIPS.get((table, ref_table))
            if proposal:
                rels.append({"fk": ref, **proposal})
        return json.dumps({"relationships": rels, "term_relations": TERM_RELATIONS.get(table, []),
                           "term_synonyms": SYNONYMS.get(table, [])})
    return "{}"


def shop_ids(source_id: str = "shop") -> dict[str, str]:
    return {t: dataset_id_for(source_id, "main", t) for t in SHOP_TABLES}


@pytest.fixture
def shop_source(tmp_path, monkeypatch):
    db = build_shop(tmp_path / "shop.sqlite")
    monkeypatch.setenv("TEST_SHOP_URL", f"sqlite:///{db}")
    return {"id": "shop", "kind": "database", "connection_url_env": "TEST_SHOP_URL"}


KG_OVERRIDES = {
    "knowledge_graph": {"ontology_path": str(REPO / "config" / "knowledge_graph_ontology.ttl")},
    "relationship_generation": {"prompt_path": str(REPO / "prompts" / "relationship_generation.txt")},
}


@pytest.fixture
def shop_mesh(mesh_factory, config_factory, shop_source):
    """A cataloged shop source: terms, keys, relationship proposals and the layered graph."""
    from knowledge_mesh.config import _deep_merge

    def make(extra: dict | None = None, catalog: bool = True, **kwargs):
        overrides = _deep_merge({"structured_sources": [shop_source], **KG_OVERRIDES}, extra or {})
        mesh = mesh_factory(config_factory(overrides), term_llm=FakeLLM(shop_responder), **kwargs)
        if catalog:
            job = mesh.catalog_source("shop", wait_seconds=180)
            assert job["status"] == "succeeded", job
        return mesh
    return make

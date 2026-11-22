"""Column list of the term generation prompt: one column per line, ``name (TYPE) ["v1" "v2"]`` with the
cataloged data type and up to 5 most common profiled values; no values for a column whose current business
term is flagged sensitive."""

import re

from knowledge_mesh.llm.client import FakeLLM
from knowledge_mesh.structured.terms import column_line, render_prompt
from tests.conftest import REPO, fake_term_responder, prompt_columns, prompt_table


def test_column_line_format():
    assert column_line("city", "VARCHAR(20)", ["New York", "washington"]) == 'city (VARCHAR(20)) ["New York" "washington"]'
    assert column_line("ssn_hash", "VARCHAR(256)", []) == "ssn_hash (VARCHAR(256))"  # values are optional
    assert column_line("code") == "code"
    many = column_line("c", "TEXT", ["", None, 'say "hi"', "back\\slash", "x" * 80, "a|b\nc", "5th", "6th"])
    assert many == 'c (TEXT) ["say \\"hi\\"" "back\\\\slash" "' + "x" * 60 + '" "a b c" "5th"]'  # 5 values, 60 chars


def test_prompt_lists_one_column_per_line():
    from knowledge_mesh.config import TermGenerationSettings

    template = (REPO / "prompts" / "term_generation.txt").read_text()
    details = {"city": column_line("city", "VARCHAR(20)", ["New York", "washington"]), "zip": column_line("zip", "CHAR(5)")}
    prompt = render_prompt(template, "customer", ["city", "zip"], TermGenerationSettings(), {}, details)
    assert re.search(r'Columns:\**[ \t]*\ncity \(VARCHAR\(20\)\) \["New York" "washington"\]\nzip \(CHAR\(5\)\)\n', prompt)
    assert prompt_columns(prompt) == ["city", "zip"] and prompt_table(prompt) == "customer"


def test_generation_prompt_uses_types_and_profiled_values_except_for_sensitive_terms(
        mesh_factory, config_factory, sqlite_source):
    llm = FakeLLM(fake_term_responder)
    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source]}), term_llm=llm)
    mesh.catalog_source("crm", wait_seconds=120)  # terms run before profiling: data types only the first time
    first = next(c["prompt"] for c in llm.calls if c["purpose"] == "term_generation" and prompt_table(c["prompt"]) == "customers")
    assert re.search(r"^first_name \(TEXT\)$", first, re.M)

    ds = mesh.db.query_one("SELECT dataset_id FROM datasets WHERE name = 'customers'")["dataset_id"]
    mesh.db.execute("UPDATE business_terms SET sensitive = 1 WHERE slug ="
                    " (SELECT term_slug FROM columns WHERE dataset_id = ? AND name = 'email')", (ds,))
    llm.calls.clear()
    mesh.generate_terms([ds], force=True, wait_seconds=60)
    prompt = next(c["prompt"] for c in llm.calls if c["purpose"] == "term_generation")
    assert prompt_columns(prompt) == ["customer_id", "email", "first_name", "region"]
    assert re.search(r'^first_name \(TEXT\) \["Ann" "Bob" "Cid" "Dee"\]$', prompt, re.M)  # most common values
    assert re.search(r'^region \(TEXT\) \["E" "N" "S" "W"\]$', prompt, re.M)
    assert re.search(r"^email \(TEXT\)$", prompt, re.M)  # flagged sensitive: no row values
    assert "@example.com" not in prompt

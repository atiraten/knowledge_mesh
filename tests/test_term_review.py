"""Review and correction of generated business terms (prompts/term_correction.txt): the term model reviews
its own output (column|data_type|business term|description|decoding rows), rejected columns go back to the
generation prompt with the reviewer's reason (only those columns), at most max_correction_rounds times, then
are left blank (term_status 'rejected', the last reason kept)."""

import re

import pytest

from knowledge_mesh.llm.client import FakeLLM
from knowledge_mesh.structured.terms import NO_VERDICT, parse_review_output
from tests.conftest import REPO, prompt_columns, prompt_table

REVIEW_PROMPT = str(REPO / "prompts" / "term_correction.txt")
PLACEHOLDERS = ("{industry_name}", "{language_name}", "{class_words}", "{abbreviations}", "{table_name}", "{rows}")


class ReviewScript:
    """Fake term model. ``bad`` columns get the one-word term "Date" (rejected by the fake reviewer) until they
    have been sent for correction ``fix_after`` times; ``silent`` columns get no verdict in the first review."""

    def __init__(self, bad=("order_date",), fix_after=1, silent=()):
        self.bad, self.fix_after, self.silent = set(bad), fix_after, set(silent)
        self.calls: list[tuple[str, str, str]] = []  # (purpose, table, prompt)
        self.corrections: dict[str, int] = {}
        self.reviews: dict[str, int] = {}  # per table

    def __call__(self, prompt, system, purpose):
        table = prompt_table(prompt)
        self.calls.append((purpose, table, prompt))
        if purpose in ("term_generation", "term_correction"):
            columns = prompt_columns(prompt)
            lines = ["<table name>,<column name>,<business term>,<sensitive>,<detailed business description>,<comments>"]
            for col in columns:
                if purpose == "term_correction":
                    self.corrections[col] = self.corrections.get(col, 0) + 1
                fixed = self.corrections.get(col, 0) >= self.fix_after
                term = "Date" if col in self.bad and not fixed else f"{table.title()} {col.replace('_', ' ').title()} Value"
                lines.append(f"{table.title()},{col},{term},N,The {col} of the {table},{col}={col}")
            return "\n".join(lines)
        if purpose == "term_review":
            self.reviews[table] = self.reviews.get(table, 0) + 1
            rows = prompt.split("column|data_type|business term|description|decoding\n", 1)[1].split("\n\n", 1)[0]
            out = ["column|status|reject reason (blank is approved)"]
            for row in rows.splitlines():
                col, dtype, term = row.split("|")[:3]
                if col in self.silent and self.reviews[table] == 1:
                    continue
                out.append(f"{col}|REJECT|single word, class word missing" if len(term.split()) < 2 else f"{col}|APPROVE|")
            return "\n".join(out)
        return "{}"

    def for_table(self, table):
        return [(p, prompt) for p, t, prompt in self.calls if t == table]


@pytest.fixture
def review_mesh(mesh_factory, config_factory, sqlite_source):
    def make(script, **term_settings):
        settings = {"correction_prompt_path": REVIEW_PROMPT, **term_settings}
        mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source], "term_generation": settings}),
                            term_llm=FakeLLM(script))
        job = mesh.catalog_source("crm", wait_seconds=120)
        assert job["status"] == "succeeded", job
        return mesh
    return make


def orders_columns(mesh):
    ds = mesh.db.query_one("SELECT dataset_id FROM datasets WHERE name = 'orders'")["dataset_id"]
    return ds, {c["name"]: c for c in mesh.catalog.active_columns(ds)}  # terms from the business glossary


def test_rejected_term_is_corrected(review_mesh):
    script = ReviewScript(fix_after=1)
    mesh = review_mesh(script)
    calls = script.for_table("orders")
    assert [p for p, _ in calls] == ["term_generation", "term_review", "term_correction", "term_review"]
    first_review = calls[1][1]
    assert not any(ph in first_review for ph in PLACEHOLDERS)
    assert prompt_table(first_review) == "orders" and "Identifier, Code, Name" in first_review
    assert "\norder_date|TEXT|Date|The order_date of the orders|order_date=order_date\n" in first_review
    assert "\namount|REAL|Orders Amount Value|" in first_review
    correction = calls[2][1]
    assert prompt_columns(correction) == ["order_date"]  # only the rejected column
    assert "### CORRECTIONS REQUIRED" in correction
    assert "order_date|previous term: Date|reason: single word, class word missing" in correction
    second_review = calls[3][1]
    assert "\norder_date|TEXT|Orders Order Date Value|" in second_review and "amount|REAL" not in second_review
    _, cols = orders_columns(mesh)
    assert cols["order_date"]["term_status"] == "generated" and cols["order_date"]["term_label"] == "Orders Order Date Value"
    assert cols["amount"]["term_label"] == "Orders Amount Value"
    run = mesh.db.query_one("SELECT * FROM term_generation_runs WHERE dataset_id = ?", (cols["amount"]["dataset_id"],))
    assert run["status"] == "succeeded" and run["failed"] == 0
    assert run["error"] == "review 1: order_date rejected: single word, class word missing"  # audit trail


def test_still_rejected_after_three_corrections_is_left_blank(review_mesh):
    script = ReviewScript(fix_after=99)
    mesh = review_mesh(script)
    calls = script.for_table("orders")
    assert [p for p, _ in calls] == ["term_generation", "term_review"] + ["term_correction", "term_review"] * 3
    ds, cols = orders_columns(mesh)
    blank = cols["order_date"]
    assert blank["term_status"] == "rejected" and blank["term_label"] is None and blank["term_slug"] is None
    assert blank["term_description"] is None and blank["term_sensitive"] is None
    assert blank["term_error"] == "rejected by review: single word, class word missing"
    view = {c["name"]: c for c in mesh.get_dataset(ds)["columns"]}
    assert view["order_date"]["business_term"] is None
    assert view["order_date"]["business_term_error"].startswith("rejected by review")
    assert view["amount"]["business_term"]["label"] == "Orders Amount Value"
    assert "order_date" not in {n.metadata["column"] for n in mesh.glossary.nodes_for(ds)}
    run = mesh.db.query_one("SELECT * FROM term_generation_runs WHERE dataset_id = ? ORDER BY created_at DESC", (ds,))
    assert run["status"] == "partial" and run["failed"] == 1
    assert run["error"].startswith("order_date: still rejected after 3 correction round(s), left blank: single word")
    assert [f"review {i}: order_date rejected" in run["error"] for i in (1, 2, 3, 4)] == [True] * 4


def test_missing_verdict_counts_as_rejected(review_mesh):
    script = ReviewScript(bad=(), silent=("amount",))
    mesh = review_mesh(script)
    calls = script.for_table("orders")
    assert [p for p, _ in calls] == ["term_generation", "term_review", "term_correction", "term_review"]
    assert f"amount|previous term: Orders Amount Value|reason: {NO_VERDICT}" in calls[2][1]
    _, cols = orders_columns(mesh)
    assert cols["amount"]["term_status"] == "generated"


def test_zero_rounds_leaves_rejected_columns_blank(review_mesh):
    script = ReviewScript(fix_after=1)
    mesh = review_mesh(script, max_correction_rounds=0)
    assert [p for p, _ in script.for_table("orders")] == ["term_generation", "term_review"]
    _, cols = orders_columns(mesh)
    assert cols["order_date"]["term_status"] == "rejected"


def test_review_is_skipped_without_its_prompt(review_mesh, tmp_path):
    script = ReviewScript(fix_after=1)
    mesh = review_mesh(script, correction_prompt_path=str(tmp_path / "missing.txt"))
    assert "term_review" not in {p for p, _, _ in script.calls}
    ds, cols = orders_columns(mesh)
    assert cols["order_date"]["term_label"] == "Date"  # not reviewed, kept as generated
    run = mesh.db.query_one("SELECT error FROM term_generation_runs WHERE dataset_id = ?", (ds,))
    assert "term review skipped: correction prompt not found" in run["error"]


def test_prompt_version_covers_the_review_prompt(review_mesh, tmp_path):
    with_review = review_mesh(ReviewScript(bad=())).terms.prompt_version
    without = review_mesh(ReviewScript(bad=()), correction_prompt_path=str(tmp_path / "missing.txt")).terms.prompt_version
    assert with_review != without


def test_parse_review_output():
    text = "\n".join([
        "```", "column|status|reject reason (blank is approved)",
        "cust_id|APPROVE|",
        "fstnm|reject|missing segment | first",
        "| adr_ln1 | REJECT | Stripped Numeric Value |",
        "|---|---|---|",
        "unknown|REJECT|x",
        "dob|maybe|unclear",
        "Some explanation without pipes", "```",
    ])
    verdicts = parse_review_output(text, ["cust_id", "fstnm", "adr_ln1", "dob", "zip"])
    assert verdicts == {"cust_id": ("approve", ""), "fstnm": ("reject", "missing segment | first"),
                        "adr_ln1": ("reject", "Stripped Numeric Value")}  # dob: no valid status; zip: no line

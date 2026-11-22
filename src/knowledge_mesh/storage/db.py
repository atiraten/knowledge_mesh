"""SQLite catalog: documents, jobs, structured metadata, keys, profiles, glossary and definitions."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    title TEXT,
    filename TEXT,
    mime_type TEXT,
    content_hash TEXT,
    size_bytes INTEGER,
    original_path TEXT,
    extracted_path TEXT,
    status TEXT NOT NULL,
    status_reason TEXT,
    duplicate_of TEXT,
    page_count INTEGER,
    section_count INTEGER,
    node_count INTEGER NOT NULL DEFAULT 0,
    leaf_count INTEGER NOT NULL DEFAULT 0,
    index_version TEXT,
    indexed_content_hash TEXT,
    metadata_json TEXT,
    ingested_at TEXT,
    updated_at TEXT,
    indexed_at TEXT,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS ix_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    parent_id TEXT,
    level INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    is_leaf INTEGER NOT NULL,
    page INTEGER,
    section TEXT,
    char_count INTEGER
);
CREATE INDEX IF NOT EXISTS ix_chunks_document ON chunks(document_id, is_leaf, ordinal);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    params_json TEXT,
    counts_json TEXT,
    result_json TEXT,
    error_json TEXT,
    trace_id TEXT,
    created_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS job_items (
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ref TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    detail_json TEXT,
    created_at TEXT,
    PRIMARY KEY (job_id, seq)
);

CREATE TABLE IF NOT EXISTS structured_sources (
    source_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT,
    status_reason TEXT,
    last_extracted_at TEXT,
    last_job_id TEXT
);

-- Structured sources added from the Settings page (non-secret settings only).
CREATE TABLE IF NOT EXISTS source_configs (
    source_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    created_by TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    schema_name TEXT,
    name TEXT NOT NULL,
    kind TEXT,
    catalog_ref TEXT NOT NULL,
    physical_ref TEXT,
    row_count INTEGER,
    column_count INTEGER,
    schema_hash TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    entity_label TEXT,
    entity_slug TEXT,
    entity_method TEXT,
    entity_run_id TEXT,
    first_seen_at TEXT,
    last_seen_at TEXT,
    last_job_id TEXT,
    retired_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_datasets_source ON datasets(source_id, active);

CREATE TABLE IF NOT EXISTS columns (
    column_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    name TEXT NOT NULL,
    ordinal INTEGER,
    data_type TEXT,
    nullable INTEGER,
    catalog_ref TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    term_label TEXT,
    term_slug TEXT,
    term_description TEXT,
    term_explanation TEXT,
    term_run_id TEXT,
    term_prompt_version TEXT,
    term_status TEXT,
    term_error TEXT,
    term_sensitive INTEGER,
    first_seen_at TEXT,
    last_seen_at TEXT,
    retired_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_columns_dataset ON columns(dataset_id, active, ordinal);
CREATE INDEX IF NOT EXISTS ix_columns_term ON columns(term_slug);

CREATE TABLE IF NOT EXISTS keys (
    key_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    key_type TEXT NOT NULL,
    origin TEXT NOT NULL,
    constraint_name TEXT,
    columns_json TEXT NOT NULL,
    referenced_dataset_id TEXT,
    referenced_columns_json TEXT,
    is_pk_candidate INTEGER NOT NULL DEFAULT 0,
    profiling_run_id TEXT,
    scope_json TEXT,
    metrics_json TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_keys_dataset ON keys(dataset_id, active);

CREATE TABLE IF NOT EXISTS profiling_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT,
    dataset_ids_json TEXT,
    scope_json TEXT,
    algorithm_json TEXT,
    status TEXT,
    error TEXT,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS column_profiles (
    run_id TEXT NOT NULL,
    column_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    row_count INTEGER,
    distinct_count INTEGER,
    empty_count INTEGER,
    null_count INTEGER,
    format_patterns_json TEXT,
    content_type TEXT,
    most_common_values_json TEXT,
    created_at TEXT,
    type_category TEXT,
    PRIMARY KEY (run_id, column_id)
);
CREATE INDEX IF NOT EXISTS ix_profiles_column ON column_profiles(column_id);

CREATE TABLE IF NOT EXISTS term_generation_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT,
    dataset_id TEXT,
    provider TEXT,
    model TEXT,
    prompt_version TEXT,
    status TEXT,
    error TEXT,
    generated INTEGER,
    failed INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS business_definitions (
    definition_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    definition_text TEXT NOT NULL,
    author TEXT,
    source_ref TEXT,
    created_at TEXT,
    updated_at TEXT,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS definition_links (
    definition_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_label TEXT NOT NULL,
    target_slug TEXT NOT NULL,
    PRIMARY KEY (definition_id, target_type, target_slug)
);

-- Business glossary: one row per term / entity. columns.term_slug and datasets.entity_slug reference them.
-- origin: generated | document | manual | edited; deleted = 1 is a tombstone (never re-created by generation).
CREATE TABLE IF NOT EXISTS business_terms (
    slug TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    description TEXT,
    sensitive INTEGER,
    origin TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    updated_by TEXT
);
CREATE TABLE IF NOT EXISTS business_entities (
    slug TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    description TEXT,
    origin TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    updated_by TEXT
);

CREATE TABLE IF NOT EXISTS doc_entity_mentions (
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    entity_slug TEXT NOT NULL,
    entity_label TEXT NOT NULL,
    PRIMARY KEY (chunk_id, entity_slug)
);
CREATE INDEX IF NOT EXISTS ix_mentions_document ON doc_entity_mentions(document_id);

CREATE TABLE IF NOT EXISTS doc_relationships (
    relationship_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_slug TEXT NOT NULL,
    source_label TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    target_slug TEXT NOT NULL,
    target_label TEXT NOT NULL,
    evidence TEXT,
    method TEXT
);
CREATE INDEX IF NOT EXISTS ix_relationships_document ON doc_relationships(document_id);

-- observability: one row per agent search, its evaluations, user feedback, audit trail, alerts
CREATE TABLE IF NOT EXISTS searches (
    trace_id TEXT PRIMARY KEY,
    span_id TEXT,
    created_at TEXT NOT NULL,
    caller TEXT,
    session_id TEXT,
    query TEXT,
    mode TEXT,
    outcome TEXT,
    latency_ms INTEGER,
    llm_calls INTEGER,
    tool_calls INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    replans INTEGER,
    compliance REAL,
    checks_json TEXT,
    judge_status TEXT
);
CREATE INDEX IF NOT EXISTS ix_searches_created ON searches(created_at);

CREATE TABLE IF NOT EXISTS search_evaluations (
    trace_id TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    name TEXT NOT NULL,
    score REAL,
    label TEXT,
    explanation TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (trace_id, evaluator, name)
);

CREATE TABLE IF NOT EXISTS search_feedback (
    feedback_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT,
    caller TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_feedback_trace ON search_feedback(trace_id);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    trace_id TEXT,
    caller TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_ids_json TEXT,
    decision TEXT NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_events(ts);

-- knowledge graph semantics proposed by the term-generation model (relationship stage), and reviews
CREATE TABLE IF NOT EXISTS relationship_runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT,
    dataset_id TEXT,
    provider TEXT,
    model TEXT,
    prompt_version TEXT,
    status TEXT,
    error TEXT,
    relationships INTEGER,
    term_relations INTEGER,
    synonyms INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS semantic_relationships (
    relationship_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,            -- the referencing dataset whose FK the proposal was made for
    source_id TEXT NOT NULL,
    key_id TEXT NOT NULL,                -- explicit join mapping: the FK the relationship was proposed for
    key_direction TEXT NOT NULL,         -- forward: source entity is the FK's referencing side; reverse otherwise
    name TEXT NOT NULL,
    source_entity_slug TEXT NOT NULL,
    source_entity_label TEXT NOT NULL,
    target_entity_slug TEXT NOT NULL,
    target_entity_label TEXT NOT NULL,
    definition TEXT,
    synonyms_json TEXT,
    inverse_name TEXT,
    cardinality TEXT,
    run_id TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    PRIMARY KEY (relationship_id, key_id)
);
CREATE INDEX IF NOT EXISTS ix_semrel_source ON semantic_relationships(source_id, active);
CREATE INDEX IF NOT EXISTS ix_semrel_dataset ON semantic_relationships(dataset_id, active);

CREATE TABLE IF NOT EXISTS term_relations (
    relation_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    subject_slug TEXT NOT NULL,
    subject_label TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_slug TEXT NOT NULL,
    object_label TEXT NOT NULL,
    definition TEXT,
    run_id TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    PRIMARY KEY (relation_id, dataset_id)
);
CREATE INDEX IF NOT EXISTS ix_termrel_source ON term_relations(source_id, active);

CREATE TABLE IF NOT EXISTS term_synonyms (
    term_slug TEXT NOT NULL,
    synonym TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    run_id TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (term_slug, synonym, dataset_id)
);
CREATE INDEX IF NOT EXISTS ix_synonyms_source ON term_synonyms(source_id, active);

-- verify / reject decisions. object_type: foreign_key | relationship | term_relation | entity_mapping |
-- term_mapping. subject_slug pins a mapping decision to the entity/term it was made for.
CREATE TABLE IF NOT EXISTS review_decisions (
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    status TEXT NOT NULL,
    subject_slug TEXT,
    actor TEXT,
    note TEXT,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (object_type, object_id)
);

-- searches waiting for the user's answer to a clarification question
CREATE TABLE IF NOT EXISTS pending_clarifications (
    trace_id TEXT PRIMARY KEY,
    root_trace_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    caller TEXT,
    session_id TEXT,
    question TEXT NOT NULL,
    state_json TEXT NOT NULL,
    status TEXT NOT NULL,
    answered_at TEXT,
    continuation_trace_id TEXT
);

CREATE TABLE IF NOT EXISTS ops_alerts (
    fingerprint TEXT PRIMARY KEY,
    alertname TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT,
    summary TEXT,
    starts_at TEXT,
    ends_at TEXT,
    received_at TEXT NOT NULL,
    labels_json TEXT
);
"""


# (table, column, declaration) of columns added to existing tables; added at startup when missing (existing rows keep
# NULL there until they are rewritten).
ADDED_COLUMNS = [
    ("column_profiles", "type_category", "TEXT"),  # generic type from profiling: string, integer, decimal, date, ...
    ("columns", "term_sensitive", "INTEGER"),  # 1 = the business term can identify a person; NULL = not assessed
    ("business_definitions", "document_id", "TEXT"),  # the policy document where the definition is defined
    ("business_definitions", "location", "TEXT"),     # page or section in that document
]


class Database:
    """Thin SQLite wrapper. One connection per thread; WAL mode for concurrent readers."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            for table, column, decl in ADDED_COLUMNS:  # columns added after a table was first created
                if column not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        yield self._conn()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._write_lock:
            cur = self._conn().execute(sql, tuple(params))
            return cur.rowcount

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self.transaction() as conn:
            conn.executemany(sql, [tuple(r) for r in rows])

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn().execute(sql, tuple(params)).fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        row = self._conn().execute(sql, tuple(params)).fetchone()
        return dict(row) if row else None

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        row = self._conn().execute(sql, tuple(params)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def dumps(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)

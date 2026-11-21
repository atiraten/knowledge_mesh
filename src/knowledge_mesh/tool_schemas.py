"""Published output schemas of the MCP tools ``query_knowledge_graph``, ``execute_sql`` and ``generate_html``.

Every MCP result also carries ``ok`` (false when the call itself failed, with ``error``) and ``trace_id``.
Models allow extra fields so that additions stay backward compatible.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolError(_Open):
    code: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Safe diagnostic message.")


class _Envelope(_Open):
    ok: bool = Field(description="false when the tool call failed before producing a result (see error).")
    trace_id: str | None = Field(default=None, description="Trace of this call.")
    error: ToolError | None = Field(default=None, description="Present when ok is false or status is error.")


# -- query_knowledge_graph ------------------------------------------------------------------------
class KGTableRef(_Open):
    iri: str
    name: str | None = None
    mapping_status: str | None = Field(default=None, description="verified | inferred")


class KGEntity(_Open):
    iri: str = Field(description="Stable IRI, e.g. urn:mesh:entity:customer.")
    name: str | None = None
    definition: str | None = None
    alt_names: list[str] = Field(default_factory=list)
    classification: list[str] = Field(default_factory=list, description="e.g. owl:Class, mesh:BusinessEntity.")
    role: str | None = Field(default=None, description="matched (resolved from the queries) | path (bridge entity).")
    match_type: str | None = Field(default=None, description="iri | exact | label | alias | token_set | semantic.")
    matched_queries: list[str] = Field(default_factory=list)
    tables: list[KGTableRef] = Field(default_factory=list)
    term_iris: list[str] = Field(default_factory=list)


class KGColumnRef(_Open):
    iri: str
    name: str | None = None
    table: str | None = None
    catalog_ref: str | None = None
    data_type: str | None = None
    mapping_status: str | None = None


class KGTerm(_Open):
    iri: str = Field(description="Stable IRI of the SKOS concept, e.g. urn:mesh:term:birth-date.")
    name: str | None = None
    entity_iri: str | None = Field(default=None, description="Owning entity when there is exactly one.")
    entity_iris: list[str] = Field(default_factory=list)
    definition: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    property_iri: str | None = Field(default=None, description="The term's separate OWL datatype property.")
    role: str | None = Field(default=None, description="matched | join_column.")
    match_type: str | None = None
    matched_queries: list[str] = Field(default_factory=list)
    columns: list[KGColumnRef] = Field(default_factory=list)
    related_terms: list[dict[str, Any]] = Field(default_factory=list)


class KGPathEdge(_Open):
    relationship_iri: str
    predicate: str = Field(description="Predicate name in the traversal direction (inverse name when reverse).")
    direction: str = Field(description="forward | reverse (relative to the relationship's own direction).")
    kind: str = Field(description="semantic | navigational (from a foreign key) | document.")
    status: str = Field(description="verified | inferred.")
    join_refs: list[str] = Field(default_factory=list, description="join_id values in physical_mappings.joins.")
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None


class KGPath(_Open):
    path_id: str
    hops: int
    nodes: list[str] = Field(description="Ordered entity IRIs, including bridge entities.")
    node_names: list[str] = Field(default_factory=list)
    edges: list[KGPathEdge]
    label: str | None = None
    executable: bool = Field(description="True when every hop has a join mapping usable for SQL.")
    missing_mappings: list[str] = Field(default_factory=list)
    validation_status: str = Field(description="verified only when every hop is a verified semantic relationship.")
    relevance_score: float = Field(description="Uncalibrated ranking score (not a probability).")
    score_explanation: dict[str, Any] = Field(default_factory=dict)


class KGColumnPair(_Open):
    position: int
    from_column: str | None = None
    to_column: str | None = None
    from_column_iri: str | None = None
    to_column_iri: str | None = None


class KGJoin(_Open):
    join_id: str
    foreign_key_iri: str
    relationship_iri: str
    relationship: str | None = None
    method: str = Field(description="explicit | automatic | foreign_key.")
    direction: str | None = None
    from_table: str | None = None
    to_table: str | None = None
    column_pairs: list[KGColumnPair] = Field(description="Ordered source-to-target column pairs (composite keys).")
    join_predicate: str
    origin: str | None = Field(default=None, description="declared | inferred foreign key.")
    status: str | None = None
    foreign_key_status: str | None = None
    confidence: str | None = None


class KGPhysicalMappings(_Open):
    tables: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[dict[str, Any]] = Field(default_factory=list)
    joins: list[KGJoin] = Field(default_factory=list)


class KGQueryOutput(_Envelope):
    intent_id: str | None = None
    status: str | None = Field(default=None, description="ok | no_match | error.")
    queries: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)
    business_entities: list[KGEntity] = Field(default_factory=list)
    business_terms: list[KGTerm] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    paths: list[KGPath] = Field(default_factory=list)
    physical_mappings: KGPhysicalMappings | None = None
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list, description="References (IDs) only.")
    warnings: list[str] = Field(default_factory=list)
    ambiguities: list[dict[str, Any]] = Field(default_factory=list)
    assertion_level: str | None = Field(default=None, description='"schema": no record-level facts.')
    note: str | None = None
    graph: dict[str, Any] | None = Field(default=None, description="version, published_at, named_graphs.")
    truncated: dict[str, int] | None = Field(default=None, description="Items omitted per list to respect "
                                                                     "knowledge_graph.max_response_chars.")


# -- execute_sql -----------------------------------------------------------------------------------
class SqlColumn(_Open):
    name: str
    type: str | None = None
    sensitive: bool | None = Field(default=None, description="true: computed from a column whose business term can "
                                   "identify a person (PII / PHI / GDPR), COUNT excepted; null: not assessed.")
    sensitive_terms: list[str] = Field(default_factory=list, description="The sensitive business terms involved.")


class SqlOutput(_Envelope):
    status: str | None = Field(default=None, description="ok | error.")
    source_id: str | None = None
    dialect: str | None = None
    sql: str | None = None
    tables: list[dict[str, Any]] = Field(default_factory=list, description="Cataloged datasets the query reads.")
    columns: list[SqlColumn] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list, description="Empty when the query matched nothing.")
    row_count: int | None = None
    truncated: bool | None = Field(default=None, description="More rows exist than max_rows.")
    max_rows: int | None = None
    elapsed_ms: int | None = None


# -- generate_html ---------------------------------------------------------------------------------
class HtmlOutput(_Envelope):
    status: str | None = None
    html: str | None = Field(default=None, description='One <div class="km-html"> fragment; values escaped; '
                                                  'tables are grids (<table class="km-grid">); cells of '
                                                  'sensitive_columns have class="km-sensitive" (red).')
    blocks_rendered: int | None = None
    truncated: bool | None = None
    format: str | None = None
    allowed_tags: list[str] = Field(default_factory=list)

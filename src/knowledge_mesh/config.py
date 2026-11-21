"""Load and validate ``config/mesh.yaml``.

Relative paths are resolved against the repository root, defined as the parent of the
directory holding the config file (``<repo>/config/mesh.yaml`` -> ``<repo>``).
Credentials are never stored here: settings only name environment variables or key files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_ENV = "KNOWLEDGE_MESH_CONFIG"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StorageSettings(_Base):
    originals_dir: Path = Path("./data/originals")
    extracted_dir: Path = Path("./data/extracted")
    metadata_db: Path = Path("./data/metadata.sqlite")
    allowed_ingest_roots: list[Path] = Field(default_factory=lambda: [Path("./data/inbox")])
    # Directories under which Parquet repositories may be added from the Settings page.
    allowed_source_roots: list[Path] = Field(default_factory=lambda: [Path("./data/sources")])
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".html", ".htm"]
    )

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_ext(cls, v: list[str]) -> list[str]:
        return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in v]


class CrawlerSettings(_Base):
    allowed_scope: Literal["hostname"] = "hostname"
    allowed_hosts: list[str] = Field(default_factory=list)
    max_depth: int = Field(default=3, ge=1, le=20)  # 1 = the start page only
    max_pages: int = Field(default=100, ge=1, le=100_000)
    concurrency_per_host: int = Field(default=2, ge=1, le=32)
    requests_per_second_per_host: float = Field(default=2.0, gt=0)
    request_timeout_seconds: float = Field(default=20, gt=0)
    max_response_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    respect_robots_txt: bool = True
    use_sitemap: bool = True
    deduplicate_urls: bool = True
    deduplicate_content_sha256: bool = True
    user_agent: str = "KnowledgeMeshBot/0.1 (+https://github.com/atiraten/knowledge_mesh)"
    max_redirects: int = Field(default=5, ge=0, le=20)
    max_url_length: int = Field(default=2048, ge=64)
    max_query_variants_per_path: int = Field(default=10, ge=1)
    strip_query_params: list[str] = Field(
        default_factory=lambda: [
            "utm_*", "gclid", "fbclid", "mc_cid", "mc_eid", "msclkid",
            "sessionid", "phpsessid", "jsessionid", "sid", "ref",
        ]
    )
    blocked_hosts: list[str] = Field(
        default_factory=lambda: ["metadata.google.internal", "metadata", "instance-data", "localhost"]
    )
    # Only for local testing against private hosts. Never enable for a shared deployment.
    allow_private_networks: bool = False
    max_manifest_skipped: int = Field(default=1000, ge=0)


class ChunkingSettings(_Base):
    library: Literal["llamaindex"] = "llamaindex"
    strategy: Literal["hierarchical"] = "hierarchical"
    chunk_sizes_tokens: list[int] = Field(default_factory=lambda: [2048, 1024, 256])
    chunk_overlap_tokens: int = Field(default=20, ge=0)

    @field_validator("chunk_sizes_tokens")
    @classmethod
    def _descending(cls, v: list[int]) -> list[int]:
        if not v or any(s <= 0 for s in v):
            raise ValueError("chunk_sizes_tokens must contain positive sizes")
        if v != sorted(v, reverse=True):
            raise ValueError("chunk_sizes_tokens must be ordered from largest parent to leaf")
        return v


class EmbeddingSettings(_Base):
    # "hashing" is a deterministic offline embedding used only by the automated tests.
    provider: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    model: str = "sentence-transformers/all-mpnet-base-v2"
    revision: str | None = None
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    local_files_only: bool = False


class VectorStoreSettings(_Base):
    provider: Literal["llamaindex_simple"] = "llamaindex_simple"
    document_persist_dir: Path = Path("./data/index/documents")
    glossary_persist_dir: Path = Path("./data/index/glossary")


class StructuredSourceSettings(_Base):
    id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
    kind: Literal["database", "parquet", "databricks"]
    description: str | None = None
    # Where the secret (database connection URL / Databricks token) comes from:
    # "env" = the environment variable named below; "secret_store" = the OS keychain entry
    # written by the Settings page. Secrets never appear in this model.
    credential_store: Literal["env", "secret_store"] = "env"
    # database, described by fields (password from the keychain or password_env) ...
    db_type: Literal["postgresql", "mysql", "mssql", "oracle"] | None = None
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = None            # Oracle: service name
    username: str | None = None
    password_env: str | None = None
    # ... or by a full SQLAlchemy URL (SQLite files, other dialects)
    connection_url_env: str | None = None
    connection_display: str | None = None  # password-masked URL, for display only
    schemas: list[str] | None = None
    # parquet
    path: Path | None = None
    # databricks (hostname and HTTP path are not secrets: give the value or an env var name)
    server_hostname: str | None = None
    server_hostname_env: str | None = None
    http_path: str | None = None
    http_path_env: str | None = None
    token_env: str | None = None
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    # Optional allowlist of dataset names ("table" or "schema.table"); others are ignored.
    datasets: list[str] | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _check_kind(self) -> "StructuredSourceSettings":
        env = self.credential_store == "env"
        if self.kind == "database":
            if self.db_type or self.host or self.database:
                missing = [name for name, value in (("database type", self.db_type), ("hostname", self.host),
                                                    ("database name", self.database)) if not value]
                if missing:
                    raise ValueError(f"source {self.id}: database sources need {', '.join(missing)}")
            elif env and not self.connection_url_env:
                raise ValueError(f"source {self.id}: database sources need a database type, hostname and "
                                 "database name (or a connection URL)")
        if self.kind == "parquet" and not self.path:
            raise ValueError(f"source {self.id}: parquet sources require path")
        if self.kind == "databricks":
            missing = [k for k, ok in (
                ("server_hostname or server_hostname_env", self.server_hostname or self.server_hostname_env),
                ("http_path or http_path_env", self.http_path or self.http_path_env),
                ("token_env (or credential_store: secret_store)", self.token_env or not env),
                ("catalog", self.catalog),
            ) if not ok]
            if missing:
                raise ValueError(f"source {self.id}: databricks sources require {', '.join(missing)}")
        return self

    def public_view(self) -> dict:
        """Settings safe to display: env var names only, never their values."""
        data = self.model_dump(by_alias=True, exclude_none=True)
        if "path" in data:
            data["path"] = str(data["path"])
        return data


class ProfilingSettings(_Base):
    max_rows: int = Field(default=100_000, ge=0)
    max_key_size: int = Field(default=4, ge=1, le=8)
    max_fk_arity: int = Field(default=3, ge=1, le=6)
    max_columns: int = Field(default=64, ge=1, le=256)
    ucc_algorithm: Literal["hyucc", "apriori"] = "hyucc"
    null_equals_null: bool = True
    min_rows_for_inference: int = Field(default=2, ge=2)
    key_excluded_type_keywords: list[str] = Field(
        default_factory=lambda: ["float", "double", "real", "binary", "blob", "json", "bytes"]
    )
    binder_buckets: int = Field(default=16, ge=1, le=1024)
    binder_min_distinct_dependent: int = Field(default=2, ge=1)
    binder_allow_cross_type: bool = False
    # Foreign keys are identifiers (see structured/fk_rules.py). A column is an identifier when it is
    # part of a declared key, is a UUID, its business term ends with one of these class words, or (no
    # term yet) its name ends with one of the name tokens. Set fk_identifiers_only: false to let BINDER
    # report inclusions between any columns.
    fk_identifiers_only: bool = True
    # FKs reference the primary key: a table's declared primary/unique keys when it has a declared
    # primary key, otherwise its primary-key candidates (null-free unique keys). False: any unique key.
    fk_reference_primary_keys: bool = True
    fk_identifier_class_words: list[str] = Field(
        default_factory=lambda: ["Identifier", "Code", "Number", "Key", "Indicator", "Flag"]
    )
    fk_identifier_name_tokens: dict[str, str] = Field(default_factory=lambda: {
        "id": "Identifier", "ident": "Identifier", "identifier": "Identifier", "uuid": "Identifier",
        "guid": "Identifier", "key": "Key", "cd": "Code", "code": "Code", "num": "Number", "nbr": "Number",
        "no": "Number", "number": "Number", "ind": "Indicator", "indicator": "Indicator", "flg": "Flag",
        "flag": "Flag",
    })
    max_nary_candidates: int = Field(default=10_000, ge=1)
    max_ucc_candidates: int = Field(default=100_000, ge=1)
    most_common_values: int = Field(default=5, ge=0, le=100)
    store_most_common_values: bool = True
    format_patterns: int = Field(default=5, ge=1, le=50)
    max_pattern_length: int = Field(default=64, ge=8)
    content_type_threshold: float = Field(default=0.9, gt=0, le=1)


class RankingWeights(_Base):
    """Path score = semantic * mean(edge relevance) + coverage * join coverage + status * mean(status weight)."""

    semantic: float = Field(default=0.6, ge=0)
    coverage: float = Field(default=0.25, ge=0)
    status: float = Field(default=0.15, ge=0)


class KnowledgeGraphSettings(_Base):
    """RDF knowledge graph (rdflib Dataset persisted as TriG) and the ``query_knowledge_graph`` tool."""

    enabled: bool = True                                   # query_knowledge_graph available
    storage_backend: Literal["trig_file"] = "trig_file"    # dataset_location = data_path
    data_path: Path = Path("./data/knowledge_graph.trig")
    shapes_path: Path = Path("./config/knowledge_graph_shapes.ttl")
    ontology_path: Path = Path("./config/knowledge_graph_ontology.ttl")
    default_max_hops: int = Field(default=5, ge=1, le=20)
    max_hops_limit: int = Field(default=8, ge=1, le=20)    # largest max_hops a caller may ask for
    default_top_k: int = Field(default=5, ge=1)
    max_top_k: int = Field(default=20, ge=1)
    max_candidate_paths: int = Field(default=200, ge=1)    # candidate paths kept for ranking
    max_expansions: int = Field(default=5000, ge=1)        # traversal steps per request
    query_timeout_seconds: float = Field(default=5.0, gt=0)
    max_response_chars: int = Field(default=20000, ge=2000)
    include_inferred_relationships: bool = True            # false: traverse verified relationships only
    embedding_retrieval_enabled: bool = True               # glossary vectors + relationship similarity
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)
    view_max_nodes: int = Field(default=600, ge=10)        # catalog knowledge graph view

    @model_validator(mode="after")
    def _bounds(self) -> "KnowledgeGraphSettings":
        if self.default_max_hops > self.max_hops_limit:
            raise ValueError("knowledge_graph.default_max_hops cannot exceed knowledge_graph.max_hops_limit")
        if self.default_top_k > self.max_top_k:
            raise ValueError("knowledge_graph.default_top_k cannot exceed knowledge_graph.max_top_k")
        return self


class RelationshipGenerationSettings(_Base):
    """Named relationships, term relations and term synonyms proposed by the term-generation model."""

    enabled: bool = True
    prompt_path: Path = Path("./prompts/relationship_generation.txt")
    min_fk_confidence: Literal["low", "medium", "high"] = "medium"   # inferred FKs below this are skipped
    max_synonyms: int = Field(default=5, ge=0, le=20)


class SqlExecutionSettings(_Base):
    """``execute_sql``: one read-only SELECT on a cataloged database or Databricks source."""

    enabled: bool = True
    default_max_rows: int = Field(default=200, ge=1)
    max_rows_limit: int = Field(default=1000, ge=1)
    timeout_seconds: float = Field(default=30, gt=0)
    max_sql_chars: int = Field(default=10000, ge=100)

    @model_validator(mode="after")
    def _bounds(self) -> "SqlExecutionSettings":
        if self.default_max_rows > self.max_rows_limit:
            raise ValueError("sql_execution.default_max_rows cannot exceed sql_execution.max_rows_limit")
        return self


class HtmlOutputSettings(_Base):
    """``generate_html``: deterministic, escaped HTML fragments."""

    max_chars: int = Field(default=200000, ge=1000)
    max_table_rows: int = Field(default=500, ge=1)


class EntityExtractionSettings(_Base):
    enabled: bool = True
    method: Literal["lexicon", "llm"] = "lexicon"
    entity_lexicon: list[str] = Field(
        default_factory=lambda: [
            "Customer", "Order", "Order Line", "Supplier", "Vendor", "Account", "Payment",
            "Product", "Invoice", "Employee", "Shipment", "Contract", "Store", "Transaction",
            "Claim", "Policy", "Loan", "Address", "Warehouse", "Department",
        ]
    )
    include_catalog_entities: bool = True
    max_gap_words: int = Field(default=8, ge=1, le=30)
    max_llm_chunks_per_document: int = Field(default=50, ge=1)


class RetrievalSettings(_Base):
    top_k: int = Field(default=5, ge=1)
    max_graph_results: int = Field(default=10, ge=0)
    max_top_k: int = Field(default=20, ge=1)
    expand_parent_context: bool = True
    max_context_tokens: int = Field(default=6000, ge=0)
    reranker_enabled: bool = False
    min_document_similarity: float = Field(default=0.3, ge=-1, le=1)
    min_glossary_similarity: float = Field(default=0.3, ge=-1, le=1)
    excerpt_chars: int = Field(default=2000, ge=100)  # above the longest 256-token leaf, so chunks arrive whole
    max_query_chars: int = Field(default=2000, ge=10)
    max_queries: int = Field(default=5, ge=1, le=20)           # queries in one multi-query retrieval
    max_parallel_queries: int = Field(default=4, ge=1, le=16)  # of those, searched at the same time

    @model_validator(mode="after")
    def _bounds(self) -> "RetrievalSettings":
        if self.top_k > self.max_top_k:
            raise ValueError("retrieval.top_k cannot exceed retrieval.max_top_k")
        return self


class RerankerSettings(_Base):
    """Laya decision-model reranking of the retrieved document passages (retrieval.reranker_enabled).

    Call 1 ("description", choice) gives each top-k passage a probability (they sum to 1).
    Passages above ``min_probability`` are kept, the best ``top_n`` of them go to call 2
    ("quality", score), and the best one rated ``min_quality`` or better is the response.
    Otherwise the response is ``no_answer_message``.
    """

    provider: Literal["laya"] = "laya"
    model: str = "convaiinnovations/laya"
    subfolder: str | None = None          # "multilingual" for non-English corpora
    device: str | None = None             # None = Laya's default (cuda/mps/cpu)
    min_probability: float = Field(default=0.8, gt=0, le=1)
    top_n: int = Field(default=2, ge=1, le=20)
    min_quality: str = "Acceptable"
    quality_levels: list[str] = Field(
        default_factory=lambda: ["Incorrect", "Ambiguous", "Acceptable", "Clear", "Excellent"])
    choice_instructions: str = "Which response best matches answer the question?"
    quality_instructions: str = "Is the chosen response correct?"
    max_candidate_chars: int = Field(default=1500, ge=50)
    choice_head_max_len: int = Field(default=448, ge=64, le=512)  # token room for the options in call 1
    no_answer_message: str = "I could not find a response."

    @model_validator(mode="after")
    def _levels(self) -> "RerankerSettings":
        if len(self.quality_levels) < 2:
            raise ValueError("reranker.quality_levels needs at least two levels")
        if self.min_quality not in self.quality_levels:
            raise ValueError(f"reranker.min_quality must be one of {self.quality_levels}")
        return self


class LLMSettings(_Base):
    provider: Literal["google", "openai", "fake"] | None = None
    model: str | None = None
    api_key_env: str | None = None
    api_key_file: Path | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = Field(default=8192, ge=1)
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0)

    @property
    def configured(self) -> bool:
        return bool(self.provider and self.model)


class TermGenerationSettings(LLMSettings):
    prompt_path: Path = Path("./prompts/term_generation.txt")
    industry_name: str = "General Business"
    language_name: str = "English"
    class_words: list[str] = Field(
        default_factory=lambda: [
            "Identifier", "Code", "Name", "Description", "Date", "Timestamp", "Amount",
            "Quantity", "Count", "Number", "Rate", "Percent", "Indicator", "Text", "Type", "Status",
        ]
    )
    abbreviations: dict[str, str] = Field(default_factory=dict)
    abbreviations_path: Path | None = None
    max_columns_per_request: int = Field(default=40, ge=1, le=500)


class AnswerGenerationSettings(LLMSettings):
    enabled: bool = False
    max_evidence_items: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _enabled_needs_model(self) -> "AnswerGenerationSettings":
        if self.enabled and not self.configured:
            raise ValueError("answer_generation.enabled requires provider and model")
        return self


class AgentSettings(LLMSettings):
    """Search agent harness: planner LLM -> orchestration loop over the plan's tasks -> final response.

    The planner call uses ``planner_prompt_path`` and never sees the tool list. For each ready task an
    executor loop runs: the model evaluates the last tool results and picks the next tool calls
    (``executor_prompt_path``, with the intents and the tool list) until it completes the task or
    ``max_iterations`` calls were made. Each task result is then evaluated: next task, replan
    (planner LLM, at most ``max_replans`` times) or finish. ``synthesis_prompt_path`` writes the answer.
    """

    enabled: bool = False
    planner_prompt_path: Path = Path("./prompts/planner.txt")
    executor_prompt_path: Path = Path("./prompts/executor.txt")
    synthesis_prompt_path: Path = Path("./prompts/synthesis.txt")
    max_iterations: int = Field(default=20, ge=1, le=100)     # executor LLM calls per task
    max_intents: int = Field(default=5, ge=1, le=20)
    max_parallel_intents: int = Field(default=3, ge=1, le=16)  # ready tasks run together (parallel/mixed plans)
    max_retrieval_queries: int = Field(default=3, ge=1, le=5)  # per intent, from the planner
    max_replans: int = Field(default=1, ge=0, le=5)            # planner calls to revise the plan per search
    max_tool_calls_per_step: int = Field(default=4, ge=1, le=10)
    max_parallel_tool_calls: int = Field(default=4, ge=1, le=16)
    max_observation_chars: int = Field(default=8000, ge=500)  # one tool result in the context
    max_context_chars: int = Field(default=60000, ge=2000)    # steps shown to the model / evidence for synthesis
    max_evidence_items: int = Field(default=24, ge=1)          # evidence blocks passed to synthesis
    sql_generation_prompt_path: Path = Path("./prompts/sql_generation.txt")  # generate_sql tool (agent model)
    clarification_enabled: bool = True     # ask the user for missing information instead of "nothing found"
    clarification_ttl_hours: float = Field(default=24, gt=0)   # pending clarifications expire after this
    max_clarifications: int = Field(default=2, ge=0, le=10)    # questions per request chain

    @model_validator(mode="after")
    def _enabled_needs_model(self) -> "AgentSettings":
        if self.enabled and not self.configured:
            raise ValueError("agent.enabled requires provider and model")
        return self


class JobSettings(_Base):
    workers: int = Field(default=2, ge=1, le=32)
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_initial_backoff_seconds: float = Field(default=0.5, ge=0)
    retry_max_backoff_seconds: float = Field(default=10, ge=0)


class ObservabilitySettings(_Base):
    enabled: bool = True
    service_name: str = "knowledge-mesh"
    otlp_endpoint: str = "http://localhost:4317"
    capture_sensitive_content: bool = False
    phoenix_enabled: bool = True
    prometheus_enabled: bool = True
    loki_enabled: bool = True
    ragas_offline_evaluation: bool = True
    phoenix_url: str = "http://localhost:6006"   # trace links in the UI and span annotations (judge, feedback)
    online_judge: bool = False                   # LLM judge (agent model) on every agent search, in the background
    judge_prompt_path: Path = Path("./prompts/judge.txt")
    alert_token_file: Path = Path("./deploy/secrets/alert_webhook_token")  # Alertmanager -> /api/ops/alerts
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"


class SecretsSettings(_Base):
    # keyring = OS secret store (macOS Keychain, Windows Credential Locker, Secret Service on Linux)
    backend: Literal["keyring", "none"] = "keyring"
    service_name: str = "knowledge-mesh"


class ApiSettings(_Base):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key_env: str | None = "KNOWLEDGE_MESH_API_KEY"


class MeshConfig(_Base):
    storage: StorageSettings = Field(default_factory=StorageSettings)
    crawler: CrawlerSettings = Field(default_factory=CrawlerSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    structured_sources: list[StructuredSourceSettings] = Field(default_factory=list)
    profiling: ProfilingSettings = Field(default_factory=ProfilingSettings)
    knowledge_graph: KnowledgeGraphSettings = Field(default_factory=KnowledgeGraphSettings)
    relationship_generation: RelationshipGenerationSettings = Field(default_factory=RelationshipGenerationSettings)
    sql_execution: SqlExecutionSettings = Field(default_factory=SqlExecutionSettings)
    html_output: HtmlOutputSettings = Field(default_factory=HtmlOutputSettings)
    entity_extraction: EntityExtractionSettings = Field(default_factory=EntityExtractionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    term_generation: TermGenerationSettings = Field(default_factory=TermGenerationSettings)
    answer_generation: AnswerGenerationSettings = Field(default_factory=AnswerGenerationSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    jobs: JobSettings = Field(default_factory=JobSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    secrets: SecretsSettings = Field(default_factory=SecretsSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @field_validator("structured_sources")
    @classmethod
    def _unique_ids(cls, v: list[StructuredSourceSettings]) -> list[StructuredSourceSettings]:
        ids = [s.id for s in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate structured source ids: {sorted(dupes)}")
        return v

    def source(self, source_id: str) -> StructuredSourceSettings | None:
        return next((s for s in self.structured_sources if s.id == source_id), None)

    def resolve(self, path: Path | str | None) -> Path | None:
        if path is None:
            return None
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def resolved_paths(self) -> "MeshConfig":
        """Return a copy where all filesystem paths are absolute."""
        data = self.model_copy(deep=True)
        s = data.storage
        s.originals_dir = self.resolve(s.originals_dir)
        s.extracted_dir = self.resolve(s.extracted_dir)
        s.metadata_db = self.resolve(s.metadata_db)
        s.allowed_ingest_roots = [self.resolve(p) for p in s.allowed_ingest_roots]
        s.allowed_source_roots = [self.resolve(p) for p in s.allowed_source_roots]
        data.vector_store.document_persist_dir = self.resolve(data.vector_store.document_persist_dir)
        data.vector_store.glossary_persist_dir = self.resolve(data.vector_store.glossary_persist_dir)
        data.knowledge_graph.data_path = self.resolve(data.knowledge_graph.data_path)
        data.knowledge_graph.shapes_path = self.resolve(data.knowledge_graph.shapes_path)
        data.knowledge_graph.ontology_path = self.resolve(data.knowledge_graph.ontology_path)
        data.relationship_generation.prompt_path = self.resolve(data.relationship_generation.prompt_path)
        data.agent.sql_generation_prompt_path = self.resolve(data.agent.sql_generation_prompt_path)
        for llm in (data.term_generation, data.answer_generation, data.agent):
            llm.api_key_file = self.resolve(llm.api_key_file)
        data.term_generation.prompt_path = self.resolve(data.term_generation.prompt_path)
        data.agent.planner_prompt_path = self.resolve(data.agent.planner_prompt_path)
        data.agent.executor_prompt_path = self.resolve(data.agent.executor_prompt_path)
        data.agent.synthesis_prompt_path = self.resolve(data.agent.synthesis_prompt_path)
        data.observability.judge_prompt_path = self.resolve(data.observability.judge_prompt_path)
        data.observability.alert_token_file = self.resolve(data.observability.alert_token_file)
        data.term_generation.abbreviations_path = self.resolve(data.term_generation.abbreviations_path)
        for src in data.structured_sources:
            if src.path is not None:
                src.path = self.resolve(src.path)
        return data


class ConfigError(RuntimeError):
    pass


def default_config_path() -> Path:
    env = os.environ.get(DEFAULT_CONFIG_ENV)
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "mesh.yaml"
        if candidate.exists():
            return candidate
    return Path.cwd() / "config" / "mesh.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> MeshConfig:
    """Load, validate and path-resolve the configuration. Raises ConfigError on failure."""
    cfg_path = Path(path) if path else default_config_path()
    if not cfg_path.exists():
        raise ConfigError(f"configuration file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a mapping at the top level")
    if overrides:
        raw = _deep_merge(raw, overrides)
    # Normalize sections written as empty keys (e.g. "structured_sources:" with no items).
    raw = {k: v for k, v in raw.items() if v is not None}
    cfg_dir = cfg_path.resolve().parent
    base_dir = cfg_dir.parent if cfg_dir.name == "config" else cfg_dir
    try:
        cfg = MeshConfig.model_validate({**raw, "base_dir": base_dir})
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {cfg_path}:\n{exc}") from exc
    return cfg.resolved_paths()


def config_from_dict(data: dict, base_dir: str | Path) -> MeshConfig:
    try:
        cfg = MeshConfig.model_validate({**data, "base_dir": Path(base_dir)})
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    return cfg.resolved_paths()


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# Knowledge Mesh — Requirements Specification

**Version:** 1.0  
**Date:** 2026-11-19  
**Status:** Ready for incremental POC implementation, with source connector and LLM choices pending

## 1. Purpose

Build a Python-based Knowledge Mesh with a FastAPI browser interface for ingesting documents and websites and cataloging configured database, Parquet, and Databricks sources. The system indexes document passages, extracts business entities and relationships from documents, discovers structured keys, generates business terms and entities from schemas, and captures business definitions. It persists structured metadata in SQLite and represents business concepts, relationships, and constraints in an RDF knowledge graph. A single query can retrieve document passages, structured metadata, and graph evidence for search or optional grounded answers with citations.

## 2. Scope

### In scope
#### Unstructured Data
- User-uploaded **PDF**, **TXT**, and common document files (initially **DOCX**), plus HTML pages retrieved from URLs.
- Ingestion from local files and local directories.
- Crawling websites starting from a user-supplied HTTP/HTTPS URL.
- Local storage of original content and extracted text.
- Hierarchical chunking and vector indexing with LlamaIndex.
- Local embedding inference using `sentence-transformers/all-mpnet-base-v2`.
- LlamaIndex `SimpleVectorStore` with persistent storage across restarts.
- Top-K semantic retrieval, with default K = 5.
- MCP tools for crawling, ingestion, indexing, document reading, and retrieval.
- FastAPI page with a simple browser interface for uploads, URL submission, search, and chat.
- Extract Business Entities from the documents such as Customer, Orders, Supplier, Account, Payment ...etc and the relationships between these entities.

### Business Definitions
- Capture Business Definitions such as "Active Customer", "Net Revenue" 
- Provide a web page to provide enter these definitions. These definitions shall have to their corresponding business terms and/or business entities.

#### Structured Data
- Use SQLite to store structured metadata.
- Use an MSP tool to extract the metadata from databases, parquet and databricks.
- Use an MCP tool to extract unique keys, primary keys and foreign key between entities and populate the metadata accordingly. Use HyUCC and Apriori algorithms to detect unique keys. A unique Key can be a composite key. Use Binder to detect Foreign Keys.
- Use an LLM configured in config/mess.yaml to generate a business term for each column and Business Entities for table name. Use the prompt in prompts/term_generation.txt . 
- Create a special index for Business Terms (extracted from column names) and Business Entities (extracted from table names) in LlamaIndex `SimpleVectorStore`  to search for similar business terms.
- Build web page to view the metadata with hierarchy Database list --> Schemas --> Tables--> Columns
- Column shall have display a link to the business terms.

### Knowledge Graph
- Build an knowledge graph to represent the entities, their relationship and constraints.
- Use RDFLib + pySHACL for Knowledge Graph to store the Business Entities, Business Terms, Business Definitions and Constraints.

#### Observability
- OpenTelemetry-based instrumentation; Phoenix, Prometheus, Grafana, Loki, and Ragas for observability/evaluation.

### Out of scope for this POC

- SharePoint integration.
- Distributed vector databases and horizontal scaling.
- Mandatory reranking, BM25/hybrid retrieval, or OCR for scanned PDFs (potential later enhancements).
- Multi-tenant administration. Any externally accessible deployment must still provide appropriate authentication and access controls.

## 3. System Overview

```text
Browser UI (FastAPI)
    |
    v
Knowledge Mesh orchestrator
    |-- MCP document ingestion, crawling, indexing, and reading
    |     -> originals + extracted text + LlamaIndex node/document store
    |     -> document passage index + document entity and relationship evidence
    |-- MCP structured metadata extraction and key discovery
    |     -> SQLite catalog of sources, tables, columns, and keys
    |     -> LLM-generated business entities and terms -> business glossary index
    |-- Business definition capture and graph publication
    |     -> RDFLib graph + pySHACL validation
    `-- MCP retrieval across both indexes and the knowledge graph
          -> ranked passages, metadata references, and graph evidence
          -> optional grounded answer with citations

Observability across all layers:
OpenTelemetry -> Phoenix (AI traces), Prometheus/Grafana (metrics),
                 Loki (logs); Ragas (offline evaluation)
```

Document passages and the business glossary SHALL use separate persistent LlamaIndex `SimpleVectorStore` indexes. The orchestrator SHALL pass bounded evidence to the answer-generation model. Ingestion and profiling tools SHALL return durable references and bounded summaries.

## 4. Functional Requirements

### FR-01 — Supported inputs

1. The system SHALL ingest local **PDF, DOCX, TXT**, and HTML content from URLs.
2. The system SHALL accept document uploads through the web UI.
3. The system SHALL ingest supported files from a configured local directory.
4. The system SHALL accept a starting HTTP/HTTPS URL and crawl eligible pages on that website.
5. Unsupported formats SHALL be rejected or marked as skipped with a recorded reason.
6. Parsing failures on one document SHALL NOT terminate the entire ingestion job.
7. The system SHALL accept configured structured sources for relational databases, Parquet files, and Databricks.

### FR-02 — Source identity and provenance

1. Every ingested source SHALL receive a stable `document_id`.
2. Metadata SHALL include: source type, original URL/path, title/filename, MIME type, content hash, ingestion timestamp, processing status, and optional page/section details.
3. Every indexed chunk SHALL reference its source `document_id`, its own `chunk_id`, and its parent node ID if applicable.
4. Document citations SHALL resolve to the original URL or stored document and the page or section when available.
5. Structured metadata SHALL carry stable source, dataset, and column identifiers that resolve to the configured source and its physical schema or file. Discovered keys SHALL also reference the profiling run and the data scope evaluated.
6. Business entity, term, definition, relationship, and constraint records SHALL carry stable references and provenance. Provenance SHALL identify the supporting document passage, structured catalog record, profiling run, or captured definition as applicable.

### FR-03 — Website crawling rules

1. Crawling SHALL begin with the supplied URL and follow only HTTP/HTTPS links within the configured scope.
2. The default allowed scope SHALL be the exact starting hostname; other hosts and subdomains SHALL require explicit allowlisting.
3. Crawling SHALL enforce configurable maximum depth, maximum pages, concurrency, per-host request rate, timeout, and maximum response size.
4. Recommended POC defaults: **depth = 3**, **maximum pages = 100**, **concurrent requests per host = 2**. These are defaults, not hardcoded limits.
5. The crawler SHALL obey `robots.txt` rules applicable to its user agent and SHOULD honor crawl-delay where practical.
   *Exception:* a single-page fetch (max depth 1, depth counted from 1 = the start page) is user-requested, not a crawl, and does not consult `robots.txt`, the sitemap or page links.
6. The crawler SHALL normalize URLs (fragments removed, canonicalization policy documented) and deduplicate repeated URLs.
7. The crawler SHALL deduplicate identical downloaded content using a content hash; distinct URLs with identical content SHALL not generate duplicate embeddings.
8. The crawler SHOULD discover eligible URLs from same-site links and may use a sitemap where available.
9. The crawler SHALL avoid non-content links (e.g., mailto, javascript), infinite URL patterns, and repeated query parameter variants where safe to normalize.
10. The crawler SHALL reject private/loopback/link-local IP destinations, cloud metadata endpoints, and unsafe redirects; DNS resolution and redirect targets SHALL be validated to mitigate SSRF.
11. Each page SHALL report `indexed`, `skipped`, or `failed` status with a reason and source URL.
12. A large crawl SHALL return a `job_id` and a reference to a paginated manifest, not the full page content.

### FR-04 — Document storage

1. Uploaded and downloaded original documents SHALL be stored in an application-managed local directory.
2. Normalized extracted text SHALL be saved or reproducibly reconstructible, with links back to the original source.
3. Document metadata MAY be stored in the SQLite catalog used for structured metadata.
4. The system SHALL maintain LlamaIndex node/document storage in addition to the vector store so chunk text and hierarchy can be reconstructed.
5. Index data SHALL persist across application restarts.
6. Reingesting unchanged content SHALL not produce duplicate active indexed chunks.
7. Updating a source SHALL invalidate or replace its stale indexed chunks using the source ID and content hash.
8. Deletes SHALL remove or deactivate a source and its indexed chunks consistently.

### FR-05 — Hierarchical chunking and indexing

1. The system SHALL use **LlamaIndex** for hierarchical document chunking.
2. Initial configurable target levels SHALL be **2,048 tokens (large parent)**, **1,024 tokens (small parent)**, and **256 tokens (leaf)**; final values can be tuned using retrieval evaluations.
3. Leaf chunks SHALL be embedded and placed in `SimpleVectorStore`; parent nodes SHALL remain retrievable through the document/node store.
4. Each child SHALL preserve a reference to its parent(s) and source document.
5. Chunking SHOULD preserve section headings, paragraph boundaries, and available page references. Table boundaries SHOULD be respected where extraction provides structural information.
6. Indexing SHALL capture counts of documents, nodes, embedded chunks, failures, and time spent at each stage.
7. The chunking implementation SHALL be configurable and testable independently of the MCP transport.

### FR-06 — Embeddings

1. The embedding model SHALL be **`sentence-transformers/all-mpnet-base-v2`**, executed **locally**.
2. The embedding model name, device, batch size, and normalization configuration SHALL be specified in **`config/mesh.yaml`**.
3. Both vector indexes SHALL use the local embedding model. Each index SHALL use a matching embedding configuration for indexing and retrieval.
4. The index metadata SHALL record the embedding model name, model revision when available, and vector dimensionality.
5. Changing to an incompatible embedding model SHALL require reindexing or creation of a separate index.
6. No external embedding API SHALL be required for the POC.
7. The selected model is primarily English-focused; multilingual retrieval is not an acceptance requirement for this POC.

### FR-07 — Retrieval

1. The `retrieve_documents` MCP tool SHALL search document leaf chunks and the separate index of business terms and entities by vector similarity. A single query SHALL be able to return evidence from both indexes.
2. **K SHALL default to 5 per searched index**, configured under retrieval settings in **`config/mesh.yaml`**. The caller MAY override K within configured bounds and filter by evidence type or source.
3. Document results SHALL include `chunk_id`, `document_id`, excerpt, source URL/path, available page or section, similarity score, and parent reference.
4. Glossary results SHALL include the business label, record type, similarity score, and references to the originating table or column. The tool SHALL return relevant business definitions, relationships, and constraints by looking up business concepts in the knowledge graph. Graph evidence SHALL include provenance and respect the configured result limit.
5. The orchestrator MAY expand retrieved leaf nodes to their parents within a configurable context budget.
6. Vector scores SHALL be identified as similarity scores.
7. The system SHALL return an empty evidence set or an explicit insufficient-evidence response if retrieval yields no relevant evidence.
8. Reranking and hybrid keyword retrieval remain optional POC enhancements.

### FR-08 — Search and grounded answers

1. The application SHALL expose **search mode**, returning retrieved passages, catalog references, and graph evidence as available. Search over persisted evidence SHALL work without a generative LLM.
2. The application SHOULD expose **answer mode**, generating responses from the retrieved evidence using a separately configurable LLM.
3. Generated answers SHALL cite the relevant document passages, structured metadata, or graph evidence. Citation references SHALL resolve to the supporting sources. Answers that rely on inferred constraints SHALL identify them as inferred.
4. If the retrieved evidence is insufficient, the assistant SHALL identify what the available sources leave unresolved.
5. Retrieved text and metadata SHALL be treated as untrusted data. The orchestrator SHALL follow its own instructions when processing retrieved evidence.

### FR-09 — Structured metadata extraction

1. The system SHALL expose an MCP tool to extract schema metadata from configured relational databases, Parquet files, and Databricks.
2. For each source, the system SHALL persist dataset identifiers, table or file names, column names, data types, and nullability where available in SQLite. It SHALL capture declared unique, primary, and foreign keys where the source exposes them.
3. Re-extracting metadata SHALL update the records for the same source. Each structured ingestion job SHALL report progress and failures by dataset.

### FR-10 — Key discovery

1. The system SHALL expose an MCP tool that profiles selected datasets and records discovered keys and candidate keys in the SQLite catalog.
2. The system SHALL use HyUCC and Apriori-style candidate generation and pruning to find minimal unique column combinations, including composite keys. Non-null unique combinations SHALL be eligible as primary-key candidates. [HyUCC paper](https://www.btw2017.informatik.uni-stuttgart.de/slidesandpapers/F3-13-21-short/paper_web.pdf) · [Apriori paper](https://www.vldb.org/conf/1994/P487.PDF)
3. The system SHALL use BINDER to discover single-column and composite inclusion dependencies and record foreign-key candidates that reference a declared or observed unique key. It SHALL preserve the ordered mapping of referencing to referenced columns. [BINDER paper](https://www.vldb.org/pvldb/vol8/p774-papenbrock.pdf)
4. Each result SHALL identify its columns, whether it was declared by the source or inferred from profiling, and the profiling scope for inferred results. Candidates derived from a sample SHALL be labeled with that scope.

### FR-11 — Business entities and relationships from documents

1. The system SHALL extract business entities and relationships from ingested documents and web pages. Each relationship SHALL identify its source entity, relationship type, target entity, and supporting passage.
2. The system SHALL publish these entities and relationships to the knowledge graph with links to their supporting documents and chunks.

### FR-12 — Business terms and entities from structured metadata

1. The system SHALL use the term-generation LLM configured in `config/mesh.yaml` and the template in [`prompts/term_generation.txt`](../prompts/term_generation.txt) to generate a business term for every column. It SHALL generate a business entity label for every cataloged table or equivalent Parquet dataset.
2. The system SHALL validate generated output against the source tables and columns before publishing it. The business term SHALL link to its column and the business entity to its table or dataset. When returned by the template, the term description and explanation SHALL be retained.
3. The system SHALL index these business terms and entities in a dedicated persistent LlamaIndex `SimpleVectorStore` and publish them to the knowledge graph.
4. A term-generation failure SHALL be reported against the affected dataset. Catalog extraction and key discovery SHALL remain available when the term-generation model is unconfigured or unavailable.

### FR-13 — Business definitions

1. The system SHALL let users capture and retrieve a named business definition, such as “Active Customer,” with its definition text and links to the business entities or terms it describes.
2. The system SHALL retain the definition's author or source reference and store the definition in the knowledge graph so retrieval can return it as evidence.

### FR-14 — Knowledge graph and constraint validation

1. The system SHALL use RDFLib to build and persist an RDF graph of business entities, terms, definitions, relationships, and constraints. It SHALL link business concepts to their source documents or structured catalog records.
2. The graph SHALL represent declared keys and inferred key candidates with their respective provenance. Composite keys and foreign-key relationships SHALL preserve column membership and mapping.
3. The system SHALL use pySHACL to validate graph updates against SHACL shapes before making the updated graph available to retrieval. Validation failures SHALL report the affected records and violated shapes.

## 5. MCP Tool Requirements

The MCP server(s) SHALL expose the following interfaces. The application services SHALL also be directly callable from Python for testing and FastAPI integration.

| Tool | Inputs | Outputs |
|---|---|---|
| `crawl_website` | `url`, optional `max_depth`, `max_pages` | `job_id`, status, counts, manifest reference |
| `ingest_document` | uploaded file reference or local path; optional metadata | `document_id`, status, processing job ID |
| `index_documents` | one or more document IDs; optional chunk settings | `job_id`, processed/error counts, index version |
| `read_document` | `document_id`, optional bounded page/section/chunk range | Bounded text, source metadata, references |
| `extract_structured_metadata` | configured structured source ID, optional dataset selection | `job_id`, status, counts, SQLite catalog references |
| `discover_keys` | catalog dataset IDs, optional profiling scope | `job_id`, declared and inferred key counts, profiling and catalog references |
| `retrieve_documents` | query, optional `top_k`, evidence types, and source filters | Ranked passages and glossary matches, linked graph evidence, source references |

### MCP operational requirements

- Tool responses SHALL be structured and bounded by size.
- Large results SHALL be materialized in storage and returned by ID/URI reference, not inserted whole into the LLM context.
- Long-running crawling, indexing, metadata extraction, and key discovery jobs SHALL expose status and error details.
- Each tool call SHALL propagate trace/correlation IDs where supported.
- The server SHALL validate tool parameters, restrict local paths to authorized roots, and enforce authorization on document, catalog, and graph references.
- Tools SHALL return structured errors for invalid arguments, unavailable sources or records, and failed operations.

**Example `retrieve_documents` response**

```json
{
  "query": "How do we define an active customer, and how is a customer identified?",
  "top_k": 5,
  "results": [
    {
      "rank": 1,
      "document_id": "doc-1001",
      "chunk_id": "chunk-0081",
      "parent_id": "node-0020",
      "source_uri": "/documents/customer.pdf",
      "page": 12,
      "section": "Customer Identification",
      "score": 0.82,
      "text": "Assign each customer a unique identifier."
    }
  ],
  "glossary_results": [
    {
      "rank": 1,
      "type": "business_term",
      "label": "Customer Identifier",
      "score": 0.86,
      "catalog_ref": "catalog:crm/customers/customer_id",
      "graph_ref": "urn:mesh:term:customer-identifier"
    }
  ],
  "graph_evidence": [
    {
      "type": "business_definition",
      "label": "Active Customer",
      "text": "A customer with a completed order in the past 90 days.",
      "graph_ref": "urn:mesh:definition:active-customer",
      "source_ref": "/api/business-definitions/active-customer"
    }
  ],
  "trace_id": "example-trace-id"
}
```

The identifiers, definition, and scores are illustrative.

## 6. Configuration Requirements

The repository SHALL contain **`config/mesh.yaml`** for storage, connectors, profiling, indexing, graph validation, and both LLM settings. The application SHALL load and validate it at startup. The following example illustrates the configuration.

```yaml
storage:
  originals_dir: ./data/originals
  extracted_dir: ./data/extracted
  metadata_db: ./data/metadata.sqlite

crawler:
  allowed_scope: hostname
  max_depth: 3
  max_pages: 100
  concurrency_per_host: 2
  request_timeout_seconds: 20
  max_response_bytes: 10485760
  respect_robots_txt: true
  deduplicate_urls: true
  deduplicate_content_sha256: true

chunking:
  library: llamaindex
  strategy: hierarchical
  chunk_sizes_tokens: [2048, 1024, 256]

embedding:
  provider: sentence_transformers
  model: sentence-transformers/all-mpnet-base-v2
  device: cpu
  batch_size: 32
  normalize: true

vector_store:
  provider: llamaindex_simple
  document_persist_dir: ./data/index/documents
  glossary_persist_dir: ./data/index/glossary

structured_sources:
  - id: crm
    kind: database
    connection_url_env: CRM_DATABASE_URL
  - id: customer_export
    kind: parquet
    path: ./data/sources/customers.parquet
  - id: orders
    kind: databricks
    server_hostname_env: DATABRICKS_SERVER_HOSTNAME
    http_path_env: DATABRICKS_HTTP_PATH
    token_env: DATABRICKS_TOKEN
    catalog: sales
    schema: public

knowledge_graph:
  data_path: ./data/knowledge_graph.ttl
  shapes_path: ./config/knowledge_graph_shapes.ttl

retrieval:
  top_k: 5
  max_graph_results: 10
  max_top_k: 20
  expand_parent_context: true
  max_context_tokens: 6000
  reranker_enabled: false

term_generation:
  provider: null  # Configure to enable business term and entity generation.
  model: null
  prompt_path: ./prompts/term_generation.txt

answer_generation:
  enabled: false  # Enable after selecting and configuring an LLM.
  provider: null
  model: null

observability:
  enabled: true
  service_name: knowledge-mesh
  otlp_endpoint: http://localhost:4317
  capture_sensitive_content: false
  phoenix_enabled: true
  prometheus_enabled: true
  loki_enabled: true
  ragas_offline_evaluation: true
```

Source configuration SHALL reference connector credentials through environment variables or a secret store. The application SHALL document how to configure profiling scope and limits. It SHALL document which settings require a restart, reindexing, or regeneration and which take effect on subsequent requests.

## 7. FastAPI Application and UI

The application SHALL offer a lightweight browser interface for document uploads, crawl submissions, structured cataloging and key discovery, job status, business definition capture, and search. Users SHALL be able to view configured structured sources, inspect cataloged datasets and keys, and follow citations from retrieved documents, metadata, and graph evidence.

| Method | Route | Function |
|---|---|---|
| `POST` | `/api/documents` | Upload document |
| `GET` | `/api/documents` | List documents |
| `GET` | `/api/documents/{id}` | Document metadata/status |
| `POST` | `/api/crawl` | Submit website crawl |
| `GET` | `/api/structured-sources` | List configured structured sources |
| `POST` | `/api/structured-sources/{id}/catalog` | Catalog selected datasets and queue key discovery and configured term generation |
| `GET` | `/api/datasets/{id}` | Inspect schema, keys, and linked business metadata |
| `POST` | `/api/business-definitions` | Capture a business definition |
| `GET` | `/api/business-definitions/{id}` | Read a captured definition |
| `GET` | `/api/jobs/{id}` | Job progress and failures |
| `POST` | `/api/search` | Retrieve ranked passages and glossary matches with linked graph evidence |
| `POST` | `/api/chat` | Generate grounded answer (when LLM configured) |
| `GET` | `/api/health` | Readiness/health status |

- The frontend MAY be plain HTML/CSS/JavaScript served by FastAPI.
- API errors SHALL use consistent HTTP status codes and machine-readable error bodies.
- API responses SHALL expose trace IDs for correlation with telemetry.
- Uploads SHALL enforce file type and size limits and prevent unsafe path traversal.

## 8. Observability Requirements

### Required stack

- **OpenTelemetry:** Standard trace, metric, and log instrumentation/exports.
- **Arize Phoenix:** AI/RAG trace inspection and evaluations.
- **Prometheus + Grafana:** Time-series metrics, infrastructure metrics, and dashboards.
- **Loki:** Centralized structured application logs accessible through Grafana.
- **Ragas:** Offline RAG evaluation using labeled datasets.

### Ten observability dimensions

| Dimension | Required capture / behavior |
|---|---|
| 1. Execution tracing | Steps, dependencies, spans, tool execution order, trace IDs |
| 2. LLM observability | Model, requests/responses where permitted, tokens, latency, failures |
| 3. Tool observability | MCP tool names, sanitized parameters, results/references, retries, failures |
| 4. Context observability | Prompt template version, context size, retrieved evidence IDs, pruning events; redact sensitive text |
| 5. RAG observability | Query, document and chunk IDs, glossary match IDs, graph evidence IDs, ranks, similarity scores, parent expansion, and grounding references |
| 6. Planning/decisions | Observable task plans, selected actions, changes and tool choices; do not require hidden model reasoning |
| 7. Quality/evaluation | Retrieval Recall@K, ranking metrics, faithfulness, answer relevance, test results |
| 8. Performance/cost | End-to-end and stage latency, CPU/RAM, throughput, tokens, estimated cost |
| 9. Security/guardrails | ACL decisions when applicable, suspicious document instructions, blocked URLs, sensitive-data leakage signals |
| 10. Reliability/operations | Errors, retries, timeouts, failed jobs, recovery behavior, feedback, alerts |

### Instrumentation requirements

1. Trace identifiers SHALL correlate FastAPI requests, orchestrator execution, MCP calls, document processing, metadata extraction, key discovery, term generation, graph validation, retrieval, and optional answer generation.
2. Latency and failures SHALL be observable for each stage. Metrics SHALL include document, chunk, and dataset counts, discovered-key counts, graph validation failures, and indexing throughput.
3. Logs SHALL be structured. Full private documents, prompts, connector credentials, and raw profiling values SHALL be excluded from default logs.
4. Evaluation SHALL run at meaningful quality checkpoints, **not necessarily as a model-graded evaluation of every processing step or every production request**.
5. Ragas evaluations SHALL be runnable independently on a repeatable test set, with scores linked to relevant run/version metadata.
6. Prometheus/Grafana dashboards SHOULD include request rates/errors, latency distributions, ingestion throughput, failure counts, indexing duration, and CPU/memory utilization.
7. Local POC deployments MAY run supporting observability services using Docker Compose.

## 9. Security and Reliability

- Treat URL contents, uploaded files, structured metadata, captured definitions, and model outputs as untrusted inputs.
- Enforce upload size limits, allowed file types, local storage path restrictions, and crawl URL allowlists/boundaries.
- Mitigate SSRF for initial URLs, DNS resolution, and redirects.
- Store original files in non-executable locations and sanitize display output.
- Validate tool arguments and return bounded results. Structured connectors SHALL use source accounts with read-only access and restrict access to configured datasets. Profiling SHALL keep raw row values within the profiling process and expose key findings through catalog references.
- Term generation SHALL use catalog metadata as input and exclude connector secrets and raw row values.
- Provide a configurable retry policy with backoff and clear terminal failure states.
- Ingestion retries SHALL be idempotent across the catalog, both vector indexes, and the graph. A successful refresh or deletion SHALL retire stale records from the affected source while preserving evidence from other sources.
- Protect the local document store, catalog, graph, and telemetry endpoints from unauthorized access.

## 10. Acceptance Criteria

The POC SHALL pass the following tests:

1. **File ingestion:** Upload and index representative PDF, DOCX, and TXT documents; inspect ingestion status.
2. **URL ingestion:** Crawl a test website with at least 50 reachable pages within the configured scope; do not exceed the configured depth/page limits.
3. **Deduplication:** Re-crawling the same unchanged content produces no duplicate active indexed chunks.
4. **Persistence:** Restart the application and retrieve previously indexed passages, catalog records, glossary matches, and graph evidence without rerunning ingestion.
5. **Hierarchical nodes:** Demonstrate parent/child links and retrieval of a leaf passage with its contextual parent.
6. **Embedding configuration:** Confirm that local `sentence-transformers/all-mpnet-base-v2` is used as specified in `config/mesh.yaml` and no embedding API calls are required.
7. **Top-K configuration:** Verify that document and glossary searches each return up to five matches by default and honor an allowed override.
8. **Document retrieval quality:** Achieve **Recall@5 >= 80%** on an initial labeled test set of at least 50 questions. Record the corpus and test methodology.
9. **Source citations:** Verify that document citations resolve to supported pages or sections, structured citations resolve to the originating dataset or column, and graph citations resolve to supporting evidence. Generated answers cite the retrieved evidence when enabled.
10. **Insufficient evidence:** Queries without evidence do not fabricate retrieved sources or confident answers.
11. **MCP context isolation:** Verify that crawling, indexing, and structured profiling tools return references and bounded summaries.
12. **Observability:** Inspect correlated traces, metrics, and logs for document ingestion, structured cataloging, key discovery, graph validation, and mixed retrieval, including one intentionally failing request.
13. **Crawl safety:** Reject an unauthorized off-scope link and a blocked private-network URL.
14. **User experience:** From a browser, upload a file, submit a URL, start a structured catalog job, inspect its results, capture a definition, and follow source references from search.
15. **Answer mode (conditional):** Once an answer model is configured, demonstrate a grounded answer that cites both document and structured or graph evidence and an explicit insufficient-evidence response to an unsupported question.
16. **Structured catalog:** Extract metadata from a relational database, a Parquet file, and Databricks. Verify that schemas and available declared keys are stored in SQLite with stable source references.
17. **Key discovery:** Use fixtures containing a single-column unique key, a composite unique key, and single-column and composite foreign-key candidates. Verify the expected findings, column mappings, and the distinction between declared keys and inferred candidates.
18. **Document entities and relationships:** Ingest a document describing a relationship between two business entities. Verify that both entities and their relationship appear in the graph with references to the supporting passage.
19. **Business terminology:** With the term-generation model configured, generate one term for each column and an entity for each table in a fixture. Verify that a similar phrase retrieves the expected term or entity with its physical schema reference.
20. **Definitions and graph validation:** Capture a definition such as “Active Customer” and verify that a query retrieves it through its linked business concept. Verify that a graph update violating a SHACL shape reports the violation and leaves the previously published evidence available.
21. **Mixed search:** Run a query with support in both a document passage and structured metadata. Verify that the same search returns the passage, glossary matches, and relevant graph evidence with resolvable provenance.
22. **Structured refresh:** Retry an unchanged structured ingestion and verify that catalog records, glossary vectors, and graph assertions remain unique. Refresh a changed schema and verify that stale evidence is retired.

## 11. Suggested Implementation Sequence

**Phase 1 — Document search:** Build the FastAPI UI, local and website ingestion, local storage, hierarchical chunking, local embeddings, the persistent document index, and MCP retrieval.

**Phase 2 — Structured catalog and business glossary:** Add source configuration and connectors, MCP metadata extraction and key discovery, the SQLite catalog, LLM term and entity generation, and the persistent glossary index.

**Phase 3 — Knowledge graph and mixed retrieval:** Extract entities and relationships from documents, add business definition capture, publish the RDF graph with SHACL validation, and retrieve evidence across both indexes and the graph. Add optional grounded answer generation.

**Phase 4 — End-to-end validation:** Exercise the acceptance criteria and complete Phoenix tracing, Prometheus/Grafana dashboards, Loki logging, and Ragas evaluation. Instrument each phase as its features are added.

## 12. Remaining Implementation Decisions

- Select the first relational database connector and the credential mechanism for each source type.
- Define profiling limits and null handling, and decide how HyUCC and Apriori-style discovery will be combined.
- Select the term-generation model and define how the table-level entity label will be produced alongside the prompt's per-column output.
- Select the document entity and relationship extraction method, the rules for matching concepts across sources, and the initial SHACL shapes.
- Select a separately configurable answer-generation model if answer mode is enabled.

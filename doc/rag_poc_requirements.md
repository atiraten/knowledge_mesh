# RAG Document Search POC — Requirements Specification

**Version:** 1.0  
**Date:** 2026-11-19  
**Status:** Ready for POC implementation, subject to configuration of the answer-generation model

## 1. Purpose

Build a Python-based web chatbot that ingests documents and websites, indexes document sections, retrieves relevant evidence for user questions, and optionally generates grounded answers with source citations. The POC shall use Model Context Protocol (MCP) tools, LlamaIndex hierarchical chunking, local embeddings, and open-source observability.

## 2. Scope

### In scope

- User-uploaded **PDF**, **TXT**, and common document files (initially **DOCX**), plus HTML pages retrieved from URLs.
- Ingestion from local files and local directories.
- Crawling websites starting from a user-supplied HTTP/HTTPS URL.
- Local storage of original content and extracted text.
- Hierarchical chunking and vector indexing with LlamaIndex.
- Local embedding inference using `sentence-transformers/all-mpnet-base-v2`.
- LlamaIndex `SimpleVectorStore` with persistent storage across restarts.
- Top-K semantic retrieval, with default K = 5.
- MCP tools for crawling, ingestion, indexing, document reading, and retrieval.
- FastAPI application with a simple browser interface for uploads, URL submission, search, and chat.
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
RAG Harness / Orchestrator
    |-- MCP: crawl_website / ingest_document
    |-- MCP: index_documents
    |-- MCP: read_document
    |-- MCP: retrieve_context
    |-- Search mode: return relevant sections and citations
    `-- Answer mode: invoke configurable LLM on retrieved evidence
                 |
                 v
Storage: originals + extracted text + document metadata +
         LlamaIndex node/document store + SimpleVectorStore

Observability across all layers:
OpenTelemetry -> Phoenix (AI traces), Prometheus/Grafana (metrics),
                 Loki (logs); Ragas (offline evaluation)
```

The harness SHALL NOT put full crawled pages or full documents into an LLM context by default. Tools shall return durable document/job references and bounded content selections.

## 4. Functional Requirements

### FR-01 — Supported inputs

1. The system SHALL ingest local **PDF, DOCX, TXT**, and HTML content from URLs.
2. The system SHALL accept document uploads through the web UI.
3. The system SHALL ingest supported files from a configured local directory.
4. The system SHALL accept a starting HTTP/HTTPS URL and crawl eligible pages on that website.
5. Unsupported formats SHALL be rejected or marked as skipped with a recorded reason.
6. Parsing failures on one document SHALL NOT terminate the entire ingestion job.

### FR-02 — Source identity and provenance

1. Every ingested source SHALL receive a stable `document_id`.
2. Metadata SHALL include: source type, original URL/path, title/filename, MIME type, content hash, ingestion timestamp, processing status, and optional page/section details.
3. Every indexed chunk SHALL reference its source `document_id`, its own `chunk_id`, and its parent node ID if applicable.
4. Citations SHALL resolve to the original URL or stored document and available page/section location.

### FR-03 — Website crawling rules

1. Crawling SHALL begin with the supplied URL and follow only HTTP/HTTPS links within the configured scope.
2. The default allowed scope SHALL be the exact starting hostname; other hosts and subdomains SHALL require explicit allowlisting.
3. Crawling SHALL enforce configurable maximum depth, maximum pages, concurrency, per-host request rate, timeout, and maximum response size.
4. Recommended POC defaults: **depth = 3**, **maximum pages = 100**, **concurrent requests per host = 2**. These are defaults, not hardcoded limits.
5. The crawler SHALL obey `robots.txt` rules applicable to its user agent and SHOULD honor crawl-delay where practical.
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
3. The metadata catalog MAY use SQLite for the POC.
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
3. Indexing and retrieval SHALL use the same embedding model configuration.
4. The index metadata SHALL record the embedding model name, model revision when available, and vector dimensionality.
5. Changing to an incompatible embedding model SHALL require reindexing or creation of a separate index.
6. No external embedding API SHALL be required for the POC.
7. The selected model is primarily English-focused; multilingual retrieval is not an acceptance requirement for this POC.

### FR-07 — Retrieval

1. The `retrieve_context` MCP tool SHALL accept a query and retrieve the **top K** relevant leaf chunks using vector similarity.
2. **K SHALL default to 5**, configured under retrieval settings in **`config/mesh.yaml`**.
3. The caller MAY override K within application-defined maximum bounds; when omitted, the configured default SHALL apply.
4. The tool SHALL return ranked sections containing at least `chunk_id`, `document_id`, excerpt text, source URL/path, available page/section, similarity score, and parent reference.
5. The harness MAY expand retrieved leaf nodes to their parents before answer generation, with a configurable context budget.
6. Scores SHALL be identified as retrieval similarity scores, not calibrated probabilities.
7. The system SHALL return an empty evidence set or an explicit insufficient-evidence response when no relevant passages are found.
8. Reranking and hybrid keyword retrieval are optional POC enhancements, not mandatory requirements.

### FR-08 — Search and grounded answers

1. The application SHALL expose **search mode**, returning relevant retrieved sections without requiring a generative LLM.
2. The application SHOULD expose **answer mode**, generating a response based on retrieved sections using a configurable LLM.
3. The answer-generation provider/model SHALL be configured separately; **an LLM choice has not yet been specified**.
4. Generated answers SHALL cite actual source documents and sections; citation references SHALL be resolvable.
5. If retrieved evidence is insufficient, the assistant SHALL explicitly state that the documents do not support a confident answer.
6. Retrieved website content SHALL be treated as untrusted data, never as instructions overriding the harness's rules.

## 5. MCP Tool Requirements

The MCP server(s) SHALL expose the following interfaces. The application services SHALL also be directly callable from Python for testing and FastAPI integration.

| Tool | Inputs | Outputs |
|---|---|---|
| `crawl_website` | `url`, optional `max_depth`, `max_pages` | `job_id`, status, counts, manifest reference |
| `ingest_document` | uploaded file reference or local path; optional metadata | `document_id`, status, processing job ID |
| `index_documents` | one or more document IDs; optional chunk settings | `job_id`, processed/error counts, index version |
| `read_document` | `document_id`, optional bounded page/section/chunk range | Bounded text, source metadata, references |
| `retrieve_context` | query, optional `top_k`, optional source filters | Ranked passages with scores and source references |

### MCP operational requirements

- Tool responses SHALL be structured and bounded by size.
- Large results SHALL be materialized in storage and returned by ID/URI reference, not inserted whole into the LLM context.
- Long-running crawl/index operations SHALL expose job status and error details.
- Each tool call SHALL propagate trace/correlation IDs where supported.
- Tool parameters SHALL be validated; filesystem paths SHALL be restricted to authorized roots and document references SHALL enforce authorization.
- Tools SHALL return explicit, structured errors for invalid arguments, unavailable documents, and failed operations.

**Example `retrieve_context` response**

```json
{
  "query": "How are customer records identified?",
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
      "text": "An illustrative retrieved excerpt."
    }
  ],
  "trace_id": "example-trace-id"
}
```

All values in this sample are illustrative.

## 6. Configuration Requirements

The repository SHALL contain **`config/mesh.yaml`**, and the application SHALL load and validate it at startup. The following is a proposed initial configuration, not a requirement to use every field verbatim:

```yaml
storage:
  originals_dir: ./data/originals
  extracted_dir: ./data/extracted
  index_dir: ./data/index
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
  persist_dir: ./data/index

retrieval:
  top_k: 5
  max_top_k: 20
  expand_parent_context: true
  max_context_tokens: 6000
  reranker_enabled: false

answer_generation:
  enabled: false  # Enable after selecting and configuring an LLM.
  provider: null
  model: null

observability:
  enabled: true
  service_name: rag-poc
  otlp_endpoint: http://localhost:4317
  capture_sensitive_content: false
  phoenix_enabled: true
  prometheus_enabled: true
  loki_enabled: true
  ragas_offline_evaluation: true
```

The application SHALL define whether changed settings require restart, reindexing, or only affect subsequent requests.

## 7. FastAPI Application and UI

The application SHALL offer a lightweight browser page to upload documents, submit a URL for crawling, view indexing/job status, enter a question, and inspect retrieved sections and source citations.

| Method | Route | Function |
|---|---|---|
| `POST` | `/api/documents` | Upload document |
| `GET` | `/api/documents` | List documents |
| `GET` | `/api/documents/{id}` | Document metadata/status |
| `POST` | `/api/crawl` | Submit website crawl |
| `GET` | `/api/jobs/{id}` | Job progress and failures |
| `POST` | `/api/search` | Retrieve top-K relevant sections |
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
| 5. RAG observability | Query, document/chunk IDs, ranks, similarity scores, parent expansion, grounding references |
| 6. Planning/decisions | Observable task plans, selected actions, changes and tool choices; do not require hidden model reasoning |
| 7. Quality/evaluation | Retrieval Recall@K, ranking metrics, faithfulness, answer relevance, test results |
| 8. Performance/cost | End-to-end and stage latency, CPU/RAM, throughput, tokens, estimated cost |
| 9. Security/guardrails | ACL decisions when applicable, suspicious document instructions, blocked URLs, sensitive-data leakage signals |
| 10. Reliability/operations | Errors, retries, timeouts, failed jobs, recovery behavior, feedback, alerts |

### Instrumentation requirements

1. Trace identifiers SHALL correlate FastAPI requests, harness execution, MCP tool calls, parsing, chunking, embeddings, vector search, and optional LLM generation.
2. Latency, error count, document count, chunk count, and indexing throughput SHALL be observable.
3. Log entries SHALL be structured and SHALL NOT log full private documents or prompts by default.
4. Evaluation SHALL run at meaningful quality checkpoints, **not necessarily as a model-graded evaluation of every processing step or every production request**.
5. Ragas evaluations SHALL be runnable independently on a repeatable test set, with scores linked to relevant run/version metadata.
6. Prometheus/Grafana dashboards SHOULD include request rates/errors, latency distributions, ingestion throughput, failure counts, indexing duration, and CPU/memory utilization.
7. Local POC deployments MAY run supporting observability services using Docker Compose.

## 9. Security and Reliability

- Treat URL contents, uploaded files, and retrieved passages as untrusted inputs.
- Enforce upload size limits, allowed file types, local storage path restrictions, and crawl URL allowlists/boundaries.
- Mitigate SSRF for initial URLs, DNS resolution, and redirects.
- Store original files in non-executable locations and sanitize display output.
- Validate tool arguments and return bounded results.
- Provide a configurable retry policy with backoff and clear terminal failure states.
- Avoid duplicate active vectors on retries and restarts (idempotent indexing behavior).
- Do not expose the local document store or telemetry endpoints publicly without appropriate protection.

## 10. Acceptance Criteria

The POC SHALL pass the following tests:

1. **File ingestion:** Upload and index representative PDF, DOCX, and TXT documents; inspect ingestion status.
2. **URL ingestion:** Crawl a test website with at least 50 reachable pages within the configured scope; do not exceed the configured depth/page limits.
3. **Deduplication:** Re-crawling the same unchanged content produces no duplicate active indexed chunks.
4. **Persistence:** Restart the application and retrieve previously indexed content without rerunning ingestion.
5. **Hierarchical nodes:** Demonstrate parent/child links and retrieval of a leaf passage with its contextual parent.
6. **Embedding configuration:** Confirm that local `sentence-transformers/all-mpnet-base-v2` is used as specified in `config/mesh.yaml` and no embedding API calls are required.
7. **Top-K configuration:** Verify that retrieval returns up to five passages by default and honors an allowed override; a smaller result set is valid if fewer passages exist.
8. **Retrieval quality:** Achieve **Recall@5 >= 80%** on an initial labeled test set of at least 50 questions; record the corpus and test methodology.
9. **Source citations:** Search results carry source ID/URI and page or section information where available; generated answers (when enabled) cite retrieved evidence.
10. **Insufficient evidence:** Queries without evidence do not fabricate retrieved sources or confident answers.
11. **MCP context isolation:** Crawling/indexing tools return references and bounded summaries rather than full page/document bodies.
12. **Observability:** Inspect a correlated end-to-end trace and relevant metrics/logs for ingestion and retrieval, including one intentionally failing request.
13. **Crawl safety:** Reject an unauthorized off-scope link and a blocked private-network URL.
14. **User experience:** From a browser, upload a file, submit a URL, check job status, search, and open or inspect source references.
15. **Answer mode (conditional):** Once a generation model is configured, demonstrate a grounded answer with citations and a no-evidence abstention.

## 11. Suggested Implementation Sequence

**Phase 1 — Core search:** FastAPI UI, local upload, local storage/catalog, LlamaIndex hierarchical parser, local embeddings, persistent SimpleVectorStore + node store, and the MCP retrieval tool.

**Phase 2 — Website and RAG:** Safe bounded website crawler, content deduplication, index lifecycle, MCP job/reference handling, configurable answer-generation model, grounded answers, citations.

**Phase 3 — Observability and validation:** OpenTelemetry instrumentation, Phoenix tracing, Prometheus/Grafana/Loki, Ragas benchmark suite, safety tests, and operational dashboards. Instrument from phase 1 so phase 3 adds reporting and completeness rather than retrofitting everything.

## 12. Remaining Implementation Decision

The only major unresolved product choice is the **answer-generation LLM**. Search mode can be completed without one; chat answer mode requires selecting a local or hosted model and configuring it. The choice SHALL remain pluggable and SHALL NOT affect the selected embedding model or indexed data model.

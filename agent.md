# Knowledge Mesh

## Authoritative specification

Read the [Knowledge Mesh requirements](doc/knowledge_mesh_requirements.md) before implementing or modifying functionality. Use its acceptance criteria to verify changes and surface unresolved decisions or assumptions.

## Mandatory architecture

- Use Python and FastAPI for the browser UI and API. Expose MCP tools for website crawling, document ingestion, indexing, document reading, structured metadata extraction, key discovery, and retrieval. Keep the underlying services callable from Python.
- Use LlamaIndex hierarchical chunking and retain parent nodes in the document store. Persist separate LlamaIndex `SimpleVectorStore` indexes for document passages and the business glossary. Use local `sentence-transformers/all-mpnet-base-v2` embeddings for both.
- Store structured metadata from relational databases, Parquet files, and Databricks in SQLite. Preserve declared unique, primary, and foreign keys. Use HyUCC and Apriori-style discovery for single-column and composite unique keys and BINDER for foreign-key candidates. Record the profiling scope and distinguish inferred candidates from declared keys. Profile the datasets and store the results in SQLite table. The profile shall contain for each column: dataset row count, total distinct values, total empty or blank, total null, data format (N for numeric, T for text, keey special characters. For instance 1999-12-22 --> NNNN-NN-NN, PAUL MERSON --> TTTT TTTTTT), data content type, and most common values.
- Extract business entities and relationships from documents and web pages. Use the configured term-generation LLM and [term-generation prompt](prompts/term_generation.txt) to generate a business term for each column and an entity label for each table or equivalent dataset.
- Use RDFLib to persist business entities, terms, captured definitions, relationships, and constraints with provenance. Validate graph updates with pySHACL before publishing them.
- Retrieve document passages and glossary matches in one search, with up to five results per index by default and bounded evidence from the knowledge graph. Search must work without a generative LLM. When answer mode is enabled, cite the sources that support each answer.
- Load and validate settings from `config/mesh.yaml`. Configure term generation and answer generation independently. Keep connector credentials in environment variables or a secret store.
- Instrument requests and jobs with OpenTelemetry. Use Phoenix for AI traces, Prometheus and Grafana for metrics, Loki for logs, and Ragas for offline evaluation.

## Engineering rules

- Implement requirements incrementally and write automated tests for each component.
- Preserve stable document, chunk, dataset, and column identifiers. Keep citations and graph assertions traceable to document passages, catalog records, profiling runs, or captured definitions.
- Return bounded tool results and use durable references for large outputs. Keep full crawled documents out of LLM context and raw profiling values out of retrieval, prompts, and logs.
- Use read-only structured source connections and enforce authorized source and path boundaries. Treat crawled content, metadata, captured definitions, and model outputs as untrusted inputs.
- Make ingestion retries idempotent. Retire stale records on successful refresh or deletion across the catalog, vector indexes, and graph while preserving evidence from other sources.
- Keep configurable settings in `config/mesh.yaml` and propagate trace IDs through application and MCP calls.

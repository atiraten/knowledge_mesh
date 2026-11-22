# Knowledge Graph + GraphRAG + MCP Retrieval — Implementation Requirements

**Version:** 1.1  
**Date:** 2026-11-21  
**Status:** Implemented in the POC (version 1.0 of 2026-11-20 was the specification before implementation)

The keywords SHALL, SHALL NOT, SHOULD and MAY have their RFC 2119 meaning. Section 12 lists the changes from
version 1.0.

## 1. Purpose and scope

**1.1** The system SHALL use an RDF-based knowledge graph to represent business concepts and their mapping to
structured data, and SHALL support graph-aware retrieval for user questions.

**1.2** The knowledge graph SHALL support:

- Business Entities such as Customer, Order, Order Line and Product.
- Business Terms such as Customer Identifier, Transaction Date and Settlement Date.
- Relationships between Business Entities, including multi-hop paths through bridge entities.
- Relationships between Business Terms (e.g., Settlement Date is expected to follow Transaction Date).
- Physical metadata (tables, columns, primary keys, foreign keys, including composite keys).
- Semantic-to-physical mappings, definitions, synonyms, confidence, provenance and validation status.
- Links to relevant document evidence, when available.

**1.3** The solution SHALL use RDFLib 7.6.0 for RDF graph representation and retrieval. RDF, RDFS, OWL, SKOS and
SHACL MAY be used as appropriate to encode semantics, glossary concepts and validation rules. SHACL execution, if
required, SHALL use a compatible validation implementation (e.g., pySHACL); RDFLib alone SHALL NOT be assumed to
execute SHACL.

**1.4** The MCP server SHALL publish the tools specified in Section 3 for knowledge graph retrieval, document
retrieval, catalog and business-concept lookup, SQL generation and execution, and HTML generation.

**1.5** The system SHALL use the existing planner and agent harness architecture. The planner returns structured
intents and retrieval queries. The harness SHALL use the MCP server's tools only, called through the MCP protocol
(Section 5.3); new tools SHALL integrate through the existing MCP publication. The harness SHALL support the
clarification flow defined in Section 5.11.

**1.6** This specification covers graph and document retrieval, database record retrieval, HTML output and catalog
navigation. It SHALL NOT require loading or copying all underlying customer/order/product records into the RDF
graph, nor replacing the existing SQL executor, vector store, document ingestion pipeline or agent loop.

**1.7** The system SHALL keep its existing catalog behavior: it scans the databases configured in application
settings and catalogs their metadata.

## 2. RDF knowledge graph model

**2.1** The graph SHALL differentiate business semantics from physical metadata and their mappings. Logical named
graphs SHOULD include:

1. **semantic**: Business Entities, Business Terms and relationships.
2. **physical**: tables, columns, keys and FK column mappings.
3. **mapping**: semantic-to-physical correspondences and relationship-to-join mappings.
4. **documents**: document/chunk references and semantic links, if documents are indexed.
5. **constraints**: SHACL shapes and/or business rule references.

**2.2** A Business Entity SHALL have a stable unique IRI, a name, a definition (when known), alternative names
(when known) and a classification. An OWL class is the preferred conceptual representation for an entity type.

**2.3** A Business Term SHALL have a stable unique IRI, a preferred name, a definition (when known), alternate
labels/synonyms and a link to its owning Business Entity. A SKOS concept SHALL represent the governed term. When the
term denotes a semantic attribute, it SHOULD also be associated with a separate OWL datatype or object property, as
appropriate. The SKOS term and the OWL property SHALL NOT be conflated.

**2.4** Business Entity relationships SHALL be named, directional semantic relationships, such as
`Customer --placesOrder--> Order`, `Order --containsOrderLine--> OrderLine` and
`OrderLine --referencesProduct--> Product`. They SHALL have source and target entities and MAY include cardinality,
business conditions, validity, confidence and provenance.

**2.5** Entity relationships SHALL also be first-class metadata objects, so that they can be mapped to one or more
physical FK/join definitions. An OWL object property MAY describe the semantic predicate. Schema-level navigational
edges MAY be materialized for efficient graph traversal, but SHALL be distinguished from record-level factual
assertions.

**2.6** Physical tables SHALL include a table IRI, a name, source/system identification where available, and
membership relationships to their columns.

**2.7** Physical columns SHALL include a column IRI, a name, the owning table and the data type when known. In the
existing catalog, each physical table SHALL be linked to one or more Business Entities, and each physical column
SHALL be linked to one or more Business Terms in the glossary. Multiple physical tables MAY link to the same
Business Entity, and multiple physical columns MAY link to the same Business Term. Such mappings SHALL support
confidence and provenance.

**2.8** Primary keys, foreign keys and unique keys SHALL be modeled as explicit resources. Foreign keys SHALL
contain ordered pairs of source and target columns so that composite-key joins are represented correctly. Joins
SHALL NOT be inferred from column-name similarity alone.

**2.9** The graph SHALL support semantic relationships between Business Terms such as `expectedAfter`,
`derivedFrom`, `equivalentTo`, `dependsOn` and `isPartOf`, where these meanings have been explicitly defined.
Descriptive term-to-term relations SHALL NOT automatically be interpreted as validated record-level facts.

**2.10** Validation constraints MAY be represented using SHACL, including date comparison rules or required
attributes. Rule exceptions and the applicable entity scope SHALL be represented when known.

**2.11** The graph SHOULD record source/provenance, confidence, status (inferred / verified / rejected) and
timestamps for automatically discovered relationships. Unverified inferred FK relationships SHALL NOT be silently
presented as authoritative business semantics.

**2.12** Document chunks MAY be connected to the Business Entities, Business Terms and relationships they describe.
Embeddings and full document content MAY remain in the existing document/vector systems; the graph SHOULD store
stable identifiers/references instead of duplicating vectors and content.

## 3. MCP server requirements

### Published tools and connection

**3.1** The MCP server SHALL publish the following tools:

- `query_knowledge_graph` (new): knowledge graph retrieval.
- `retrieve_documents` (existing): document retrieval. It keeps its name and implementation; its `queries`
  parameter is a list (approved change, Section 3.17).
- `find_datasets`, `get_dataset`, `get_business_term`, `get_business_entity`, `get_business_definition` (new):
  catalog and business-concept lookup (Section 3.19).
- `generate_sql` (new): one read-only SQL query written by an LLM from the evidence passed (Section 3.20).
- `execute_sql` (new): authorized database queries.
- `generate_html` (new): HTML content for a web page.

The existing ingestion and catalog tools (`crawl_website`, `ingest_document`, `index_documents`, `read_document`,
`extract_structured_metadata`, `discover_keys`, `get_job`) stay published.

**3.2** The harness SHALL connect to the MCP server in one of two modes, set by `mcp.mode` in `config/mesh.yaml`:

- `in_process` (default): an in-memory MCP client session to the MCP server in the same process.
- `streamable_http`: the MCP server is mounted at `/mcp` in the FastAPI process (one copy of the data, indexes,
  graph and jobs). The harness connects to `mcp.url` (default `http://127.0.0.1:8000/mcp`); outside MCP clients MAY
  use the same endpoint. When an API key is configured (`api.api_key_env`), `/mcp` SHALL require the same key as
  `/api`, and the harness SHALL send it.

### query_knowledge_graph

**3.3** The tool name SHALL be `query_knowledge_graph`. Its public schema SHALL be explicit and documented. It SHALL
return structured, machine-readable results suitable for use by the existing executor LLM.

**3.4** The tool SHALL accept retrieval queries derived from a planner intent. Recommended request schema:

```json
{
  "intent_id": "I1",
  "intent": "list customer born before 1990",
  "retrieval_queries": [
    "customer born before 1990",
    "customers birth date before 1990"
  ],
  "max_hops": 5,
  "top_k": 5,
  "include_physical_mappings": true,
  "include_related_terms": true
}
```

- `intent_id`: optional correlation identifier; older clients may omit it.
- `intent`: natural-language business objective.
- `retrieval_queries`: list of alternative search formulations from the planner; SHOULD be nonempty (1–5).
- `max_hops`: maximum semantic-relationship traversals (configured default 5, at most
  `knowledge_graph.max_hops_limit`, default 8).
- `top_k`: maximum ranked retrieval hits/paths (configured default 5, at most `knowledge_graph.max_top_k`,
  default 20).
- `include_physical_mappings` / `include_related_terms`: optional retrieval flags.
- `structured_source_ids`, `dataset_ids`, `document_ids`: optional scope filters; they only narrow the result.
- `view`: `full` (default, the documented output) or `agent` (the compact shape used by the search agent,
  Section 3.18).

**3.5** The tool SHALL resolve text to candidate Business Entities, Business Terms, synonyms and relevant named
relationships. It SHOULD use deterministic exact/alias matching first, and MAY use lexical/semantic retrieval to
resolve ambiguous phrases. Matching SHALL NOT assume that every token in a query corresponds to an ontology IRI.

**3.6** For each retrieval query, the tool SHALL retrieve relevant entities, terms and connecting semantic
relationships. It SHALL merge and deduplicate overlapping results across the `retrieval_queries` of the same intent
while preserving which query matched each result.

**3.7** The tool SHALL support multi-hop traversal through intermediate/bridge entities (e.g.,
Customer -> Order -> OrderLine -> Product). It SHALL return intermediate entity and relationship identifiers, not
merely a boolean stating that two concepts are connected.

**3.8** Multi-hop traversal SHALL be bounded by `max_hops` and configured resource limits, SHALL avoid cycles within
an individual candidate path, and SHOULD support directed traversal by default. Reverse traversal MAY be supported
only with clear reverse-relationship semantics.

**3.9** The tool SHALL rank candidate paths using semantic relevance to the intent, relationship meaning, verified
join-mapping coverage and path length. A shorter path SHALL NOT outrank a semantically correct longer path solely
because it has fewer hops. Relevance scores SHALL NOT be labeled as calibrated probabilities unless calibration has
been performed.

**3.10** The tool SHALL expand selected semantic paths with relevant Business Terms, definitions, aliases,
associated tables/columns, composite FK column pairs, join predicates and applicable business constraints. This
expansion SHOULD be configurable to control response size.

**3.11** The tool SHALL distinguish between:

- semantic/model-level connectivity, which describes how data could be joined; and
- record-level facts, which require actual data retrieval or explicit instance triples.

The tool SHALL NOT assert that a named customer purchased a named product based only on schema-level paths.

**3.12** The tool SHALL report explicit uncertainty when an entity, term, relationship, FK mapping or complete path
cannot be resolved or verified. It SHALL NOT invent table names, columns, joins or relationships.

**3.13** Recommended response schema (illustrative; it does not claim that these resources exist):

```json
{
  "intent_id": "I1",
  "status": "ok",
  "matched_queries": ["customer born before 1990"],
  "business_entities": [
    {"iri": "https://example.com/kg/Customer", "name": "Customer", "definition": "..."}
  ],
  "business_terms": [
    {"iri": "https://example.com/kg/term_birth_date", "name": "Birth Date",
     "entity_iri": "https://example.com/kg/Customer", "definition": "..."}
  ],
  "relationships": [],
  "paths": [],
  "physical_mappings": {"tables": [], "columns": [], "joins": []},
  "constraints": [],
  "evidence": [],
  "warnings": [],
  "ambiguities": [],
  "assertion_level": "schema",
  "graph": {"version": "..."}
}
```

When paths are found, each path SHALL identify ordered node IRIs, semantic predicate IRIs/names, source-to-target
direction, mapping references, validation status and an explainable relevance score. A result shortened to respect
`knowledge_graph.max_response_chars` SHALL say so (`truncated`).

**3.14** An empty result SHALL use an explicit status (e.g., `no_match`) and empty result arrays, rather than
fabricated fallback metadata. Errors SHALL use a separate explicit status (e.g., `error`) with a safe diagnostic
message.

**3.15** The tool SHOULD expose relevant result provenance (graph name/source, discovered vs. verified) and stable
identifiers that the executor can reference in later actions.

**3.16** Security: the graph tool SHALL enforce the application's existing authorization model, including access
restrictions on source metadata and any document evidence. The MCP tool SHALL NOT bypass row/column or metadata
access controls. If raw SPARQL is supported as a separate administrative capability, it SHALL be read-only,
access-controlled and protected by query limits; user-originated strings SHALL NOT be interpolated unsafely into
SPARQL.

### retrieve_documents and evidence markers

**3.17** The existing `retrieve_documents` tool SHALL remain the mechanism for document retrieval. It takes a list
of `queries` (searched in parallel and merged; each result lists the queries that matched it), plus the existing
filters, `expand_parents` and `view`. Its result fields are unchanged. The graph tool MAY return document
references, but SHALL NOT replace `retrieve_documents` or introduce a second, incompatible document retrieval
implementation.

**3.18** Items a caller can cite SHALL carry evidence markers:

- `evidence_ref`: a stable key (the same chunk, glossary entry, graph fact, table or concept always has the same
  ref);
- `evidence_kind`: `document_passage`, `document_text`, `glossary`, `graph`, `dataset`, `business_term`,
  `business_entity`, `business_definition`, `query_result` or `html_fragment`;
- `evidence_detail: true` for a richer version of an item another tool may already have returned.

`retrieve_documents` and `query_knowledge_graph` with `view: "agent"` SHALL return the compact shape shown to the
search agent's executor, with each item's full citation data in `evidence_data`. Other MCP clients MAY ignore the
markers.

### Catalog lookup, SQL and HTML tools

**3.19** The catalog and business-concept lookup tools SHALL return:

- `find_datasets` (`name_contains`, `column_contains`, `source_id`, `limit` 1–50): cataloged tables whose name,
  schema or business entity, or one of whose columns or business terms, contains the text, with their dataset IDs.
- `get_dataset` (`dataset_id`): one table's columns with data types, business terms and a profile summary, its
  primary/unique keys and foreign keys (declared, or inferred with their confidence), and the tables that reference
  it. It SHALL NOT return raw profiled values.
- `get_business_term` (`slug`): the term's description, the columns it is bound to, its definitions and related
  graph facts.
- `get_business_entity` (`slug`): the entity's description, the tables that represent it and related graph facts.
- `get_business_definition` (`definition_id`): the definition's text, author and the business terms it describes.

**3.20** The `generate_sql` tool SHALL accept `source_id`, `intent` (what the rows must answer, with every filter,
comparison and date range) and ALL the evidence the query needs:

- `tables`: `[{name, columns: [{name, data_type, business_term}], keys: [{type, columns, references}]}]`, names
  schema-qualified when the table has a schema;
- `joins`: `[{from_table, to_table, column_pairs: [{from_column, to_column}]}]`, column pairs in order;
- `facts`: business facts as text.

It SHALL look up nothing else except the source's SQL dialect from its configuration. An LLM (the agent model with
`agent.sql_generation_prompt_path`) SHALL write exactly one read-only query from that evidence. The query SHALL be
validated (read-only, cataloged tables of the source) and SHALL NOT be run. The result SHALL have a status: `ok`
(with `sql`, `dialect`, `tables`), `invalid` (with the validation problem), `cannot_answer` (with the reason) or
`error`.

**3.21** The `execute_sql` tool SHALL accept a cataloged database identifier and a read-only SQL query, plus
optional scope filters (`structured_source_ids`, `dataset_ids`) and a `caller` recorded in the audit trail. It SHALL
enforce the requesting user's database and data access permissions, execute the query using the configured
connection, and return structured results containing a status, column metadata and rows on success. A successful
query with zero rows SHALL return an empty row list. Execution failures SHALL return a distinct error status and a
safe diagnostic message. The MCP tool SHALL use the existing SQL execution mechanism.

**3.22** The `generate_html` tool SHALL accept authorized content, including query results when applicable, and
return an HTML fragment for display on the web page. The web application SHALL render the fragment when HTML output
is requested. Input values SHALL be treated as content and escaped as needed. The application SHALL sanitize
returned markup before rendering it.

**3.23** `query_knowledge_graph`, `execute_sql` and `generate_html` SHALL publish documented input and output
schemas. Every tool description SHALL state the tool's input, purpose and output (Section 5.5).

## 4. LLM planner requirements

**4.1** The harness SHALL receive the original user query and call its existing planner LLM integration to produce
a structured plan before execution, following the current architecture.

**4.2** The planner SHALL determine whether the user query needs decomposition into multiple intents. A query with
one clear objective SHALL normally produce one intent and `decompose=false`. Multi-objective queries MAY produce
multiple intents and `decompose=true`.

**4.3** The planner SHALL output a list of intents. Each intent SHALL have a nonempty list of `retrieval_queries`,
representing alternate, search-oriented formulations of that intent. Retrieval queries SHOULD be suitable for
searching either the knowledge graph or documents and SHALL preserve important constraints from the original
intent.

**4.4** The planner output SHALL preserve the following JSON structure and field names:

```json
{
  "original_query": "list customer born before 1990",
  "decompose": false,
  "intents": [
    {
      "id": "I1",
      "intent": "list customer born before 1990",
      "retrieval_queries": [
        "customer born before 1990",
        "customers birth date before 1990"
      ],
      "depends_on": []
    }
  ],
  "reasoning": "The query has a single clear objective to find customers born before 1990 and does not require decomposition.",
  "execution": "single"
}
```

**4.5** Field requirements:

- `original_query`: the exact incoming user question or faithfully preserved original text.
- `decompose`: boolean indicating whether multiple intents are required.
- `intents`: nonempty ordered list of intent objects.
- `intents[].id`: unique identifier within the plan (e.g., `I1`, `I2`).
- `intents[].intent`: natural-language objective for one intent.
- `intents[].retrieval_queries`: nonempty array of alternative retrieval strings.
- `intents[].depends_on`: array of intent IDs that must complete first, or `[]`.
- `reasoning`: concise, high-level decomposition rationale; SHALL NOT require hidden chain-of-thought.
- `execution`: one of the values the harness accepts: `single` (one independent intent), `parallel`, `sequential`
  or `mixed`. No other values SHALL be invented.

**4.6** Dependencies SHALL reference valid intent IDs, SHALL NOT include self-dependencies and SHALL NOT form
cycles. Independent intents MAY be executed according to existing harness behavior. New scheduling semantics SHALL
NOT be introduced as part of this work.

**4.7** The planner SHALL preserve query filters, comparators, temporal constraints, negations, requested entities
and output requirements. For example, "born before 1990" SHALL NOT be reduced to an unconstrained
"customer birth date" retrieval query only.

**4.8** The planner SHALL NOT embed SQL, guess joins or assume physical tables/columns without retrieved evidence.
Its function is to produce intents and retrieval formulations, not to perform knowledge-graph traversal or physical
query planning.

**4.9** The planner output SHALL be validated against the existing schema (or equivalent validation added at the
planner boundary only). Invalid outputs SHALL follow the existing error/retry/fallback policy; this project SHALL NOT
redesign the harness loop.

## 5. Harness / executor requirements

**5.1** The harness SHALL retain its existing agent loop, tool-calling cycle, termination checks, retry logic and
replanning behavior. It SHALL handle human clarification at the response boundary. This specification SHALL NOT
introduce a replacement nested loop over intents, a separate executor loop or an alternative workflow engine.

**5.2** The implementation SHALL integrate by:

1. Preserving the existing planner call and output envelope.
2. Making `retrieval_queries` available to the existing executor/tool-selection context through existing supported
   interfaces.
3. Registering the tools of Section 3.1 on the MCP server so that the executor can discover and invoke them.
4. Returning graph context in a tool result consumable by the existing executor.

**5.3** The harness SHALL use the MCP server's tools only, through the MCP protocol: it discovers them with
`list_tools` and calls them with `call_tool` (Section 3.2). The agent SHALL NOT have its own tool implementations.

- The executor SHALL be offered the read-only tools listed in `mcp.tools` (the knowledge-graph and SQL tools only
  when enabled), never the write tools (crawl, ingest, index, catalog, key discovery, jobs).
- The harness SHALL fill in, and hide from the model, the parameters it owns: the search request's scope, the
  caller, `traceparent` and `view`. The model MAY narrow `evidence_types`, `top_k`, `document_ids` and
  `dataset_ids` of `retrieve_documents` within the request's scope, never widen them.
- The harness SHALL number the items that carry evidence markers (Section 3.18) as evidence IDs (D, G, K, T, B, R,
  H) for citations, synthesis, quality checks and clarification resume, and SHALL bound each result shown to the
  model (`agent.max_observation_chars`).
- A `generate_html` table block MAY name an `execute_sql` result of the search (`result_ref`, e.g. `R1`); the
  harness SHALL fill in its columns and rows before the MCP call and keep the returned fragment out of the model's
  context (it is shown on the page).

**5.4** The executor SHALL determine whether the intent requires semantic or schema metadata, structured records,
unstructured documents, or a combination. It SHALL select MCP tools based on the information and output required by
the intent:

- `query_knowledge_graph`: business entities, business terms, entity paths, relationships, physical table/column
  mappings and joins.
- `retrieve_documents`, `read_document`: unstructured documents and document chunks.
- `find_datasets`, `get_dataset`, `get_business_term`, `get_business_entity`, `get_business_definition`: catalog
  tables and business concepts.
- `generate_sql`, then `execute_sql`: records from the selected database.
- `generate_html`: HTML content for web display.

**5.5** The MCP tool descriptions SHALL state the input, purpose and output of each tool. The executor MAY call
multiple tools when the intent requires more than one source of evidence or output format.

**5.6** No hardcoded, unconditional per-intent retrieval loop SHALL be added outside the existing loop. The executor
SHALL remain responsible for deciding which tool(s) to call and with what arguments under the current tool-calling
protocol.

**5.7** The executor SHALL receive the relevant intent text and `retrieval_queries` in its existing context and
SHOULD pass them to `query_knowledge_graph` when calling it. It SHALL be allowed to use multiple
`retrieval_queries` for recall, with graph-side deduplication.

**5.8** When an intent requires database records, the executor SHALL call `generate_sql`, which invokes an LLM to
write a read-only SQL query for the target database dialect (Section 3.20). The executor SHALL pass the resolved
intent and ALL the catalog and knowledge-graph evidence the query needs (tables, columns, keys, joins and filters),
taken from earlier tool results.

**5.9** The existing replanning process SHALL remain unchanged. The graph tool SHALL provide structured
no-match/ambiguity feedback that the current harness can use according to its existing decision rules; it SHALL NOT
force replanning or directly invoke the planner.

**5.10** The executor SHALL submit the generated query and the target database identifier to `execute_sql`. It
SHALL use the returned rows as evidence for record-level answers, and SHALL invoke `generate_html` with the relevant
authorized content when the response requires HTML for web display.

**5.11** When the executor reports that it is unsure how to proceed or cannot answer with confidence, or when the
final result would be "No information found", the harness SHALL ask the user for the specific additional information
needed before finalizing the response. The harness SHALL retain the original request and relevant context while
waiting for the reply, and SHALL continue processing the request with the clarification once it arrives.

**5.12** Without a usable agent model, or when the MCP server is unreachable, search SHALL fall back to direct
retrieval and say why (`mode: "retrieval"`, `agent.reason`).

## 6. Knowledge graph retrieval behavior

**6.1** Entity and term retrieval SHALL support direct lookup by stable IRI, exact label, alternate label,
case-insensitive normalized label and relevant contextual matching. Additional semantic/embedding retrieval MAY be
used for approximate matching.

**6.2** For a query such as "list customer born before 1990", the graph tool SHOULD identify Customer and a Birth
Date term/property, resolve any physical table/column mappings, and provide the condition meaning and type metadata
if available. The graph tool SHALL NOT return actual customer rows from semantic metadata alone.

**6.3** For a query such as "customer bought a product", graph retrieval SHALL discover a semantically valid
multi-hop path such as Customer -> Order -> OrderLine -> Product when the graph supports it. The result SHOULD
identify each relationship and any verified join mapping along that path.

**6.4** If multiple routes exist (e.g., via Review, SupportTicket, Return or Order), the retriever SHALL rank for the
meaning of "bought" or "purchased" rather than blindly selecting the shortest path. Paths requiring unverified
relationship semantics SHOULD carry warnings or lower confidence.

**6.5** The retriever SHALL support a configurable maximum hop count, top-k results, maximum candidate expansions,
timeout and total output size. Defaults SHOULD be documented following the existing configuration conventions; no
particular new config file is mandated.

**6.6** Standard SPARQL 1.1 property-path queries MAY establish reachability, but the retriever SHALL return actual
intermediate nodes and predicates. Bounded traversal MAY be implemented with BFS/DFS over a semantic adjacency index
or explicit bounded SPARQL query patterns. It SHALL NOT rely on unsupported SPARQL 1.1 bounded path syntax such as
`predicate{1,5}`.

**6.7** Duplicate entities, terms, paths and mappings SHALL be deduplicated by stable identifiers. Different physical
implementations of the same semantic relationship SHOULD remain distinguishable.

**6.8** Graph retrieval SHOULD prefer validated relationships and declared keys. Inferred keys and mappings SHALL
retain their confidence and provenance; their validation status SHALL be included in results.

**6.9** A complete semantic path SHALL NOT be claimed executable for SQL generation if a required physical mapping
or join pair is missing. The tool SHALL return partial context and clearly identify the missing mapping instead.

## 7. Catalog UI requirements

**7.1** The catalog SHALL display a navigation facet on the left with a tree of cataloged databases. Each database
SHALL contain its schemas, and each schema SHALL contain its physical tables (Databases -> Schemas -> Tables).

**7.2** Right-clicking a database, schema or table in the catalog tree SHALL open the knowledge graph view for that
selection. The view SHALL show the selected table, or the tables under the selected schema or database, together
with their cataloged columns, keys and linked business metadata, subject to the user's catalog permissions.

## 8. Configuration, operations and observability

**8.1** The implementation SHALL use the existing project configuration conventions (`config/mesh.yaml`). Configured
values and their defaults:

| Setting | Default | Meaning |
|---|---|---|
| `knowledge_graph.enabled` | `true` | `query_knowledge_graph` available |
| `knowledge_graph.storage_backend` | `trig_file` | graph storage |
| `knowledge_graph.data_path` | `./data/knowledge_graph.trig` | dataset location |
| `knowledge_graph.shapes_path`, `ontology_path` | `./config/knowledge_graph_shapes.ttl`, `./config/knowledge_graph_ontology.ttl` | SHACL shapes, vocabulary |
| `knowledge_graph.default_max_hops` / `max_hops_limit` | 5 / 8 | hops per request / largest a caller may ask for |
| `knowledge_graph.default_top_k` / `max_top_k` | 5 / 20 | ranked paths returned |
| `knowledge_graph.max_candidate_paths` | 200 | candidate paths kept for ranking |
| `knowledge_graph.max_expansions` | 5000 | traversal steps per request |
| `knowledge_graph.query_timeout_seconds` | 5 | per request |
| `knowledge_graph.max_response_chars` | 20000 | response size |
| `knowledge_graph.include_inferred_relationships` | `true` | `false`: verified relationships only |
| `knowledge_graph.embedding_retrieval_enabled` | `true` | embedding matching for loose phrasings |
| `knowledge_graph.ranking_weights` | semantic 0.6, coverage 0.25, status 0.15 | path score |
| `mcp.mode` / `mcp.url` | `in_process` / `http://127.0.0.1:8000/mcp` | harness connection (Section 3.2) |
| `mcp.tools` | the read-only tools of Section 5.4 | tools offered to the executor |
| `sql_execution.*` | `default_max_rows` 200, `max_rows_limit` 1000, `timeout_seconds` 30, `max_sql_chars` 10000 | `execute_sql` |
| `html_output.*` | `max_chars` 200000, `max_table_rows` 500 | `generate_html` |
| `agent.sql_generation_prompt_path` | `./prompts/sql_generation.txt` | `generate_sql` prompt |

**8.2** The existing `retrieve_documents` configuration (`retrieval.*`) SHALL remain intact.

**8.3** MCP tool invocations SHOULD produce traceable logs/telemetry for `intent_id`, tool name, lookup types, query
count, graph version, matched entity/term counts, paths considered/returned, hop count, duration, status and
warnings. Logging SHALL avoid exposing restricted values or unnecessary sensitive user text.

**8.4** The system SHOULD support loading/reloading validated RDF datasets and SHOULD report ontology/metadata
indexing failures without corrupting the prior usable graph.

**8.5** Results SHOULD include stable IRIs, relationship IDs, source identifiers and the graph version/timestamp to
aid reproducibility and debugging.

**8.6** The system SHALL provision the RDF knowledge graph with references to the cataloged Business Terms,
Business Entities, physical tables, physical columns and keys, including primary, foreign, unique and composite
keys. The graph SHALL preserve the catalog's table-to-entity and column-to-term links.

## 9. Acceptance tests

| ID | Test |
|---|---|
| AT-01 | **MCP publication.** `query_knowledge_graph`, `execute_sql` and `generate_html` are discoverable with valid, documented schemas. `retrieve_documents` is discoverable, takes a list of `queries` and returns the same result fields as before. |
| AT-02 | **Planner single intent.** For "list customer born before 1990", the planner emits the exact required field structure, `decompose=false`, one intent, nonempty `retrieval_queries`, `depends_on=[]` and `execution="single"`. |
| AT-03 | **Planner multiple intents.** A genuinely multi-objective question yields multiple unique intent IDs with valid dependencies and retrieval queries, using only the execution modes the harness supports. |
| AT-04 | **Harness compatibility.** The existing tool-calling cycle, termination checks, retries and replanning operate as before, and the harness handles clarification at the response boundary. |
| AT-05 | **Business Entity lookup.** A query for "customer" retrieves the Customer entity with its stable IRI and relevant definition/mappings where available. |
| AT-06 | **Business Term lookup.** A query for "customer date of birth" resolves an appropriate Birth Date term, its owning entity and its mapped columns, if present. Synonyms work. |
| AT-07 | **Multi-hop.** With Customer -> Order -> OrderLine -> Product in the test graph, the query "customer bought product" returns the ordered three-hop semantic path with predicate names and intermediate nodes. |
| AT-08 | **Competing paths.** With both purchase and support-ticket paths available, the "bought" intent ranks the purchase path above an unrelated support-ticket path. |
| AT-09 | **Composite FK.** A composite two-column FK is returned as two correctly ordered source-to-target column pairs and is not flattened into an incorrect single-column join. |
| AT-10 | **Missing metadata.** If a join mapping is absent, graph retrieval returns partial evidence plus a warning, without inventing a join. |
| AT-11 | **Record vs. schema.** The semantic schema alone does not produce a claim that a real customer bought a real product; record-level claims require actual data evidence. |
| AT-12 | **No match.** An unknown entity/term returns `status=no_match` (or a documented equivalent), without crashing or hallucinating graph metadata. |
| AT-13 | **Boundary limits.** `max_hops`, `top_k`, timeouts, traversal limits and access controls are enforced. |
| AT-14 | **Document retrieval regression.** `retrieve_documents` keeps its result fields and behavior in the regression tests; its `queries` parameter is a list (approved change). |
| AT-15 | **Combined retrieval.** The existing executor can call `query_knowledge_graph` and `retrieve_documents` within its current loop when both structured and unstructured context are needed. |
| AT-16 | **Business term rule.** The graph can represent "Settlement Date expected after Transaction Date" without asserting that all underlying records satisfy the rule; executable validation is performed only through an explicitly configured rule engine. |
| AT-17 | **Catalog configuration and mappings.** The databases specified in settings are scanned and their metadata is cataloged. Each physical table links to a Business Entity and each physical column links to a glossary Business Term, including cases where multiple tables share an entity and multiple columns share a term. |
| AT-18 | **Graph provisioning.** The knowledge graph provisioned from a populated catalog references the cataloged Business Terms, Business Entities, physical tables, physical columns and keys, and preserves the table-to-entity and column-to-term links. |
| AT-19 | **Catalog navigation.** The catalog displays a left-hand facet with a tree that nests schemas under their databases and tables under their schemas. |
| AT-20 | **Catalog knowledge graph.** Right-clicking a database, a schema and a table in the catalog tree displays the knowledge graph scoped to the selected item and its cataloged contents. |
| AT-21 | **SQL retrieval.** For a request for database records, the executor calls `generate_sql` (an LLM writes the SQL from the evidence passed), calls `execute_sql` for the target database and uses the returned rows in the answer. The results of an empty query and of an execution error are verified. |
| AT-22 | **HTML output.** For an HTML response for web display, `generate_html` returns an HTML fragment from the supplied content, and the web page displays it with data values rendered as content. |
| AT-23 | **Human clarification.** Executor uncertainty and a final result of "No information found" each cause the harness to ask the user for additional information. After a reply, execution continues with the original request and the clarification. |
| AT-24 | **MCP tools only.** The executor's tool list comes from the MCP server's `list_tools`: only the `mcp.tools` allowlist, no write tools, and none of the parameters the harness fills in. The search scope is added to every call and can only be narrowed. |
| AT-25 | **Connection modes.** Search works with `mcp.mode: in_process` and with `streamable_http`; in `streamable_http` mode the harness and an outside MCP client both use `/mcp`, which requires the API key when one is configured. |
| AT-26 | **SQL from the evidence passed.** `generate_sql` writes its prompt only from the tables, joins and facts it receives (no catalog lookup), and rejects a request without tables. |
| AT-27 | **MCP server unreachable.** When the MCP server cannot be reached, search falls back to direct retrieval and says why. |

## 10. Implementation deliverables

- **10.1** RDF vocabulary/ontology definitions for business concepts, physical metadata, mappings, constraints and
  provenance.
- **10.2** RDF dataset access/repository layer using RDFLib 7.6.0.
- **10.3** Entity/term resolution and bounded multi-hop path retrieval.
- **10.4** Semantic path ranking and physical mapping/context expansion.
- **10.5** New MCP tool `query_knowledge_graph`, including input/output schema, help text, authorization, error
  handling and tests.
- **10.6** Preservation and regression testing of the existing MCP document tool `retrieve_documents`.
- **10.7** Planner output contract validation using the existing field names and data shape.
- **10.8** Integration through the existing harness and executor, including clarification and continuation at the
  response boundary.
- **10.9** Unit/integration/regression tests corresponding to AT-01 through AT-27.
- **10.10** README/technical documentation with example query requests, responses, ontology examples,
  configuration and known limitations.
- **10.11** RDF knowledge graph provisioning from the existing catalog.
- **10.12** Catalog UI facet for database, schema and table navigation with a right-click knowledge graph view.
- **10.13** MCP tools `execute_sql` and `generate_html`, including documented schemas and integration with the
  executor and web application.
- **10.14** MCP tools `find_datasets`, `get_dataset`, `get_business_term`, `get_business_entity`,
  `get_business_definition` and `generate_sql`.
- **10.15** The harness MCP client with its two connection modes (Section 3.2), the read-only allowlist and the
  hidden parameters (Section 5.3), and `/mcp` mounted in the FastAPI app.
- **10.16** Evidence markers and the compact `view: "agent"` (Section 3.18).

## 11. Non-goals and design constraints

- **11.1** DO NOT rewrite, replace or nest the existing harness execution loop.
- **11.2** DO NOT reimplement or rename `retrieve_documents` (its list of `queries` is an approved interface
  change, not a reimplementation).
- **11.3** DO NOT make graph traversal depend on the LLM generating each next graph hop.
- **11.4** DO NOT collapse the business ontology into physical tables/columns.
- **11.5** DO NOT equate a database FK with a verified business relationship without semantic evidence.
- **11.6** DO NOT treat schema-level connections as proof of particular record-level transactions.
- **11.7** DO NOT invent a new planner output format or new execution enum values not supported by the current
  harness.
- **11.8** DO NOT make retrieval depend on an entirely new SQL engine, document loader, vector DB or workflow
  framework.
- **11.9** DO NOT give the agent its own tool implementations; its tools are the MCP server's tools.

## 12. Changes from version 1.0

New numbers first; version 1.0 numbers in brackets.

**Document**

- Converted to Markdown; status changed from "not implemented" to implemented; version 1.1.
- Sections reordered: Catalog UI is now Section 7 [11]; Configuration 8 [7]; Acceptance tests 9 [8];
  Deliverables 10 [9]; Non-goals 11 [10]. Acceptance test IDs AT-01 … AT-23 are unchanged.
- Keyword case: "SHALL not" -> "SHALL NOT" in 2.8, 3.5 [3.4], 3.9 [3.8], 4.6, 4.7 and 6.9.
- Wording and grammar corrected throughout (e.g., "the application's existing authorization model" in 3.16 [3.15]);
  example queries kept verbatim ("list customer born before 1990"), as the tests use them.

**Requirements**

- 1.4: lists the lookup tools and SQL generation among the published tools.
- 1.5: the harness uses the MCP server's tools only, through the MCP protocol; clarification reference updated to
  5.11 [5.10].
- 1.7: worded as a requirement (keep the existing catalog behavior).
- 3.1: adds `find_datasets`, `get_dataset`, `get_business_term`, `get_business_entity`, `get_business_definition`
  and `generate_sql`; `retrieve_documents` takes a list of `queries`; the existing ingestion/catalog tools stay
  published.
- 3.2 (new): MCP connection modes `in_process` and `streamable_http` (`/mcp`, `mcp.url`, same API key as `/api`).
- 3.3–3.16 [3.2–3.15]: renumbered. 3.4 [3.3] adds the scope filters, `view`, the 1–5 queries and the configured
  limits (`max_hops_limit`, `max_top_k`). 3.13 [3.12] adds `relationships`, `ambiguities`, `assertion_level`,
  `graph` and `truncated`.
- 3.17 [3.16]: `retrieve_documents` takes a list of `queries`, plus `expand_parents` and `view`; result fields
  unchanged.
- 3.18 (new): evidence markers (`evidence_ref`, `evidence_kind`, `evidence_detail`) and `view: "agent"` with
  `evidence_data`.
- 3.19 (new): the catalog and business-concept lookup tools; `get_dataset` never returns raw profiled values.
- 3.20 (new): `generate_sql` receives ALL its evidence (tables, joins, facts) and looks up only the SQL dialect.
- 3.21 [3.17]: `execute_sql` accepts optional scope filters and a `caller` for the audit trail.
- 3.22 [3.18]: unchanged. 3.23 [3.19]: also requires `query_knowledge_graph`'s schemas and input/purpose/output
  descriptions for every tool.
- 4.5: `execution` values named (`single`, `parallel`, `sequential`, `mixed`).
- 5.2: item 3 registers all the tools of 3.1.
- 5.3 (new): MCP tools only, read-only allowlist, hidden parameters, scope narrowing, evidence IDs, bounded results,
  `result_ref` for `generate_html`.
- 5.4 [5.3]: tool list adds `read_document`, the lookups and `generate_sql`. 5.5–5.7 [5.4–5.6]: renumbered.
- 5.8 [5.7]: the executor calls the MCP tool `generate_sql` and passes all the evidence (was "the Executor LLM
  SHALL invoke an LLM").
- 5.9–5.11 [5.8–5.10]: renumbered.
- 5.12 (new): fallback to direct retrieval without an agent model or MCP server.
- 8.1 [7.1]: the actual setting names and defaults (`data_path` instead of `dataset_location`, `max_hops_limit`,
  `max_top_k`, `max_expansions`, `ranking_weights`) plus `mcp.*`, `sql_execution.*`, `html_output.*` and
  `agent.sql_generation_prompt_path`.
- AT-01, AT-14: `retrieve_documents` takes a list of `queries` (AT-14 no longer requires an unchanged signature).
- AT-21: the executor calls `generate_sql`.
- AT-24 – AT-27 (new): MCP tools only, connection modes, SQL from the evidence passed, MCP server unreachable.
- 10.9 [9.9]: tests AT-01 through AT-27. 10.14–10.16 (new): lookup tools and `generate_sql`, the harness MCP client,
  evidence markers.
- 11.2 [10.2]: notes the approved `queries` interface change. 11.9 (new): no agent-only tool implementations.

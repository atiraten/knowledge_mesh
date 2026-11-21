# Knowledge Mesh

Python POC that ingests documents and websites, catalogs relational/Parquet/Databricks sources,
discovers keys, generates a business glossary, captures business definitions, and publishes
everything to an RDF knowledge graph. One query returns document passages, glossary matches and
graph evidence (search mode, no LLM needed), with optional grounded answers that cite them.

Specification: [doc/knowledge_mesh_requirements.md](doc/knowledge_mesh_requirements.md).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[google,dev]"        # add ,databricks / ,openai / ,eval as needed
.venv/bin/python src/harness.py check-config    # validate config/mesh.yaml
.venv/bin/python src/harness.py serve           # UI + API on http://127.0.0.1:8000
.venv/bin/python src/harness.py mcp             # MCP server (stdio)
.venv/bin/python -m pytest                      # test suite (offline, no model download)

scripts/mesh.sh start                           # observability stack (Podman) + app on http://localhost:9801
scripts/mesh.sh status | stop | restart         # add --no-observability to leave the stack alone
scripts/observability.sh up | down | status | logs [service]   # only the observability stack
```

The first indexing run downloads `sentence-transformers/all-mpnet-base-v2` (~420 MB) into the
Hugging Face cache; after that embeddings run fully locally (set `embedding.local_files_only: true`
to forbid any hub access). `src/harness.py` works without installing the package; once installed,
`knowledge-mesh <command>` is equivalent.

> macOS note: if `import knowledge_mesh` fails after an editable install, macOS may have set the
> `hidden` flag on `.venv/.../site-packages/*.pth` (Python 3.12+ skips hidden `.pth` files). Run
> `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`, use `src/harness.py`, or set `PYTHONPATH=src`.

CLI commands: `serve`, `mcp [--transport streamable-http]`, `check-config`, `ingest PATH`,
`crawl URL [--max-depth N --max-pages N]`, `catalog SOURCE_ID [--dataset NAME]`,
`search QUERY [--retrieve-only]` (agent search; `--retrieve-only` for direct retrieval),
`reindex [--rebuild]`, `eval TESTSET.jsonl [--k K --ragas]`.

## Layout

```text
src/harness.py                     run from a checkout (adds src/ to sys.path)
src/knowledge_mesh/
  config.py                        load + validate config/mesh.yaml (pydantic, strict keys)
  mesh.py                          orchestrator: every tool operation, callable from Python
  mcp_server.py                    MCP tools (crawl, ingest, index, read, extract, discover keys, retrieve,
                                   query_knowledge_graph, execute_sql, generate_html)
  tool_schemas.py                  published output schemas of the knowledge graph / SQL / HTML MCP tools
  api/app.py, api/static/          FastAPI routes, middleware, browser UI (plain HTML/CSS/JS)
  unstructured/                    loaders (PDF/DOCX/TXT/MD/HTML), document registry, hierarchical
                                   chunking, crawler + URL/SSRF policy, entity/relationship extraction,
                                   indexing pipeline
  structured/                      connectors (SQLAlchemy, Parquet, Databricks), catalog, profiling,
                                   keys/ucc.py (HyUCC + Apriori), keys/binder.py (BINDER),
                                   key_discovery.py, fk_rules.py (FK identifier rules + confidence),
                                   terms.py (LLM business terms), glossary.py, relationships.py (LLM
                                   named relationships, term relations, synonyms), reviews.py
                                   (verify / reject), sql_exec.py (execute_sql)
  indexing/                        local embeddings, persistent SimpleVectorStore wrapper
  graph/                           RDF vocabulary, layer builders, SHACL-gated graph store,
                                   retriever.py (query_knowledge_graph), view.py (catalog graph view)
  common/html_render.py            generate_html: escaped HTML fragments from content blocks
  retrieval/                       mixed retrieval, Laya reranking, evidence blocks, one-shot answers
  agent/                           search agent harness: planner, per-intent tool loop, tools, evidence
                                   ledger, synthesis (prompts/planner.txt, executor.txt, synthesis.txt)
  definitions.py                   business definitions
  jobs.py                          durable background jobs + paginated manifests
  llm/                             Gemini / OpenAI-compatible clients with tracing + cost metrics
  observability/                   OpenTelemetry traces, metrics, JSON logs
  evaluation/                      Recall@K / MRR / nDCG and optional Ragas
config/mesh.yaml                   configuration
config/knowledge_graph_shapes.ttl  SHACL shapes
config/knowledge_graph_ontology.ttl  vocabulary (OWL / SKOS classes and properties)
prompts/relationship_generation.txt, prompts/sql_generation.txt   relationship and SQL generation prompts
src/knowledge_mesh/api/static/vendor/cytoscape.min.js   cytoscape.js 3.34.4 (MIT, see cytoscape.LICENSE)
deploy/                            Docker Compose: OTel Collector, Phoenix, Prometheus, Grafana, Loki
tests/                             automated tests (one module per component + acceptance flows)
```

Runtime data lives under `data/` (git-ignored): `originals/`, `extracted/`, `metadata.sqlite`,
`index/documents`, `index/glossary`, `knowledge_graph.trig`, `evaluations/`.

## How it works

**Documents.** `document_id` is derived from the source URI, so a changed file or page keeps
its ID and replaces its chunks; identical bytes are a no-op. Identical content under another URI
is recorded as `skipped / duplicate_content`, so it never gets embedded twice. Text is extracted
with page numbers (PDF), headings (DOCX/Markdown/HTML) and table boundaries. LlamaIndex
`HierarchicalNodeParser` splits each section at 2048 → 1024 → 256 tokens. IDs are deterministic
(`node-*` for parents, `chunk-*` for leaves). Only leaves are embedded. Parents stay in the
docstore for context expansion and citations (`/api/chunks/{id}` shows a leaf and its parents).

**Crawler.** BFS from the start URL, exact hostname by default (`crawler.allowed_hosts` adds
hosts; `*.example.com` adds subdomains). It enforces depth, page, concurrency, per-host rate,
timeout and size limits. Depth counts page levels from 1: **max depth 1 fetches only the given
URL** (a user-requested single page: no robots.txt, sitemap or link discovery), 2 adds the pages
it links to, and so on. From depth 2 it obeys robots.txt (RFC 9309) and crawl-delay, uses
sitemaps, and honors `noindex`/`nofollow`. It skips non-content links, binary assets, repeated path segments and
query-variant explosions. URL canonicalization: lower-case scheme and host, drop default ports
and fragments, resolve dot segments, strip tracking/session parameters (`utm_*`, `gclid`, …),
sort the remaining parameters, and keep percent-encoding. Every request and redirect target is
checked against private, loopback, link-local, metadata and reserved addresses. The residual risk
is DNS rebinding between the check and the connection, so run the crawler where egress to
internal networks is firewalled. Every page gets a manifest item (`indexed|skipped|failed` + reason).

**Structured catalog.** Connectors read metadata (tables/views/files, columns, types,
nullability, declared PK/unique/FK) into SQLite. Dataset and column IDs derive from
(source, schema, name), so re-runs update in place. Missing columns or datasets are retired,
together with their keys, glossary vectors and graph assertions. Database sessions are forced
read-only where the dialect allows it; still use read-only accounts.

**Profiling and key discovery.** Each column profile records rows evaluated, distinct, empty/blank
and null counts, format patterns (`1999-12-22 → NNNN-NN-NN`), content type, the generic data type
(`type_category`: string, integer, decimal, date, datetime, uuid, …, shown as "Data type" next to the
source's own "Column data type") and most common values. Profiles saved before `type_category` existed
show it empty until the next key discovery run.
Profiles stay in SQLite and the catalog UI, never in retrieval, prompts or logs. Unique column
combinations come from **HyUCC** (sampling → minimal-hitting-set induction → validation in
Apriori order); **Apriori** level-wise search is available via `profiling.ucc_algorithm` and is
cross-checked in the tests. Non-null UCCs are flagged as primary-key candidates. **BINDER**
finds unary and n-ary inclusion dependencies, which become FK candidates with their ordered
column mapping. Every inferred key records its profiling run and scope.

Two rules (`structured/fk_rules.py`) turn inclusions into plausible foreign keys:

- **FKs are identifiers.** A column takes part only if it belongs to a declared key, is a UUID,
  its business term ends with an identifier class word (Identifier, Code, Number, Key, Indicator,
  Flag), or, until it has a term, its name ends with one (`_id`, `_cd`, `_no`, `_key`, `_ind`,
  `_flg`…). Dates, amounts and free text are never offered to BINDER. Business terms are
  generated before key discovery in a catalog run, so the term decides when it exists.
- **FKs reference the primary key.** Targets are the declared primary/unique keys of a table that
  has a declared primary key, otherwise its primary-key candidates; a column that is merely
  unique in the sample (`insurance.loan_id`) is not a target.

Each inferred FK gets a **confidence**, computed whenever it is read (dataset page, API, agent
`get_dataset`, knowledge graph) from the current business terms, so terms generated later
upgrade it without re-running discovery: **high** = same business term, or the PK's term with a
role qualifier ("Parent Counterparty Identifier" → "Counterparty Identifier"); **medium** = same
or role-prefixed column name (or `customer_id` → `customers.id`); **low** = values only, or a
term says one side is not an identifier. Composite FKs take their weakest column pair.

**Business glossary.** For each dataset the term-generation LLM (`term_generation`, Gemini by
default) receives only table and column names through `prompts/term_generation.txt`. Its CSV
output is validated against the cataloged columns: unknown columns are dropped, missing columns
are retried once and then reported. Terms keep their description and decoding comment. Each line
also carries a **sensitive flag** (`<table>,<column>,<term>,<Y|N>,<description>,<comments>`): Y when the
term can identify a physical person, alone or combined with other information (PII / PHI / GDPR: names,
address, city, zip code, email, phone, dates of birth / marriage / death, place of birth, SSN, tax and
national identifiers, Medicare / Medicaid / medical record numbers, …; a company is not a person). It is
stored in `columns.term_sensitive` (NULL = not assessed: terms generated before the flag, or a line without
it), shown as a red "sensitive" badge on the term, and drives the red marking of SQL results. Changing the
prompt changes its version, so the next catalog run regenerates the terms of every dataset (term model calls). The
table's **business entity** is the decoded table name the model writes at the start of each line;
when it equals the physical name, a heuristic is used instead (split, expand abbreviations,
singularize). Terms and entities go into a separate persistent `SimpleVectorStore`. When term
generation is not configured, cataloging and key discovery still run, and the job records a
`skipped` term stage.

**Knowledge graph.** RDFLib 7.6 dataset (TriG file) with named graphs per source. A structured
source has three layers, `urn:mesh:context:source:<id>:physical | semantic | mapping`; documents
(`document:<id>`) and captured definitions (`definition:<id>`) keep one graph each, and the
vocabulary (`config/knowledge_graph_ontology.ttl`) is `urn:mesh:context:ontology:core`.

| Layer | Content |
|---|---|
| physical | tables (`mesh:Dataset`, `mesh:hasColumn`), columns with data types, primary / unique / foreign keys; a foreign key keeps its ordered column pairs (`mesh:columnMapping` + `mesh:position`), so composite keys stay pairwise; origin, profiling run, confidence and status |
| semantic | business entities (`owl:Class` + `mesh:BusinessEntity`), business terms (`skos:Concept` in the glossary scheme with `skos:prefLabel`, `skos:altLabel` synonyms and `skos:definition`, plus a separate `owl:DatatypeProperty` linked by `mesh:termProperty`), named relationships (reified `mesh:SemanticRelationship` + an `owl:ObjectProperty` with domain and range), term relations (`mesh:TermRelation`: expectedAfter, derivedFrom, equivalentTo, dependsOn, isPartOf) |
| mapping | table → entity (`mesh:EntityMapping`) and column → term (`mesh:TermBinding`) links, relationship → foreign key join mappings (`mesh:JoinMapping`, explicit or automatic), and the schema-level navigational edges derived from foreign keys (`mesh:NavigationalEdge`, `references`) |

Refreshing a source replaces its three layers together: they are SHACL-validated as one graph
(pySHACL) and swapped atomically; a rejected update returns the violated shapes and focus nodes and
the previous graph stays live. Graphs in the earlier one-context-per-source layout are migrated at
startup (`KnowledgeMesh.provision_graph`, run by the warm-up). `POST /api/graph/reload` re-reads the
TriG file and keeps the current graph if the file does not parse. Concepts are matched across
sources by label slug (`urn:mesh:entity:customer`). Document entities/relationships come from a
deterministic lexicon matcher (`entity_extraction.method: lexicon`) or the LLM (`llm`, keeps only
entities present in the passage); each links to its supporting passage.

**Named relationships, term relations and synonyms** come from a catalog stage that runs after
key discovery (`structured/relationships.py`, `prompts/relationship_generation.txt`, the
term-generation model). For each table it receives the table's entity, its business terms and its
foreign keys (declared, or inferred with at least `relationship_generation.min_fk_confidence`,
between tables of different entities; inferred keys that repeat a declared one are skipped) and
proposes:

* one named, directional relationship per foreign key (`Customer placesOrder Order`) with a
  definition, synonyms, an inverse name and a cardinality. That key becomes the relationship's
  **explicit** join mapping; other keys between the two entities' tables become **automatic**
  candidates;
* term relations between the table's own terms, limited to the five predicates above. They are
  descriptive (`mesh:assertionLevel "descriptive"`): nothing checks them against records, because no
  rule engine is configured;
* synonyms of the table's terms (`skos:altLabel`).

Only metadata is sent, never row values; proposals are validated (known keys, entities, terms and
predicates) before they are stored. Run it again from the dataset page ("Propose relationships",
`POST /api/datasets/generate-relationships`).

**Status and review.** Declared foreign keys are `verified`. Inferred foreign keys, proposed
relationships and term relations, and the table → entity and column → term links (generated by the
term model; they carry their term-run provenance and no confidence score) start `inferred`. The
dataset page has Verify / Reject / Reset buttons for each (`POST /api/reviews`, any API caller,
audit-logged). Decisions are keyed by stable IDs, so they survive re-runs; a mapping decision applies
to the entity or term it was made for. Rejected items stay in the graph with their status and are
never traversed or returned as evidence. Foreign keys still produce the entity-level `references`
edges, now typed as navigational edges: traversable, but flagged as having no business semantics.

**Reranking with Laya** (`retrieval.reranker_enabled`, settings under `reranker:`). After the
similarity search, the top-k document passages go to the Laya decision model
(`pip install -e ".[laya]"`) in exactly two calls:

1. A `description` *choice* over the passages, with state `{"question": …}`.
2. A batched `quality` *score* for the shortlist, with state `{"question": …, "chosen_response": …}`.

Passages whose choice probability is over `min_probability` (0.8) are kept. The best `top_n` (2)
of them are scored. The best one rated `min_quality` ("Acceptable") or better, by its most likely
level, becomes the response. Otherwise the response is `no_answer_message` ("I could not find a
response."), and answer mode returns that message without calling the LLM.

Choice probabilities sum to 1, so with a threshold above 0.5 at most one passage passes. When
only one passage survives the similarity filter its probability is 1.0 by definition, and the
quality call decides. If Laya fails, results fall back to similarity order with
`reranking.applied: false`. Search requests can override the setting with `"rerank": true|false`.
Decisions (probabilities, quality levels, call count, latency) are returned under `reranking`
and traced as a `RERANKER` span.

**Retrieval.** `retrieve_context` searches both indexes (top-K per index, default 5, cosine
similarity), drops hits below `retrieval.min_*_similarity`, and returns each matched leaf chunk
with its whole parent chain under `parents` (nearest first: the 1024-token parent, then the
2048-token one). Parent text fills `max_context_tokens` one level at a time, and a parent shared
by several results counts once; parents that do not fit stay listed with `expanded: false`.
Laya reranks the leaf chunks only, before expansion. Answer mode sends each passage as
"Matched chunk" plus every expanded parent. It then looks up definitions, relationships and key
constraints for the concepts matched in the query, the glossary hits and the top passages
(bounded by `max_graph_results`). With no evidence it returns `insufficient_evidence: true`.
Passages that look like prompt injection are flagged. Answer mode passes labeled evidence to the
answer LLM as untrusted data, validates the citations it returns, and does not call the LLM at
all when there is no evidence.

**Search (agent harness).** The Search page, `POST /api/search` and `search QUERY` run:

```text
question -> planner LLM (planner prompt) -> plan
  -> orchestration loop: select ready task(s) -> executor loop per task -> evaluate -> next | replan | finish
  -> final response (synthesis prompt)
```

1. **Plan.** The planner (`prompts/planner.txt`, given the question and the plan's JSON schema, never
   the tool list) returns `{original_query, decompose, intents: [{id, intent, retrieval_queries,
   depends_on}], reasoning, execution}`. Retrieval queries are cleaned, de-duplicated and capped at
   `agent.max_retrieval_queries` (an intent without any is searched with its own text). A plan that is
   not valid JSON, has an unknown dependency or a cycle, or has more than `agent.max_intents` intents
   falls back to one intent equal to the question (`plan.fallback_reason`). Rules 9 and 10 of the
   planner prompt keep intents and queries in the user's own terms.
2. **Select ready tasks.** A task is ready when the tasks it depends on have a result. With
   `execution: parallel` or `mixed`, all ready tasks run at the same time (up to
   `agent.max_parallel_intents`); otherwise one at a time, in plan order.
3. **Executor loop per task** (`prompts/executor.txt`). Each call gets the tool list, the planner's
   intents (with retrieval queries, dependencies and status), the current task, the results of the
   tasks it depends on and the task's earlier steps; not the raw question. The model replies with
   JSON: an `evaluation` of the latest results, then either
   `{"action": "tools", "tool_calls": [{"tool", "arguments"}, ...]}` (run in parallel, up to
   `agent.max_tool_calls_per_step`) or `{"action": "complete", "status": "answered|partial|not_found",
   "answer"}`. The harness runs the tools, adds the results to the context and loops, up to
   `agent.max_iterations` (default 20) calls per task. The task's first `retrieve_context` call always
   searches its planned `retrieval_queries` (the call's `note` shows any the model left out). Invalid
   replies and tool errors are fed back and count as iterations; when the context is full, the oldest
   results become one-line stubs that keep their evidence IDs.
4. **Evaluate** (rules, no LLM call). `answered` -> next task. `partial`, `not_found`, the iteration
   limit or a failed model call -> **replan**: the planner LLM gets the question, the plan and every
   task's result and returns a revised plan (answered intents are kept and not rerun), at most
   `agent.max_replans` times per search. No ready task left -> **finish**.
5. **Final response.** One call (`prompts/synthesis.txt`) writes the grounded answer from every task's
   result and the gathered evidence, merging overlaps; citations that point at no supplied evidence
   are removed.

**Knowledge graph, SQL and HTML tools.** The executor also has `query_knowledge_graph` (business entities,
terms, ranked relationship paths and their joins; see below), `generate_sql` (the agent model writes one
read-only SELECT from the cited table, join and term evidence with `prompts/sql_generation.txt`; the query is
checked but not run), `execute_sql` (runs it; rows become `R` evidence for record-level claims) and
`generate_html` (an escaped HTML fragment, `H` evidence, shown on the Search page; with the "HTML output"
checkbox, `output_format: "html"`, the executor is told to produce it). A `query_knowledge_graph` call without
`intent_id`, `intent` or `retrieval_queries` gets them from the current task (noted on the call).

**Clarification** (`agent.clarification_enabled`). When, after the replans, a task is still `partial` /
`not_found` (the executor may say what it needs in `clarification_needed`) or nothing was found, synthesis
writes the answer it can and one question; the response has `outcome: "needs_clarification"` and
`clarification: {question, round, reply_ref}`. The request, plan, task results and evidence are kept in
SQLite (`pending_clarifications`, deleted when answered or after `agent.clarification_ttl_hours`).
`POST /api/searches/{trace_id}/clarification {"reply": ...}` reloads them, calls the existing replan step
with the reply as the reason (answered tasks are kept), shows the reply to the executor and synthesis, and
continues with the same loop; the new response has `continues` (the earlier trace). At most
`agent.max_clarifications` questions per request; each question accepts one reply.

`retrieve_context` takes a list of queries: they are searched in parallel (`retrieval.max_parallel_queries`)
and merged into one result, so a passage, glossary entry or graph fact found by several queries appears
once, with its best score and `matched_queries`. Within a task, an item already shown is returned as
`seen_before` without its text; it keeps its evidence ID. Other tools: `read_document`, `find_datasets`,
`get_dataset` (columns, business terms, profile summary without raw values, keys, foreign keys),
`get_business_term`, `get_business_entity`, `get_business_definition`. Evidence IDs are shared by all
tasks: D passage/text, G glossary match, K graph fact / path / join, T table, B business term/entity/definition,
R query result rows, H HTML fragment.
The response has `plan` and `plans` (one per version), `decisions` (the evaluation after each batch),
`tasks` (each run: status, attempt, plan version, steps with their tool calls, result, evidence IDs),
`answer`, `citations`, `evidence` and `usage`. Without a usable agent model (`agent.enabled: false`, no
key, or the planner call fails) search falls back to direct retrieval: `mode: "retrieval"` and
`agent.reason`. `POST /api/retrieve` always runs direct retrieval, and `POST /api/chat` is the
one-shot answer mode.

## Knowledge graph retrieval (`query_knowledge_graph`)

One call per planner intent (MCP tool and agent tool; `graph/retriever.py`). The spec's
`retrieve_documents` is the existing `retrieve_context` (unchanged signature and behaviour; its graph
evidence now also lists the proposed relationships and term relations, labelled with their status).

1. **Resolve** each retrieval query: an IRI in the text, then exact / normalized / plural /
   word-order-insensitive matches on labels, synonyms (`skos:altLabel`) and relationship names and
   synonyms; when nothing resolves, the glossary vector index (`embedding_retrieval_enabled`). Hits keep
   `match_type` and `matched_queries` and are merged by IRI across the intent's queries; a phrase naming
   several concepts is reported in `ambiguities`.
2. **Traverse** between the resolved entities: depth-first over a per-graph-version adjacency index, simple
   paths only, bounded by `max_hops`, `max_candidate_paths`, `max_expansions` and `query_timeout_seconds`.
   Named relationships go in their own direction (reverse only with an inverse name), navigational edges
   both ways (`references` / `isReferencedBy`), document relationships forward only. Rejected edges are
   never used; inferred ones only with `include_inferred_relationships`.
3. **Rank**: `0.6 × mean edge relevance + 0.25 × join coverage + 0.15 × mean status` (weights in
   `ranking_weights`). Edge relevance is 1 when the intent names the relationship or a synonym ("bought"),
   otherwise the local embedding similarity of the intent and the relationship's name, definition and
   synonyms. Join coverage: the foreign key behind the mapping is verified 1, inferred 0.5; an automatic
   mapping 0.5; none 0. Status: verified 1, inferred 0.6, navigational 0.3. Means keep the score
   length-neutral; fewer hops only break ties. Paths that use FK-derived navigational edges rank after
   paths that use none (fewer navigational edges first). Scores are not calibrated probabilities.
4. **Expand** the top `top_k` paths and the resolved concepts: entities (definition, alternate names,
   classification, tables), terms (owning entity, definition, synonyms, OWL property, mapped columns with
   data types, related terms), joins (ordered column pairs, join predicate, origin, status, confidence),
   keys and term relations, and references to the definitions and passages that mention them.

Request:

```json
{"intent_id": "I1", "intent": "customer bought product",
 "retrieval_queries": ["customer bought product", "products purchased by customers"],
 "max_hops": 5, "top_k": 1, "include_physical_mappings": true, "include_related_terms": false}
```

Response (shop test fixture, shortened):

```json
{"intent_id": "I1", "status": "ok",
 "matched_queries": ["customer bought product", "products purchased by customers"],
 "business_entities": [{"iri": "urn:mesh:entity:customer", "name": "Customer", "role": "matched",
                        "classification": ["owl:Class", "mesh:BusinessEntity"],
                        "tables": [{"iri": "urn:mesh:dataset:ds-65c9…", "name": "customers", "mapping_status": "inferred"}]},
                       "… Product (matched), Order and Order Line (path)"],
 "paths": [{"path_id": "P1", "hops": 3,
   "nodes": ["urn:mesh:entity:customer", "urn:mesh:entity:order", "urn:mesh:entity:order-line", "urn:mesh:entity:product"],
   "edges": [{"relationship_iri": "urn:mesh:relationship:sem-rel-6ab7…", "predicate": "placesOrder",
              "direction": "forward", "kind": "semantic", "status": "inferred", "join_refs": ["J1"]},
             "… containsOrderLine (J2), referencesProduct (J3)"],
   "label": "Customer -> [placesOrder] Order -> [containsOrderLine] Order Line -> [referencesProduct] Product",
   "executable": true, "missing_mappings": [], "validation_status": "inferred", "relevance_score": 0.638,
   "score_explanation": {"semantic_relevance": 0.4967, "join_coverage": 1.0, "status": 0.6, "hops": 3,
                         "edge_relevance": [1.0, 0.088, 0.402], "calibrated": false}}],
 "physical_mappings": {"tables": ["…"], "columns": ["…"],
   "joins": [{"join_id": "J1", "foreign_key_iri": "urn:mesh:key:key-9e8e…", "relationship": "placesOrder",
              "method": "explicit", "from_table": "orders", "to_table": "customers",
              "column_pairs": [{"position": 1, "from_column": "customer_id", "to_column": "customer_id"}],
              "join_predicate": "main.orders.customer_id = main.customers.customer_id",
              "origin": "declared", "status": "inferred", "foreign_key_status": "verified"}]},
 "constraints": [{"type": "primary_key", "table": "customers", "columns": ["customer_id"], "origin": "declared"}],
 "warnings": ["P1: inferred (unverified) relationship(s): containsOrderLine, placesOrder, referencesProduct."],
 "assertion_level": "schema",
 "note": "Schema-level knowledge: paths describe how data can be joined, not facts about particular records. …",
 "graph": {"version": 2, "named_graphs": ["urn:mesh:context:source:shop:mapping", "…"]}}
```

`status` is `ok`, `no_match` (all lists empty, a warning per unmatched query) or `error` (safe message).
A path missing a join is returned with `executable: false`, `missing_mappings` and a warning; joins are
never invented. Responses are bounded by `max_response_chars` (least important lists are cut first, the best
path always stays; `truncated` lists what was cut). The search scope (source / dataset / document filters)
limits what is resolved, traversed and returned. Telemetry: span `graph.query` (intent ID, lookup types,
query count, graph version, matched entities/terms, paths considered/returned, hops, duration, status,
warnings) and metrics `mesh_graph_queries`, `mesh_graph_query_duration`, `mesh_graph_paths`.

Ontology example (Turtle, from the published graph):

```turtle
<urn:mesh:term:birth-date> a mesh:BusinessTerm, skos:Concept ;
    skos:prefLabel "Birth Date" ; skos:altLabel "date of birth", "DOB" ;
    skos:inScheme <urn:mesh:scheme:glossary> ; mesh:termOfEntity <urn:mesh:entity:customer> ;
    mesh:termProperty <urn:mesh:property:birth-date> .
<urn:mesh:property:birth-date> a owl:DatatypeProperty ; rdfs:domain <urn:mesh:entity:customer> .
<urn:mesh:relationship:sem-rel-6ab7…> a mesh:Relationship, mesh:SemanticRelationship ;
    mesh:sourceEntity <urn:mesh:entity:customer> ; mesh:targetEntity <urn:mesh:entity:order> ;
    mesh:relationshipType "placesOrder" ; skos:altLabel "bought", "purchases" ; mesh:inverseName "placedBy" ;
    mesh:validationStatus "inferred" ; mesh:objectProperty <urn:mesh:predicate:rel-6ab7…> ;
    mesh:joinMapping <urn:mesh:join-mapping:…> ; prov:wasDerivedFrom <urn:mesh:key:key-9e8e…> .
<urn:mesh:term-relation:trel-…> a mesh:TermRelation ; mesh:subjectTerm <urn:mesh:term:settlement-date> ;
    mesh:predicate mesh:expectedAfter ; mesh:objectTerm <urn:mesh:term:transaction-date> ;
    mesh:assertionLevel "descriptive" ; mesh:validationStatus "inferred" .
```

**`execute_sql`** (MCP + agent; `structured/sql_exec.py`). Input: `source_id` (a cataloged database or
Databricks source), `sql`, optional `max_rows`. sqlglot parses the query in the source's dialect: exactly
one SELECT / WITH … SELECT, no SELECT INTO or locking clause, every table a cataloged dataset of that source
and inside the search scope. It then runs on the connector's read-only session with a statement timeout
where the dialect has one. Output: `{status: "ok", columns: [{name, type, sensitive, sensitive_terms}], rows,
row_count, truncated, …}` (`rows: []` when nothing matches) or `{status: "error", error: {code, message}}`.
A column is `sensitive: true` when its value is computed from a cataloged column whose business term is
flagged sensitive: the lineage is read with sqlglot through aliases, expressions, CTEs, derived tables,
scalar subqueries and UNION branches, and a column used only inside `COUNT(...)` does not count. `false`
when every contributing term was assessed as not sensitive (or nothing contributes, e.g. `COUNT(*)`); `null`
when a contributing term was never assessed or the lineage could not be read (then matched by output name).
Parquet sources are not supported. Row values never go to logs; every call is audit-logged.

**`generate_html`** (MCP + agent; `common/html_render.py`). Input: `title` and `blocks` (heading,
paragraph, list, citations, table with columns/rows or, in the agent, `result_ref` = an R evidence ID).
Output: `{status, html, blocks_rendered, truncated}`: one `<div class="km-html">` fragment built from a
fixed tag set with every value escaped. A table (e.g. an `execute_sql` result) is a grid,
`<table class="km-grid">` with a header row, drawn with cell borders on the Search page. A table block's
`sensitive_columns` (the agent fills them from the execute_sql result it renders) mark those headers and
cells `class="km-sensitive"`, shown in red, with a note under the table naming them. The Search page rebuilds it element by element from an allowlist
(`sanitizeFragment` in `app.js`) before showing it.

## Configuration

`config/mesh.yaml` is validated at startup (unknown keys are errors). Credentials are only
referenced: `connection_url_env`, `token_env`, `api_key_env`/`api_key_file` (env var wins).

```yaml
structured_sources:
  - id: crm
    kind: database
    connection_url_env: CRM_DATABASE_URL      # e.g. postgresql+psycopg://readonly@host/crm
    schemas: [public]                          # optional; default schema otherwise
    datasets: [customers, orders]              # optional allowlist ("table" or "schema.table")
  - id: customer_export
    kind: parquet
    path: ./data/sources/                      # file or directory of *.parquet
  - id: orders
    kind: databricks                           # pip install -e ".[databricks]"
    server_hostname_env: DATABRICKS_SERVER_HOSTNAME
    http_path_env: DATABRICKS_HTTP_PATH
    token_env: DATABRICKS_TOKEN
    catalog: sales
    schema: public
```

### Adding sources from the Settings page

**⚙ Settings** (top right of the UI) adds, tests, edits and removes structured sources without a
restart:

- **Database**: type (PostgreSQL, MySQL/MariaDB, SQL Server, Oracle), database name (Oracle:
  service name), hostname, port (the type's default when empty), optional schema (empty = every
  non-system schema; for MySQL, the connected database), username and password. Required fields
  are marked with *, and missing ones are outlined in red. Drivers: `pip install -e ".[databases]"`. The type *Other* takes a full SQLAlchemy URL
  instead, e.g. `sqlite:////abs/path.db`.
- **Parquet repository**: a `.parquet` file or a folder of them, under `storage.allowed_source_roots`
  (default `./data/sources`).
- **Databricks**: server hostname, HTTP path, catalog and schema, plus a token
  (`pip install -e ".[databricks]"`).

Credentials (the database password, a full connection URL or the Databricks token) are either an
**environment variable name** or a secret saved in the **OS keychain**
(`secrets.backend: keyring`; macOS Keychain, Windows Credential Locker, Linux Secret Service).
Stored secrets are never returned by the API, shown again, logged or written to SQLite.
Connection URLs are displayed with the password masked. *Test connection* lists the visible
datasets without saving anything. *Remove* retires the source's catalog records, glossary
vectors and graph evidence. Sources from `config/mesh.yaml` are listed read-only. Changes are
accepted only from the local machine unless an API key is configured (`api.api_key_env`). This
matters when the server is bound to `0.0.0.0`.

### Profiling scope and limits

| Setting | Default | Meaning |
|---|---|---|
| `profiling.max_rows` | 100000 | Rows read per dataset (first N). `0` = full scan. Per-call override: `discover_keys(max_rows=…)`. Results from fewer rows than the table holds are labeled `sampled: true`. |
| `profiling.max_key_size` | 4 | Largest unique column combination searched. |
| `profiling.max_fk_arity` | 3 | Largest composite foreign key searched. |
| `profiling.max_columns` | 64 | Columns considered for UCC search (in ordinal order). |
| `profiling.ucc_algorithm` | `hyucc` | `hyucc` or `apriori`. |
| `profiling.null_equals_null` | `true` | Nulls compare equal (conservative); `false` = SQL UNIQUE semantics. PK candidates always require zero nulls. |
| `profiling.key_excluded_type_keywords` | float, double, real, binary, blob, json, bytes | Column types never used in keys. |
| `profiling.binder_min_distinct_dependent` | 2 | Ignore near-constant dependent columns (flags/booleans). |
| `profiling.binder_buckets` | 16 | BINDER hash buckets. |
| `profiling.fk_identifiers_only` | `true` | Only identifier columns can form FKs (see above); `false` = any inclusion. |
| `profiling.fk_reference_primary_keys` | `true` | FK targets are primary keys (declared, else PK candidates); `false` = any unique key. |
| `profiling.fk_identifier_class_words` | Identifier, Code, Number, Key, Indicator, Flag | Business-term class words that mark an identifier. |
| `profiling.fk_identifier_name_tokens` | id, cd, code, no, num, nbr, key, ind, flg, flag, uuid, guid… | Last name token → class word, used while a column has no business term. |
| `profiling.max_ucc_candidates` / `max_nary_candidates` | 100000 / 10000 | Safety caps; hitting one marks the search `ucc_search_complete: false`. |
| `profiling.most_common_values`, `store_most_common_values` | 5, true | Most-common-value capture in profiles (set `false` to store none). |

Inferred keys are only as good as the evaluated scope. FK candidates whose referenced side was
sampled can miss matches. Select both sides of an expected relationship in the same `discover_keys` call.

### When settings take effect

Configuration is read at startup, so every change needs a restart. After the restart:

| Settings | Effect |
|---|---|
| `retrieval.*`, `crawler.*`, `answer_generation.*`, `agent.*`, `api.*`, `storage.max_upload_bytes`, `storage.allowed_*` | Apply to subsequent requests and jobs. |
| `prompts/planner.txt`, `executor.txt`, `synthesis.txt` content | Read on every search; no restart needed. |
| `profiling.*` | Next `discover_keys` / catalog run. Existing inferred keys keep their recorded scope until re-run. |
| `chunking.*`, `entity_extraction.*` | New and changed documents. Run `reindex` to re-chunk and re-extract existing documents. |
| `embedding.*` (model, revision, normalize) | **Reindex required.** An index built with another model refuses to open (`incompatible_index`). Run `reindex --rebuild`: old index directories are kept as `*.bak-<timestamp>`. |
| `term_generation.prompt_path` content | Prompt version changes, so the next catalog run regenerates every column's term. |
| `term_generation.provider/model/abbreviations/class_words` | **Regeneration required.** Run `POST /api/datasets/generate-terms {"force": true}`. |
| `knowledge_graph.shapes_path` | Validates subsequent graph updates. Re-run cataloging/indexing to re-validate existing contexts. |
| `knowledge_graph.*` (limits, `ranking_weights`, `include_inferred_relationships`, `embedding_retrieval_enabled`) | Subsequent `query_knowledge_graph` calls. |
| `relationship_generation.*`, `prompts/relationship_generation.txt` | Next catalog run or "Propose relationships". |
| `sql_execution.*`, `html_output.*`, `prompts/sql_generation.txt` | Subsequent tool calls (the prompt is read on every call). |
| `structured_sources` | Next catalog run. Removing a source does not delete its existing catalog records. |
| `storage.*_dir`, `vector_store.*`, `knowledge_graph.data_path` | Point at a different store (nothing is migrated). |
| `observability.*` | Applies at process start. |

## API, UI and MCP

The browser UI at `/` covers agent search (answer, plan, per-intent steps, evidence, HTML output, clarification
questions), uploads, crawls, the catalog, definitions and jobs. The Catalog page has a left facet with a tree of
databases (one per configured source, shown as `id (database)`), their schemas and their tables. The facet stays
visible on every catalog page; clicking an element shows it on the right: a **database** shows its description and
a grid of its schemas with their description (schema descriptions are not captured yet, so the column is blank);
a **schema** shows a grid of its tables with the business entity each one represents; a **table** shows a grid of
its columns (column name, data type, column data type, content type, data format, business term) followed by the
table details, column profiles, keys, relationships and review buttons (`#catalog/db/<source>`,
`#catalog/schema/<source>/<schema>`, `#dataset/<id>`). Right-clicking a
database, schema or table (or the context-menu key / Shift+F10 on it) opens its knowledge graph on the right: tables, columns,
keys with their column pairs, entities, terms, relationships and term relations, coloured by type, dashed when
inferred (`GET /api/graph/view?source_id=&schema=&dataset_id=`, drawn with the vendored cytoscape.js; a list view
shows the same content). API routes
are the ones in spec §7, plus `/api/retrieve`, `/api/chunks/{id}`, `/api/documents/{id}/text|original`,
`/api/business-terms/{slug}`, `/api/business-entities/{slug}`, `/api/datasets/discover-keys`,
`/api/datasets/generate-terms`, `/api/profiling-runs/{id}`, `/api/jobs/{id}/items`, `/metrics`,
`/api/searches/{trace_id}/clarification`, `/api/reviews`, `/api/datasets/{id}/relationships`,
`/api/datasets/generate-relationships`, `/api/graph/view` and `/api/graph/reload` (local caller or API key).
Errors are `{"error": {"code", "message", "details"}, "trace_id"}`. Every response has `X-Trace-Id`.
Set the env var named by `api.api_key_env` (default `KNOWLEDGE_MESH_API_KEY`) to require
`X-API-Key` on `/api/*`. The server binds to 127.0.0.1 by default.

MCP tools return bounded results with `ok`, `trace_id` and references (`job_id`, `manifest_ref`,
`/api/...`), and structured errors. `query_knowledge_graph`, `execute_sql` and `generate_html` publish input and
output schemas (`tool_schemas.py`); their descriptions state input, purpose and output. Document retrieval is
`retrieve_context` (the spec's `retrieve_documents`). They accept an optional W3C `traceparent` to join a caller's
trace, and `wait_seconds` to block until a job finishes. Local paths must be under
`storage.allowed_ingest_roots`.

```json
{"mcpServers": {"knowledge-mesh": {"command": "/path/to/knowledge_mesh/.venv/bin/python",
  "args": ["/path/to/knowledge_mesh/src/harness.py", "mcp"]}}}
```

## Observability

```bash
# once: install Podman (https://podman.io, macOS arm64 installer) and docker-compose (podman compose uses it), then
podman machine init --cpus 4 --memory 4096   # /opt/podman/bin/podman if it is not on your PATH
scripts/observability.sh up                 # Podman VM + OTel Collector, Phoenix, Loki, Prometheus, Alertmanager, Grafana
scripts/mesh.sh start                       # the stack and the app (port 9801, logs in data/logs/server.log)
```

UIs: **Phoenix** http://localhost:6006 (traces, LLM conversations, evaluations, feedback), **Grafana**
http://localhost:3000 (dashboards "Knowledge Mesh" and "Knowledge Mesh: agent quality, cost and safety", Loki
logs; admin password `GRAFANA_ADMIN_PASSWORD`, default `change-me`), **Prometheus** http://localhost:9090,
**Alertmanager** http://localhost:9093. `scripts/observability.sh up` creates the Alertmanager webhook token
in `deploy/secrets/` (git-ignored). Phoenix authentication is off because every port is bound to
127.0.0.1 (Phoenix 21 otherwise rejects OTLP traces without an API key); turn it on before exposing the stack.
If the collector is not reachable at startup, trace and log export are off for that run (one warning); if it
goes away later, the app logs one clear warning per 5 minutes instead of one per batch. `/metrics` works either way.

| Area | What is recorded | Where |
|---|---|---|
| Execution tracing | One `agent.search` span per search (`user.id`, `session.id` per browser tab, prompt versions); `agent.plan`, `agent.replan`, one `agent.intent` span per task run linked to the spans of the tasks it depends on; `agent.decision` events (next / replan / finish) and `agent.step` events in order | Phoenix (sessions, trace tree), "open trace in Phoenix" on the Search page |
| LLM | Provider, model, role (`agent_planner`, `agent_step`, `agent_synthesis`, `agent_judge`, …), OpenInference input/output messages (when `capture_sensitive_content` is on), invocation parameters, prompt-file hash, tokens, latency, retries, error type, estimated cost | Phoenix; `mesh_llm_*` metrics |
| Tools | MCP and agent tools: name, sanitized parameters, status, error code, latency, result size | `agent.tool.*` / `mcp.*` spans; `mesh_tool_calls_total{source}`, `mesh_tool_duration_seconds` |
| Context | Per executor call: characters and estimated tokens, steps and results shown, results replaced by stubs, truncated results, `seen_before` items, system-prompt version; the agent has no memory across searches | `agent.context` events; `mesh_agent_context_*` |
| RAG | Queries, retrieved IDs, scores and text, merged-duplicate ratio, Laya decisions; citation coverage and invalid citations; judge faithfulness and context relevance | retrieval spans; `mesh_retrieval_top_score`, `mesh_eval_citation_coverage` |
| Planning | Plan JSON per version, replan reasons, fallbacks, decisions, the executor's evaluation text (when capture is on) | Phoenix; the Search page; `mesh_agent_plans`, `mesh_agent_recoveries` |
| Quality | Rule checks on every search (plan valid, planned queries searched, valid replies, iteration cap, tasks answered, citation coverage, invalid citations, evidence, relevance, sensitive data, flagged passages, injection) and a compliance score; the LLM judge on every search (`observability.online_judge`, `prompts/judge.txt`): faithfulness, unsupported claims, answer relevance, completeness, context relevance, plan adherence | Search page "Evaluation" panel, `GET /api/searches/{trace_id}`, Phoenix annotations (`CODE`, `LLM`), `mesh_eval_*`, `mesh_judge_score`, `mesh_hallucinations_total` |
| Performance and cost | End-to-end and per-step latency, tokens and cost per search (set `agent.input_cost_per_1k_tokens` / `output_cost_per_1k_tokens`), CPU, memory, DB size, index nodes, pending jobs, threads | response `usage`, `searches` table, Grafana |
| Security | Caller identity (client address + API-key fingerprint), audit trail of searches (documents, datasets, tools touched), denied requests, feedback and settings changes; PII/secret scan of answers; injection scan of every tool result; policy violations | `audit_events` table, `GET /api/audit`, `knowledge_mesh.audit` logs in Loki, `mesh_policy_violations_total`, `mesh_access_denied_total` |
| Operations | Errors, LLM timeouts and rate limits, recoveries (invalid replies, tool errors, replans, iteration caps, plan and retrieval fallbacks), user feedback (👍/👎 + comment → Phoenix `HUMAN` annotation), alert rules in `deploy/alerts.yml` → Alertmanager → `POST /api/ops/alerts` | Search page alert banner, `GET /api/ops/alerts`, Grafana |

Logs are JSON with trace/span IDs. `observability.capture_sensitive_content: true` adds prompts, passages and
answers to spans (the page header warns when it is on); keep it off outside local debugging.

## Evaluation

```bash
.venv/bin/python src/harness.py eval evaluation/my_testset.jsonl --k 5 [--ragas]
```

The test set is JSON Lines: `question` plus `relevant_document_ids` or `relevant_chunk_ids`
(optional `reference_answer`); see `evaluation/testset.example.jsonl`. The command reports
Recall@K, hit rate, MRR and nDCG@K. Results go to `data/evaluations/` with the config hash,
embedding model and revision, index version and git commit. `--ragas` adds faithfulness and
answer relevancy (needs answer mode, the `eval` extra and a Ragas evaluator LLM). Acceptance
criterion 8 (Recall@5 ≥ 80% on ≥ 50 questions) needs your own labeled corpus. Record the corpus
and methodology alongside the result file.

**Agent evaluation.** `eval --agent TESTSET.jsonl [--min-pass-rate 0.8 --min-compliance 0.8
--min-faithfulness 0.7] [--ragas]` runs every question through the full agent search with the LLM judge
run synchronously. A case is JSON with `question` and optionally `expected_intents`, `expected_datasets`,
`relevant_document_ids` and `reference_answer`. Each case is scored on outcome, rule compliance, citation
coverage, judge scores, unsupported claims, document and dataset recall, intents, latency, tokens and cost,
and passes or fails with its reasons. The command exits with status 1 when the pass rate is below
`--min-pass-rate`, so it can gate CI. The searches share `session.id = eval-<run>` in Phoenix; results go to
`data/evaluations/agent-eval-<timestamp>.json`. `--ragas` adds Ragas faithfulness and response relevancy
(`pip install -e ".[eval]"`), judged by the configured agent model (Gemini, or any OpenAI-compatible model)
with the local embedding model, so no separate Ragas key is needed.

## Implementation decisions (spec §12)

- **First relational connector:** SQLAlchemy, so SQLite, PostgreSQL, MySQL and SQL Server work
  through a connection URL from an environment variable. Databricks uses a token from an
  environment variable; Parquet uses a local path.
- **Profiling and null handling:** see the table above. HyUCC drives discovery and validates in
  Apriori order; pure Apriori is a configurable alternative and the test oracle.
- **Table-level entity:** the decoded table name returned by the prompt, falling back to a heuristic (see above).
- **Document entity extraction:** deterministic lexicon by default (configurable lexicon + catalog
  entity labels; relationship verbs within one sentence; passive voice reversed); optional LLM mode.
- **Concept matching:** label slug across sources. Initial SHACL shapes: `config/knowledge_graph_shapes.ttl`.
- **Graph file:** TriG (`data/knowledge_graph.trig`) instead of Turtle, so named graphs keep per-source provenance.
- **Answer model:** not selected. Enable `answer_generation` with `provider: google|openai` and a model.
- **Agent model:** `agent.provider/model` (Gemini by default in `config/mesh.yaml`). Tool calls use a JSON
  action protocol (not provider function calling), so Gemini, OpenAI-compatible models and the test fake
  share one code path; the planner uses the provider's JSON-schema output mode.

- **Knowledge graph / GraphRAG requirements (doc/knowledge_graph_graphrag_implementation_requirements.txt),
  decided with the user:** `retrieve_context` is the spec's `retrieve_documents`; new tools have one
  implementation registered on the MCP server and in the agent tool box; SQL comes from an agent-only
  `generate_sql` tool using the agent model; `execute_sql` keeps the existing protection (API key, read-only
  session, search scope) plus a sqlglot check, for database and Databricks sources; named relationships, term
  relations and synonyms are LLM proposals made at catalog time by the term model; FK-derived edges are
  navigational; join mappings are explicit + automatic; relevance is aliases + local embeddings; verify /
  reject actions for any API caller; per-source × layer named graphs, migrated at startup; full OWL + SKOS
  model; deterministic HTML with a browser allowlist sanitizer, triggered by a checkbox or the question;
  clarification resumes through the replan step with state in SQLite; one database node per source;
  vendored cytoscape.js for the graph view.

## Known limitations

- Term relations (expectedAfter, …) are descriptive: no rule engine validates them against records.
- Access control is the existing model (API key, read-only sessions, search scope filters); there are no
  per-user source, row or column permissions.
- `execute_sql` checks tables, not columns; a wrong column fails at execution with a safe error.
- Parquet sources cannot be queried with `execute_sql`.
- Proposed relationships depend on the term model; until a steward verifies them they stay `inferred`,
  and paths through them carry a warning.
- The graph view draws at most `knowledge_graph.view_max_nodes` nodes.

## Acceptance coverage

Automated tests (`tests/`) exercise criteria 1, 3–7, 9–11, 13, 17–22 and the API/MCP surface,
including the 50+ page crawl (2) against a mocked site. Criterion 12 is covered at the
instrumentation level (trace IDs on every response, metrics, structured errors). Inspecting
correlated traces, metrics and logs needs the Docker stack. These need real infrastructure or data:

- 8: labeled corpus for Recall@5.
- 14: manual browser walkthrough.
- 15: an answer model.
- 16: live PostgreSQL/Databricks endpoints (SQLite and Parquet are tested; Databricks is tested
  with a fake driver).
- 19: the configured Gemini model (tested with a deterministic fake LLM).

Knowledge graph / GraphRAG / MCP requirements (AT-01 … AT-23):

| Test | Covers |
|---|---|
| `tests/test_mcp_kg_tools.py` | AT-01, AT-14 |
| `tests/test_planner_contract.py` | AT-02, AT-03 |
| `tests/test_agent_clarification.py` (+ the existing agent tests, unmodified) | AT-04, AT-15, AT-23 |
| `tests/test_kg_query.py` | AT-05 … AT-13, AT-16 |
| `tests/test_kg_model.py` | AT-17, AT-18, layers, OWL/SKOS, reviews |
| `tests/test_sql_html_tools.py` | AT-21, AT-22 |
| `tests/test_catalog_ui_api.py` | AT-19, AT-20 (API, static assets, app.js parse), catalog grids, generic data type, grid HTML; clicks in the browser need a walkthrough |
| `tests/test_sensitive_data.py` | sensitive term flag (prompt, parsing, storage, migration), execute_sql lineage marking, red HTML grid, MCP |

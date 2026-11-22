# Agent Loop — Implementation Requirements

## 1. Purpose

The agent loop controls the execution of user requests.

The harness uses an LLM planner to identify user intents.

The executor LLM uses MCP tools to complete the intents.

The harness controls the execution loop.

## 2. Planner

1. Receive the user query.
2. Send the user query to the planner LLM.
3. Get the planner response in JSON format.
4. Make sure that the JSON contains a list of intents.
5. Make sure that each intent contains a list of `retrieval_queries`.
6. Make sure that each intent contains a `depends_on` list.
7. Do not change the original user query.

Each intent can have one or more retrieval queries.

The planner can return one intent or multiple intents.

## 3. Agent Loop

1. Start the agent loop after the planner returns its response.
2. Give the planner response to the executor LLM.
3. Give the executor LLM the available MCP tool definitions.
4. Ask the executor LLM to select the next action.
5. Check the executor LLM response.
6. If the response contains a tool call, execute the tool.
7. Get the tool result.
8. Add the tool result to the conversation context.
9. Send the updated context to the executor LLM.
10. Repeat steps 4 through 9 until the executor LLM gives a final response.
11. Stop the loop when the executor LLM gives a final response.
12. Return the final response to the user.

## 4. MCP Tool Selection

The MCP server provides these tools:

- `query_knowledge_graph`
- `retrieve_documents`

Use `query_knowledge_graph` to get business entities, business terms, and their relationships.

Use `retrieve_documents` to get information from documents.

The executor LLM selects the tool.

The executor LLM can call the same tool more than one time.

The executor LLM can call different tools for the same intent.

Do not call a tool when the available context is sufficient.

## 5. Knowledge Graph Retrieval

1. Send the applicable retrieval query to `query_knowledge_graph`.
2. Identify the business entities and business terms.
3. Find the relationships between the entities.
4. Search for intermediate entities when necessary.
5. Get the applicable physical tables and columns.
6. Get the applicable PK/FK relationships.
7. Return the relevant graph paths and metadata.

Do not create relationships that do not exist in the knowledge graph.

## 6. Document Retrieval

1. Send the applicable retrieval query to `retrieve_documents`.
2. Get the relevant document sections.
3. Add the document sections to the executor context.
4. Keep the source references.

Do not change the existing `retrieve_documents` tool.

## 7. Multiple Intents

1. Read the `depends_on` list for each intent.
2. Identify the intents that have no unfinished dependencies.
3. Execute the applicable intents.
4. Store the results for each completed intent.
5. Give the necessary results to dependent intents.
6. Continue until all required intents are complete.

Do not execute an intent before its dependencies are complete.

## 8. Error Handling

1. Check each tool result for errors.
2. If a tool fails, give the error to the executor LLM.
3. Let the executor LLM select the next action.
4. Do not repeat a failed tool call without a reason.
5. Stop the loop when the configured execution limit is reached.
6. Report incomplete results when the task cannot finish.

## 9. Final Response

1. Check the collected results.
2. Do not add unsupported information.
3. Identify missing information.
4. Prepare the final response.
5. Return the response to the user.

## 10. Implementation Constraints

- Keep the existing harness loop unchanged.
- Keep the existing `retrieve_documents` tool unchanged.
- Add the `query_knowledge_graph` MCP tool.
- Keep the planner and executor prompts separate.
- Let the executor LLM select the MCP tools.
- Do not create a separate agent loop for each retrieval query.
- Do not require a tool call in each loop iteration.
- Keep the existing termination and error-handling controls.

## 11. Acceptance Criteria

1. The planner returns valid JSON.
2. Each intent contains its retrieval queries.
3. The executor receives the planner response.
4. The executor can use both MCP tools.
5. The knowledge graph tool supports multi-hop retrieval.
6. The executor can make multiple tool calls.
7. Tool results become available to the executor.
8. The agent loop stops after the final response.
9. The system handles tool errors.
10. The existing harness loop continues to operate without modification.
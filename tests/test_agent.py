"""Search agent harness: planner -> orchestration loop (executor loop per task, evaluate, replan) -> answer."""

import json
import re
import threading

import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.agent.plan import PLAN_SCHEMA, parse_json_object
from knowledge_mesh.agent.mcp_client import fit_observation
from knowledge_mesh.api.app import create_app
from knowledge_mesh.common.errors import InvalidArgument
from knowledge_mesh.config import AgentSettings, load_config
from knowledge_mesh.llm.client import FakeLLM
from tests.conftest import REPO

QUESTION = ("What is Customer Lifetime Value, how is it calculated, and which datasets contain the required "
            "information?")
CLV_DOC = ("# Customer Lifetime Value\nCustomer Lifetime Value (CLV) is the total revenue a customer generates "
           "over the whole relationship. It is calculated as the sum of all order amounts of the customer.\n")
I1Q = ["Customer Lifetime Value definition", "CLV business meaning"]
I2Q = ["Customer Lifetime Value calculation formula", "CLV calculation methodology"]
I3Q = ["CLV source datasets and columns", "Customer Lifetime Value physical data mappings"]
CLV_PLAN = {
    "original_query": QUESTION,
    "decompose": True,
    "intents": [
        {"id": "I1", "intent": "Find the business definition of CLV", "retrieval_queries": I1Q, "depends_on": []},
        {"id": "I2", "intent": "Find the CLV calculation methodology", "retrieval_queries": I2Q, "depends_on": []},
        {"id": "I3", "intent": "Identify datasets used to calculate CLV", "retrieval_queries": I3Q,
         "depends_on": ["I2"]},
    ],
    "reasoning": "Three related outcomes are requested.",
    "execution": "mixed",
}
SINGLE_Q = ["customer lifetime value", "CLV"]
SINGLE = {"original_query": "customer lifetime value", "decompose": False, "reasoning": "One objective.",
          "execution": "single",
          "intents": [{"id": "I1", "intent": "Find customer lifetime value", "retrieval_queries": SINGLE_Q,
                       "depends_on": []}]}
TOOLS = ["retrieve_documents", "read_document", "find_datasets", "get_dataset", "get_business_term",
         "get_business_entity", "get_business_definition"]
PROMPTS = {"planner_prompt_path": str(REPO / "prompts" / "planner.txt"),
           "executor_prompt_path": str(REPO / "prompts" / "executor.txt"),
           "synthesis_prompt_path": str(REPO / "prompts" / "synthesis.txt")}


def call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def tools(*calls):
    return {"evaluation": "I need more evidence.", "action": "tools", "tool_calls": list(calls)}


def complete(answer, status="answered"):
    return {"evaluation": "The evidence answers the task.", "action": "complete", "status": status, "answer": answer}


def first_id(prompt, prefix):
    return re.findall(rf'"id": "({prefix}\d+)"', prompt)[0]


def dataset_id(prompt, name):
    return re.search(rf'"dataset_id": "([^"]+)", "name": "{name}"', prompt).group(1)


def last_result(prompt, n=1):
    """The JSON tool result ``Result n`` of the newest step shown in an executor prompt."""
    return json.loads(re.findall(rf"^Result {n}: (\{{.*\}})$", prompt, re.M)[-1])


class Script:
    """FakeLLM responder: plan, replan, executor actions per task (by step number), synthesis."""

    def __init__(self, plan, actions, replan=None, synthesis=None):
        self.plan, self.actions, self.replan, self.synthesis = plan, actions, replan, synthesis

    @staticmethod
    def _dump(item, prompt):
        item = item(prompt) if callable(item) else item
        return item if isinstance(item, str) else json.dumps(item)

    def __call__(self, prompt, system, purpose):
        if purpose == "agent_planner":
            return self._dump(self.plan, prompt)
        if purpose == "agent_replanner":
            return self._dump(self.replan, prompt)
        if purpose == "agent_step":
            task = re.search(r"^## Current task\n(\S+):", prompt, re.M).group(1)
            n = len(re.findall(r"^### Step \d+$", prompt, re.M))
            script = self.actions[task]
            script = script(prompt) if callable(script) else script
            return self._dump(script[min(n, len(script) - 1)], prompt)
        if purpose == "agent_synthesis":
            if self.synthesis:
                return self.synthesis(prompt)
            ids = re.findall(r"^\[([A-Z]\d+)\]", prompt, re.M)
            return f"Final answer [{ids[0]}] [Z99]." if ids else "Nothing was found."
        raise AssertionError(purpose)


@pytest.fixture
def agent_mesh(mesh_factory, config_factory, sqlite_source, fake_term_llm):
    def make(script, agent=None, catalog=True, document=True):
        settings = {"enabled": True, "provider": "fake", "model": "fake", **PROMPTS, **(agent or {})}
        llm = FakeLLM(script)
        mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source], "agent": settings}),
                            term_llm=fake_term_llm, agent_llm=llm)
        if document:
            mesh.ingest_document(content=CLV_DOC.encode(), filename="clv.md", wait_seconds=30)
        if catalog:
            mesh.catalog_source("crm", wait_seconds=120)
        llm.calls.clear()  # keep only the search's calls
        return mesh, llm
    return make


def spy_retrieve(mesh, monkeypatch):
    """The query lists the MCP tool retrieve_documents searched (it calls KnowledgeMesh.retrieve_documents)."""
    seen, real, lock = [], mesh.retrieve_documents, threading.Lock()

    def spy(queries, *a, **k):
        with lock:
            seen.append(list(queries))
        return real(queries, *a, **k)
    monkeypatch.setattr(mesh, "retrieve_documents", spy)
    return seen


def clv_script():
    barrier = threading.Barrier(2, timeout=10)

    def together(action):  # I1 and I2 must both be in their first executor call at the same time
        def act(prompt):
            barrier.wait()
            return action
        return act

    return Script(CLV_PLAN, {
        "I1": [together(tools(call("retrieve_documents", queries=I1Q, evidence_types=["documents"]))),
               lambda p: complete(f"CLV is the total revenue a customer generates [{first_id(p, 'D')}].")],
        "I2": [together(tools(call("retrieve_documents", queries=I2Q))),
               lambda p: complete(f"CLV is the sum of all order amounts of the customer [{first_id(p, 'D')}].")],
        "I3": [tools(call("find_datasets", name_contains="order"), call("find_datasets", column_contains="amount")),
               lambda p: tools(call("get_dataset", dataset_id=dataset_id(p, "orders"))),
               lambda p: complete(f"orders.amount per orders.customer_id [{first_id(p, 'T')}] [X42].")],
    })


def step_calls(llm, task=None):
    return [c for c in llm.calls if c["purpose"] == "agent_step"
            and (task is None or f"## Current task\n{task}:" in c["prompt"])]


def test_clv_plan_runs_ready_tasks_together_then_dependents_then_synthesizes(agent_mesh, monkeypatch):
    mesh, llm = agent_mesh(clv_script())
    searched = spy_retrieve(mesh, monkeypatch)
    res = mesh.search(QUESTION)
    assert res["mode"] == "agent"
    assert {k: res["plan"][k] for k in CLV_PLAN} == CLV_PLAN and res["plan"]["version"] == 1
    assert len(res["plans"]) == 1

    # orchestration: I1 + I2 together (the barrier proves it), then I3, then finish
    assert [(d["after"], d["decision"]) for d in res["decisions"]] == [(["I1", "I2"], "next"), (["I3"], "next"),
                                                                       ([], "finish")]
    assert [(t["id"], t["status"], t["attempt"]) for t in res["tasks"]] == [
        ("I1", "answered", 1), ("I2", "answered", 1), ("I3", "answered", 1)]
    assert sorted(searched) == sorted([I1Q, I2Q])  # each task's retrieval queries, as planned

    i3 = res["tasks"][2]
    assert [c["tool"] for c in i3["steps"][0]["tool_calls"]] == ["find_datasets", "find_datasets"]  # one turn
    assert i3["dropped_citations"] == ["X42"] and "[X42]" not in i3["answer"]
    i3_first = step_calls(llm, "I3")[0]["prompt"]
    assert "## Results of the tasks this one depends on" in i3_first and "I2 (answered)" in i3_first

    purposes = [c["purpose"] for c in llm.calls]
    assert purposes[0] == "agent_planner" and purposes[-1] == "agent_synthesis"
    assert purposes.count("agent_synthesis") == 1 and "agent_replanner" not in purposes
    assert res["answer"].startswith("Final answer") and res["dropped_citations"] == ["Z99"]
    assert res["citations"] and all(c["ref"].startswith("/api/") for c in res["citations"])
    assert {"document_passage", "dataset"} <= {e["type"] for e in res["evidence"]}
    synthesis = llm.calls[-1]["prompt"]
    assert QUESTION in synthesis and all(f"{t} (answered)" in synthesis for t in ("I1", "I2", "I3"))


def test_planner_gets_no_tools_and_every_executor_call_gets_the_intents_and_tools(agent_mesh):
    mesh, llm = agent_mesh(clv_script())
    mesh.search(QUESTION)
    planner = llm.calls[0]
    assert planner["purpose"] == "agent_planner" and planner["json_schema"] == PLAN_SCHEMA
    assert planner["system"] == (REPO / "prompts" / "planner.txt").read_text().strip()
    assert "9. Write each intent description" in planner["system"] and "10. For each intent" in planner["system"]
    assert QUESTION in planner["prompt"]
    assert all(f'"{f}"' in planner["prompt"] for f in ("original_query", "retrieval_queries", "decompose", "execution"))
    assert not any(name in planner["prompt"] or name in planner["system"] for name in TOOLS)
    executor = (REPO / "prompts" / "executor.txt").read_text().strip()
    for c in step_calls(llm):
        assert c["system"] == executor and c["json_output"] is True
        assert all(f'"name": "{name}"' in c["prompt"] for name in TOOLS)
        assert all(f"- {i['id']}: {i['intent']}" in c["prompt"] for i in CLV_PLAN["intents"])
        assert all(json.dumps(q) in c["prompt"] for q in I1Q + I2Q + I3Q)
        assert QUESTION not in c["prompt"]
    assert not any("most_common_values" in c["prompt"] for c in llm.calls)


def test_planned_queries_are_always_searched_first(agent_mesh, monkeypatch):
    script = Script(SINGLE, {"I1": [
        tools(call("retrieve_documents", queries=[QUESTION], evidence_types=["bogus"])),  # rejected: does not count
        tools(call("retrieve_documents", query=QUESTION)),                                 # single-query form
        tools(call("retrieve_documents", queries=["order amounts"])),
        complete("Done [D1]."),
    ]})
    mesh, llm = agent_mesh(script, catalog=False)
    searched = spy_retrieve(mesh, monkeypatch)
    res = mesh.search("customer lifetime value")
    # the rejected first call (the server refuses the evidence type) is retried with the planned queries
    assert searched == [SINGLE_Q + [QUESTION], SINGLE_Q + [QUESTION], ["order amounts"]]
    steps = res["tasks"][0]["steps"]
    assert steps[0]["tool_calls"][0]["error"]["code"] == "invalid_argument"
    second = steps[1]["tool_calls"][0]
    assert second["arguments"]["queries"] == SINGLE_Q + [QUESTION] and "planned retrieval queries" in second["note"]
    assert steps[2]["tool_calls"][0]["arguments"] == {"queries": ["order amounts"]}
    assert "note" not in steps[2]["tool_calls"][0]


def test_multi_query_results_are_merged_and_items_shown_before_are_compacted(agent_mesh):
    plan = {**SINGLE, "intents": [{**SINGLE["intents"][0],
                                   "retrieval_queries": ["Customer Lifetime Value", "Customer Lifetime Value revenue"]}]}
    script = Script(plan, {"I1": [tools(call("retrieve_documents", queries=[])),
                                  tools(call("retrieve_documents", queries=["lifetime value of a customer"])),
                                  complete("Done.")]})
    mesh, llm = agent_mesh(script, catalog=False)
    mesh.search("customer lifetime value")
    first = last_result(step_calls(llm)[1]["prompt"])
    chunk = first["documents"][0]
    assert chunk["matched_queries"] == ["Customer Lifetime Value", "Customer Lifetime Value revenue"]
    assert len({d["chunk_id"] for d in first["documents"]}) == len(first["documents"])  # one entry per chunk
    again = last_result(step_calls(llm)[2]["prompt"])
    repeat = next(d for d in again["documents"] if d["id"] == chunk["id"])
    assert repeat["seen_before"] is True and "text" not in repeat


def _replan_script(second_attempt_status="answered", replan=None):
    revised = {**CLV_PLAN, "intents": [
        {"id": "I1", "intent": "Find the business definition of CLV",
         "retrieval_queries": ["Customer Lifetime Value"], "depends_on": []},
        CLV_PLAN["intents"][1]]}

    def i1(prompt):
        if '"Customer Lifetime Value"' in prompt.split("## Current task")[1].split("##")[0]:  # attempt 2
            return [complete("CLV is the total revenue of a customer [D1].", second_attempt_status)]
        return [complete("No definition found.", "not_found")]

    plan = {**CLV_PLAN, "execution": "parallel", "intents": CLV_PLAN["intents"][:2]}
    return Script(plan, {"I1": i1, "I2": [complete("Sum of order amounts [D1].")]},
                  replan=revised if replan is None else replan)


def test_unanswered_task_is_replanned_and_answered_tasks_are_not_rerun(agent_mesh):
    mesh, llm = agent_mesh(_replan_script(), catalog=False)
    res = mesh.search(QUESTION)
    assert [(d["after"], d["decision"]) for d in res["decisions"]] == [
        (["I1", "I2"], "replan"), (["I1"], "next"), ([], "finish")]
    assert "I1 not_found" in res["decisions"][0]["reason"] and res["decisions"][0]["plan_version"] == 2
    assert [(t["id"], t["attempt"], t["plan_version"], t["status"]) for t in res["tasks"]] == [
        ("I1", 1, 1, "not_found"), ("I2", 1, 1, "answered"), ("I1", 2, 2, "answered")]
    assert len(res["plans"]) == 2 and res["plans"][1]["replan_reason"].startswith("I1 not_found")
    replanner = next(c for c in llm.calls if c["purpose"] == "agent_replanner")
    assert replanner["system"] == (REPO / "prompts" / "planner.txt").read_text().strip()
    assert QUESTION in replanner["prompt"] and "I1 (not_found, attempt 1)" in replanner["prompt"]
    assert "I2 (answered, attempt 1)" in replanner["prompt"] and replanner["json_schema"] == PLAN_SCHEMA


def test_replans_are_capped_and_an_invalid_replan_keeps_the_plan(agent_mesh):
    mesh, llm = agent_mesh(_replan_script(second_attempt_status="not_found"), catalog=False)
    res = mesh.search(QUESTION)
    assert [c["purpose"] for c in llm.calls].count("agent_replanner") == 1
    assert res["decisions"][1]["decision"] == "next" and "no replan left" in res["decisions"][1]["reason"]

    mesh, llm = agent_mesh(_replan_script(replan="not a plan"), catalog=False, document=False)
    res = mesh.search(QUESTION)
    assert res["decisions"][0]["decision"] == "next" and "replan failed" in res["decisions"][0]["reason"]
    assert len(res["plans"]) == 1 and [t["id"] for t in res["tasks"]] == ["I1", "I2"]


def test_iteration_cap_is_per_task_and_triggers_a_replan(agent_mesh):
    assert AgentSettings().max_iterations == 20
    assert load_config(REPO / "config" / "mesh.yaml").agent.max_iterations == 20
    script = Script(SINGLE, {"I1": [tools(call("retrieve_documents", queries=["customer"]))]}, replan=SINGLE)
    mesh, llm = agent_mesh(script, agent={"max_iterations": 3}, catalog=False)
    res = mesh.search("customer lifetime value")
    assert [(t["status"], t["iterations"]) for t in res["tasks"]] == [("max_iterations", 3), ("max_iterations", 3)]
    assert len(step_calls(llm)) == 6  # 3 per task run, before and after the replan
    assert res["decisions"][0]["decision"] == "replan" and "max_iterations" in res["decisions"][0]["reason"]
    assert "This is the last iteration" in step_calls(llm)[2]["prompt"]
    assert llm.calls[-1]["purpose"] == "agent_synthesis" and res["answer"]


def test_invalid_replies_and_tool_errors_are_fed_back(agent_mesh):
    script = Script(SINGLE, {"I1": [
        "not json at all",
        tools(call("get_dataset", dataset_id="missing")),
        tools(call("nope")),
        tools(call("find_datasets", limit=500)),
        tools(*[call("find_datasets")] * 5),  # more than max_tool_calls_per_step
        complete("Gave up: no data found.", "not_found"),
    ]})
    mesh, llm = agent_mesh(script, agent={"max_replans": 0}, catalog=False, document=False)
    res = mesh.search("customer data")
    steps = res["tasks"][0]["steps"]
    assert [s["action"] for s in steps] == ["invalid", "tools", "tools", "tools", "invalid", "complete"]
    assert [s["tool_calls"][0]["error"]["code"] for s in steps[1:4]] == ["not_found", "unknown_tool", "invalid_argument"]
    assert "at most 4 tool calls" in steps[4]["error"]["message"]
    prompts = [c["prompt"] for c in step_calls(llm)]
    assert "invalid_action" in prompts[1] and "dataset not found" in prompts[2] and "unknown tool 'nope'" in prompts[3]
    assert res["tasks"][0]["status"] == "not_found" and res["tasks"][0]["iterations"] == 6


@pytest.mark.parametrize("plan, reason", [
    ("this is not json", "no JSON object"),
    ({**CLV_PLAN, "intents": [{"id": "I1", "intent": "a", "depends_on": ["I2"]},
                              {"id": "I2", "intent": "b", "depends_on": ["I1"]}]}, "cycle"),
    ({**CLV_PLAN, "intents": [{"id": "I1", "intent": "a", "depends_on": ["I9"]}]}, "unknown intent I9"),
    ({**CLV_PLAN, "intents": [{"id": f"I{i}", "intent": "x"} for i in range(6)]}, "at most 5"),
    ({"decompose": True}, "intents"),
])
def test_unusable_plan_falls_back_to_one_intent(agent_mesh, plan, reason):
    mesh, llm = agent_mesh(Script(plan, {"I1": [complete("Nothing to look up.")]}), catalog=False, document=False)
    res = mesh.search("customer lifetime value")
    assert reason in res["plan"]["fallback_reason"]
    task = res["tasks"][0]
    assert (task["id"], task["intent"], task["retrieval_queries"]) == (
        "I1", "customer lifetime value", ["customer lifetime value"])


def test_plan_defaults_and_retrieval_query_normalization(agent_mesh):
    plan = {"intents": [{"id": "I1", "description": "Define CLV",  # earlier key; other fields left out
                         "retrieval_queries": [" CLV  meaning ", "clv meaning", "", "a", "b", "c"]},
                        {"id": "I2", "intent": "List order tables", "retrieval_queries": []}]}
    mesh, llm = agent_mesh(Script(plan, {"I1": [complete("x")], "I2": [complete("y")]}), catalog=False,
                           document=False)
    res = mesh.search("define CLV and list order tables")
    p = res["plan"]
    assert p["original_query"] == "define CLV and list order tables" and p["execution"] == "mixed" and p["decompose"]
    assert [i["retrieval_queries"] for i in p["intents"]] == [["CLV meaning", "a", "b"], ["List order tables"]]
    assert p["intents"][0]["intent"] == "Define CLV"


def test_retrieve_documents_searches_in_parallel_and_merges_duplicates(mesh_factory, monkeypatch):
    mesh = mesh_factory()
    mesh.ingest_document(content=CLV_DOC.encode(), filename="clv.md", wait_seconds=30)
    mesh.ingest_document(content=b"# Orders\nEach order has an amount and belongs to one customer.",
                         filename="orders.md", wait_seconds=30)
    barrier, real = threading.Barrier(2, timeout=10), mesh.retrieval.retrieve

    def together(query, *a, **k):
        barrier.wait()  # both queries in flight at once
        return real(query, *a, **k)
    monkeypatch.setattr(mesh.retrieval, "retrieve", together)
    out = mesh.retrieve_documents(["customer lifetime value", "Customer Lifetime Value", "customer revenue"][1:])
    assert out["queries"] == ["Customer Lifetime Value", "customer revenue"]
    ids = [d["chunk_id"] for d in out["results"]]
    assert len(ids) == len(set(ids)) and [p["query"] for p in out["per_query"]] == out["queries"]
    scores = [d["score"] for d in out["results"]]
    assert scores == sorted(scores, reverse=True) and [d["rank"] for d in out["results"]] == list(range(1, len(ids) + 1))
    assert any(len(d["matched_queries"]) == 2 for d in out["results"])
    monkeypatch.setattr(mesh.retrieval, "retrieve", real)
    assert mesh.retrieve_documents(["a", "A", " a "])["queries"] == ["a"]
    with pytest.raises(InvalidArgument):
        mesh.retrieve_documents([f"q{i}" for i in range(6)])


def test_old_results_become_stubs_when_the_context_is_full(agent_mesh):
    script = Script(SINGLE, {"I1": [tools(call("retrieve_documents", queries=["customer lifetime value"])),
                                    tools(call("retrieve_documents", queries=["order amounts"])),
                                    tools(call("retrieve_documents", queries=["customer revenue"])),
                                    complete("Done.")]})
    mesh, llm = agent_mesh(script, agent={"max_context_chars": 3000, "max_observation_chars": 1500}, catalog=False)
    mesh.search("customer lifetime value")
    last = step_calls(llm)[-1]["prompt"]
    assert "[omitted to save space" in last
    assert re.search(r"### Step 3\n.*\nCall 1: .*\nResult 1: \{", last)  # newest result kept in full


def test_fit_observation_bounds_size():
    obs = {"text": "x" * 5000, "items": [{"t": "y" * 300} for _ in range(40)]}
    out = fit_observation(obs, 3000)
    assert len(json.dumps(out)) <= 3000 and "truncated" in out and out["text"].endswith("…")


def test_parse_json_object_repairs_only_invalid_escapes():
    assert parse_json_object(r'{"a": "$\{x\}$"}') == {"a": r"$\{x\}$"}
    assert parse_json_object(r'{"a": "\\{ kept"}') == {"a": r"\{ kept"}
    assert parse_json_object('{"a": "line\\nbreak"}') == {"a": "line\nbreak"}
    assert parse_json_object('{"answer": "a ``` fence", "x": 1}') == {"answer": "a ``` fence", "x": 1}


def test_reply_with_latex_is_accepted_on_the_first_try(agent_mesh):
    latex = ('{"evaluation": "ok", "action": "complete", "status": "answered", '
             '"answer": "Sum $\\{x_i\\}$ over orders."}')
    mesh, llm = agent_mesh(Script(SINGLE, {"I1": [latex]}), catalog=False, document=False)
    task = mesh.search("customer lifetime value")["tasks"][0]
    assert [s["action"] for s in task["steps"]] == ["complete"] and r"$\{x_i\}$" in task["answer"]


def test_search_without_agent_model_falls_back_to_direct_retrieval(mesh_factory, agent_mesh):
    plain = mesh_factory()
    plain.ingest_document(content=CLV_DOC.encode(), filename="clv.md", wait_seconds=30)
    res = plain.search("customer lifetime value")
    assert res["mode"] == "retrieval" and res["agent"]["used"] is False and "agent.enabled" in res["agent"]["reason"]
    assert res["results"]

    def broken(prompt):
        raise RuntimeError("model down")
    mesh, llm = agent_mesh(Script(broken, {}), catalog=False)
    res = mesh.search("customer lifetime value")
    assert res["mode"] == "retrieval" and "model down" in res["agent"]["reason"] and res["results"]


def test_api_search_uses_the_agent_and_retrieve_stays_direct(agent_mesh):
    mesh, llm = agent_mesh(Script(SINGLE, {"I1": [complete("Nothing needed.")]}), catalog=False)
    with TestClient(create_app(mesh)) as c:
        body = c.post("/api/search", json={"query": "customer lifetime value"}).json()
        assert body["mode"] == "agent" and body["tasks"][0]["status"] == "answered" and body["decisions"]
        direct = c.post("/api/retrieve", json={"query": "customer lifetime value"}).json()
        assert "mode" not in direct and direct["results"]
        assert c.post("/api/search", json={"query": "x", "top_k": 99}).status_code == 400
        agent = c.get("/api/health").json()["checks"]["agent"]
        assert agent["enabled"] is True and agent["max_iterations"] == 20

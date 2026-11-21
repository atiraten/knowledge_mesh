"""Human clarification at the response boundary (AT-23), harness compatibility (AT-04) and combined
document + knowledge graph retrieval in one executor turn (AT-15)."""

import re

import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.api.app import create_app
from knowledge_mesh.common.errors import Conflict, NotFound
from knowledge_mesh.llm.client import FakeLLM
from tests.conftest import REPO
from tests.kg_support import shop_mesh, shop_source  # noqa: F401
from tests.test_agent import PROMPTS, Script, call, complete, last_result, tools

PLAN = {"original_query": "loyalty tier of our customers", "decompose": False, "reasoning": "One objective.",
        "execution": "single", "intents": [{"id": "I1", "intent": "find the loyalty tier of customers",
                                            "retrieval_queries": ["customer loyalty tier"], "depends_on": []}]}
QUESTION = "Which dataset or document defines the loyalty tier?"
REPLY = "The tier is gold for customers born before 1990; use the customers table."


def executor(prompt):
    if "## Clarifications from the user" in prompt:
        return [tools(call("query_knowledge_graph")),
                lambda p: complete(f"Customers and their birth dates are in customers "
                                   f"[{last_result(p)['entities'][0]['id']}].")]
    return [tools(call("retrieve_context", queries=["customer loyalty tier"])),
            {"evaluation": "Nothing about tiers.", "action": "complete", "status": "not_found",
             "answer": "No source defines a loyalty tier.",
             "clarification_needed": "which dataset or document defines the loyalty tier"}]


def synthesis(prompt):
    if "<clarification_notes>" in prompt:
        return f"I found no definition of a loyalty tier.\nClarification question: {QUESTION}"
    ids = re.findall(r"^\[([A-Z]\d+)\]", prompt, re.M)
    return f"Gold-tier customers are listed in customers [{ids[0]}]." if ids else "Nothing."


@pytest.fixture
def clarify_mesh(shop_mesh):
    def make(agent=None, script=None):
        settings = {"enabled": True, "provider": "fake", "model": "fake", **PROMPTS, **(agent or {})}
        llm = FakeLLM(script or Script(PLAN, {"I1": executor}, replan=PLAN, synthesis=synthesis))
        mesh = shop_mesh({"agent": settings}, agent_llm=llm)
        llm.calls.clear()
        return mesh, llm
    return make


def test_unsure_executor_asks_the_user_and_continues_with_the_reply(clarify_mesh):
    """AT-23: partial / not_found after the replans -> one question; the reply continues the same request."""
    mesh, llm = clarify_mesh()
    first = mesh.search("loyalty tier of our customers")
    assert first["outcome"] == "needs_clarification"
    assert [d["decision"] for d in first["decisions"]][:1] == ["replan"]  # the existing replan ran first
    assert first["clarification"]["question"] == QUESTION and first["clarification"]["round"] == 1
    assert "Clarification question" not in first["answer"] and "no definition" in first["answer"]
    assert first["tasks"][-1]["clarification_needed"] == "which dataset or document defines the loyalty tier"
    synth = [c for c in llm.calls if c["purpose"] == "agent_synthesis"][-1]
    assert "<clarification_notes>" in synth["prompt"] and "loyalty tier" in synth["prompt"]
    row = mesh.db.query_one("SELECT * FROM pending_clarifications WHERE trace_id = ?", (first["trace_id"],))
    assert row["status"] == "pending" and row["question"] == QUESTION and "customer loyalty tier" in row["state_json"]
    assert mesh.get_search(first["trace_id"])["outcome"] == "needs_clarification"

    llm.calls.clear()
    second = mesh.continue_search(first["trace_id"], REPLY)
    replan = [c for c in llm.calls if c["purpose"] == "agent_replanner"]
    assert replan and REPLY in replan[0]["prompt"] and QUESTION in replan[0]["prompt"]
    steps = [c for c in llm.calls if c["purpose"] == "agent_step"]
    assert steps and all(REPLY in c["prompt"] and "## Clarifications from the user" in c["prompt"] for c in steps)
    synth = [c for c in llm.calls if c["purpose"] == "agent_synthesis"][-1]
    assert "<user_clarifications>" in synth["prompt"] and REPLY in synth["prompt"]
    assert second["outcome"] == "answered" and "clarification" not in second
    assert second["continues"] == first["trace_id"] and second["clarifications"][0]["answer"] == REPLY
    assert second["query"] == first["query"]  # the original request is kept
    assert second["plan"]["replan_reason"].startswith("user clarification")
    row = mesh.db.query_one("SELECT * FROM pending_clarifications WHERE trace_id = ?", (first["trace_id"],))
    assert row["status"] == "answered" and row["state_json"] == "{}" and row["continuation_trace_id"] == second["trace_id"]
    with pytest.raises(Conflict):
        mesh.continue_search(first["trace_id"], "again")


def test_no_information_found_asks_for_details(clarify_mesh):
    script = Script(PLAN, {"I1": [{"evaluation": "No sources.", "action": "complete", "status": "not_found",
                                   "answer": "No information found."}]},
                    replan=PLAN, synthesis=lambda p: "No information found.")
    mesh, _ = clarify_mesh(script=script)
    res = mesh.search("loyalty tier of our customers")
    assert res["outcome"] == "needs_clarification"
    assert "find the loyalty tier of customers" in res["clarification"]["question"]  # fallback question


def test_clarification_limits_expiry_and_switch(clarify_mesh):
    mesh, _ = clarify_mesh({"max_clarifications": 1})
    first = mesh.search("loyalty tier of our customers")
    second = mesh.continue_search(first["trace_id"], "no idea")  # still not found: no second question
    assert "clarification" not in second and second["outcome"] in ("answered", "failed")
    mesh, _ = clarify_mesh()
    first = mesh.search("loyalty tier of our customers")
    mesh.db.execute("UPDATE pending_clarifications SET expires_at = '2000-01-01T00:00:00+00:00'")
    with pytest.raises(NotFound):
        mesh.continue_search(first["trace_id"], REPLY)
    mesh, _ = clarify_mesh({"clarification_enabled": False})
    res = mesh.search("loyalty tier of our customers")
    assert "clarification" not in res and res["outcome"] == "answered"


def test_clarification_reply_endpoint(clarify_mesh):
    mesh, _ = clarify_mesh()
    client = TestClient(create_app(mesh))
    first = client.post("/api/search", json={"query": "loyalty tier of our customers"}).json()
    assert first["clarification"]["reply_ref"] == f"/api/searches/{first['trace_id']}/clarification"
    second = client.post(first["clarification"]["reply_ref"], json={"reply": REPLY})
    assert second.status_code == 200 and second.json()["continues"] == first["trace_id"]
    again = client.post(first["clarification"]["reply_ref"], json={"reply": REPLY})
    assert again.status_code == 409
    assert client.post("/api/searches/unknown/clarification", json={"reply": "x"}).status_code == 404


def test_executor_combines_documents_and_knowledge_graph_in_one_turn(clarify_mesh):
    """AT-15: retrieve_context and query_knowledge_graph in the same executor turn, inside the existing loop."""
    plan = {"original_query": "how do customers buy products", "decompose": False, "reasoning": "One objective.",
            "execution": "single", "intents": [{"id": "I1", "intent": "how customers buy products",
                                                "retrieval_queries": ["customer bought product"], "depends_on": []}]}

    def finish(p):
        kg, docs = last_result(p, 2), last_result(p, 1)
        return complete(f"Customers buy products through orders [{kg['paths'][0]['id']}] [{docs['documents'][0]['id']}].")

    script = Script(plan, {"I1": [tools(call("retrieve_context", queries=["customer bought product"]),
                                        call("query_knowledge_graph")), finish]})
    mesh, _ = clarify_mesh(script=script)
    mesh.ingest_document(content=b"# Ordering\nA customer places orders; each order line is for one product.\n",
                         filename="ordering.md", wait_seconds=30)
    res = mesh.search("how do customers buy products")
    calls = res["tasks"][0]["steps"][0]["tool_calls"]
    assert [c["tool"] for c in calls] == ["retrieve_context", "query_knowledge_graph"]
    assert all(c["status"] == "ok" for c in calls)
    ids = res["tasks"][0]["evidence_ids"]
    assert any(i.startswith("D") for i in ids) and any(i.startswith("K") for i in ids)
    assert res["outcome"] == "answered" and res["tasks"][0]["status"] == "answered"
    path = next(e for e in res["evidence"] if e["type"] == "graph_path")
    assert "Customer -> [placesOrder] Order" in path["label"]

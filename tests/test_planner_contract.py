"""Planner output contract (AT-02, AT-03): exact field structure, intents, dependencies and execution modes."""

import json

from knowledge_mesh.agent.plan import PLAN_SCHEMA, IntentPlan, validate_plan
from knowledge_mesh.llm.client import FakeLLM
from tests.conftest import REPO
from tests.test_agent import PROMPTS, Script, call, complete, tools

SPEC_PLAN = {
    "original_query": "list customer born before 1990",
    "decompose": False,
    "intents": [{"id": "I1", "intent": "list customer born before 1990",
                 "retrieval_queries": ["customer born before 1990", "customers birth date before 1990"],
                 "depends_on": []}],
    "reasoning": "The query has a single clear objective to find customers born before 1990 and does not require "
                 "decomposition.",
    "execution": "single",
}
MULTI_PLAN = {
    "original_query": "What is Customer Lifetime Value and which customers have the highest value?",
    "decompose": True,
    "intents": [
        {"id": "I1", "intent": "definition of Customer Lifetime Value",
         "retrieval_queries": ["Customer Lifetime Value definition"], "depends_on": []},
        {"id": "I2", "intent": "customers with the highest Customer Lifetime Value",
         "retrieval_queries": ["highest Customer Lifetime Value customers"], "depends_on": ["I1"]}],
    "reasoning": "Two outcomes: a definition, then the customers ranked by it.",
    "execution": "sequential",
}


def test_schema_keeps_the_required_field_names_and_execution_enum():
    assert set(PLAN_SCHEMA["properties"]) == {"original_query", "decompose", "intents", "reasoning", "execution"}
    assert set(PLAN_SCHEMA["properties"]["intents"]["items"]["properties"]) == {"id", "intent", "retrieval_queries",
                                                                               "depends_on"}
    assert PLAN_SCHEMA["properties"]["execution"]["enum"] == ["single", "parallel", "sequential", "mixed"]
    planner = (REPO / "prompts" / "planner.txt").read_text()
    assert '"born before 1990" stays' in planner and "Do not write SQL" in planner


def _search(mesh_factory, config_factory, plan):
    llm = FakeLLM(Script(plan, {i["id"]: [tools(call("retrieve_documents", queries=i["retrieval_queries"])),
                                          complete("done")] for i in plan["intents"]}))
    mesh = mesh_factory(config_factory({"agent": {"enabled": True, "provider": "fake", "model": "fake", **PROMPTS,
                                                  "clarification_enabled": False}}), agent_llm=llm)
    return mesh.search(plan["original_query"]), llm


def test_single_intent_plan_structure(mesh_factory, config_factory):
    """AT-02."""
    res, llm = _search(mesh_factory, config_factory, SPEC_PLAN)
    plan = res["plan"]
    assert {k: plan[k] for k in SPEC_PLAN} == SPEC_PLAN
    assert plan["decompose"] is False and plan["execution"] == "single" and len(plan["intents"]) == 1
    assert plan["intents"][0]["retrieval_queries"] and plan["intents"][0]["depends_on"] == []
    assert plan["fallback_reason"] is None
    planner = next(c for c in llm.calls if c["purpose"] == "agent_planner")
    assert "list customer born before 1990" in planner["prompt"] and planner["json_schema"] == PLAN_SCHEMA


def test_multi_intent_plan_with_valid_dependencies(mesh_factory, config_factory):
    """AT-03."""
    plan = IntentPlan.model_validate(MULTI_PLAN)
    assert validate_plan(plan, 5) == [["I1"], ["I2"]]
    res, _ = _search(mesh_factory, config_factory, MULTI_PLAN)
    assert res["plan"]["decompose"] is True and [i["id"] for i in res["plan"]["intents"]] == ["I1", "I2"]
    assert res["plan"]["execution"] == "sequential" and res["plan"]["fallback_reason"] is None
    assert [t["id"] for t in res["tasks"]] == ["I1", "I2"]  # the dependency ran first
    # an unsupported execution mode or an unknown dependency follows the existing fallback policy
    bad = json.loads(json.dumps(MULTI_PLAN))
    bad["execution"] = "pipeline"
    res, _ = _search(mesh_factory, config_factory, bad)
    assert res["plan"]["fallback_reason"] and len(res["plan"]["intents"]) == 1
    bad = json.loads(json.dumps(MULTI_PLAN))
    bad["intents"][1]["depends_on"] = ["I9"]
    res, _ = _search(mesh_factory, config_factory, bad)
    assert "unknown intent I9" in res["plan"]["fallback_reason"]

import re

import pytest
from fastapi.testclient import TestClient

from knowledge_mesh.api.app import create_app
from knowledge_mesh.llm.client import FakeLLM

LEVELS = ["Incorrect", "Ambiguous", "Acceptable", "Clear", "Excellent"]
DOCS = {
    "ids.md": "# Identification\nAssign each customer a unique identifier when the account is opened.",
    "ship.md": "# Shipping\nParcels for customers leave the warehouse within two business days.",
    "bill.md": "# Billing\nCustomer invoices are issued monthly and paid within thirty days.",
}


class FakeLaya:
    """Mimics laya.Agent.predict / predict_batch output shapes (probabilities sum to 1)."""

    def __init__(self, favour: dict[str, float], quality: dict[str, str], fail: bool = False):
        self.favour, self.quality, self.fail = favour, quality, fail
        self.calls: list[tuple] = []

    def predict(self, state, questions, **kw):
        self.calls.append(("predict", state, questions, kw))
        if self.fail:
            raise RuntimeError("model crashed")
        crit = questions["description"]["criteria"]
        raw = {k: next((w for word, w in self.favour.items() if word in text.lower()), 0.01) for k, text in crit.items()}
        total = sum(raw.values())
        probs = {k: v / total for k, v in raw.items()}
        return {"answers": {"description": {"type": "choice", "choice": max(probs, key=probs.get),
                                            "probabilities": probs, "confidence": 0.5}},
                "usage": {"input_tokens": 100, "state_tokens": 10, "truncated": False, "truncated_questions": []}}

    def predict_batch(self, states, questions, **kw):
        self.calls.append(("predict_batch", states, questions, kw))
        out = []
        for st in states:
            label = next((q for word, q in self.quality.items() if word in st["chosen_response"].lower()), "Incorrect")
            level = LEVELS.index(label)
            probs = {str(i): (0.7 if i == level else 0.3 / 4) for i in range(5)}
            out.append({"answers": {"quality": {"type": "score", "score": float(level),
                                                "legend": {str(i): l for i, l in enumerate(LEVELS)},
                                                "probabilities": probs, "confidence": 0.6}}})
        return out


@pytest.fixture
def rr_mesh(mesh_factory, config_factory):
    def make(agent, reranker=None, enabled=True, answer_llm=None):
        overrides = {"retrieval": {"reranker_enabled": enabled, "min_document_similarity": 0.05},
                     "reranker": reranker or {}}
        if answer_llm:
            overrides["answer_generation"] = {"enabled": True, "provider": "fake", "model": "fake"}
        mesh = mesh_factory(config_factory(overrides), reranker_agent=agent, answer_llm=answer_llm)
        for name, text in DOCS.items():
            mesh.ingest_document(content=text.encode(), filename=name, wait_seconds=30)
        return mesh
    return make


def test_selects_passage_over_threshold_rated_acceptable(rr_mesh):
    agent = FakeLaya(favour={"identifier": 30.0}, quality={"identifier": "Clear"})
    mesh = rr_mesh(agent)
    res = mesh.retrieve_context("how is a customer identified")
    rr = res["reranking"]
    assert rr["applied"] and rr["outcome"] == "selected" and rr["calls"] == 2 and rr["candidates"] == 3
    assert len(res["results"]) == 1 and "unique identifier" in res["results"][0]["text"]
    assert res["results"][0]["rerank"]["quality"] == "Clear" and res["results"][0]["rerank"]["choice_probability"] > 0.8
    assert res["response"]["found"] and res["response"]["chunk_id"] == res["results"][0]["chunk_id"]
    kind, state, questions, kw = agent.calls[0]
    assert state == {"question": "how is a customer identified"}
    assert questions["description"]["type"] == "choice"
    assert questions["description"]["instructions"] == "Which response best matches answer the question?"
    assert set(questions["description"]["criteria"]) == {"d0", "d1", "d2"} and kw["head_max_len"] == 448
    kind2, states, q2, _ = agent.calls[1]
    assert kind2 == "predict_batch" and len(states) == 1 and q2["quality"]["criteria"] == LEVELS
    assert states[0]["question"] == "how is a customer identified" and "identifier" in states[0]["chosen_response"]


def test_no_passage_over_threshold_means_no_response_and_one_call(rr_mesh):
    agent = FakeLaya(favour={"identifier": 1.0, "parcels": 1.0, "invoices": 1.0}, quality={})
    mesh = rr_mesh(agent)
    res = mesh.retrieve_context("how is a customer identified")
    assert res["reranking"]["outcome"] == "no_response" and res["reranking"]["calls"] == 1
    assert res["results"] == [] and res["response"] == {"found": False, "text": "I could not find a response."}
    assert res["message"] == "I could not find a response."
    assert all(not d["passed_probability"] for d in res["reranking"]["decisions"])


def test_quality_below_acceptable_means_no_response(rr_mesh):
    agent = FakeLaya(favour={"identifier": 30.0}, quality={"identifier": "Ambiguous"})
    res = rr_mesh(agent).retrieve_context("how is a customer identified")
    rr = res["reranking"]
    assert rr["outcome"] == "no_response" and rr["calls"] == 2 and res["results"] == []
    top = rr["decisions"][0]
    assert top["passed_probability"] and top["quality"] == "Ambiguous" and top["acceptable"] is False


def test_top_two_scored_in_one_call_and_best_quality_wins(rr_mesh):
    agent = FakeLaya(favour={"identifier": 5.0, "parcels": 4.0, "invoices": 0.05},
                     quality={"identifier": "Acceptable", "parcels": "Excellent"})
    res = rr_mesh(agent, reranker={"min_probability": 0.3}).retrieve_context("customer")
    rr = res["reranking"]
    passed = [d for d in rr["decisions"] if d["passed_probability"]]
    assert len(passed) == 2 and rr["calls"] == 2
    assert len(agent.calls[1][1]) == 2  # both shortlisted passages in the single quality call
    assert "Parcels" in res["results"][0]["text"]  # higher quality beats higher choice probability
    assert res["results"][0]["rerank"]["quality"] == "Excellent"


def test_candidate_text_is_bounded(rr_mesh):
    agent = FakeLaya(favour={"identifier": 30.0}, quality={"identifier": "Clear"})
    rr_mesh(agent, reranker={"max_candidate_chars": 60}).retrieve_context("customer identifier")
    assert all(len(t) <= 60 for t in agent.calls[0][2]["description"]["criteria"].values())


def test_reranker_failure_keeps_similarity_order(rr_mesh):
    res = rr_mesh(FakeLaya({}, {}, fail=True)).retrieve_context("customer identifier")
    assert res["reranking"]["applied"] is False and res["reranking"]["outcome"] == "failed"
    assert "model crashed" in res["reranking"]["error"] and len(res["results"]) == 3
    assert "response" not in res


def test_per_request_override(rr_mesh):
    agent = FakeLaya(favour={"identifier": 30.0}, quality={"identifier": "Clear"})
    mesh = rr_mesh(agent, enabled=False)
    assert "reranking" not in mesh.retrieve_context("customer identifier")
    assert mesh.retrieve_context("customer identifier", rerank=True)["reranking"]["applied"]
    mesh2 = rr_mesh(FakeLaya({}, {}), enabled=True)
    assert "reranking" not in mesh2.retrieve_context("customer identifier", rerank=False)


def test_answer_mode_uses_selected_passage_or_says_not_found(rr_mesh):
    def answerer(prompt, system, purpose):
        ids = re.findall(r"^\[(D\d+)\]", prompt, re.M)
        return f"Customers get a unique identifier [{ids[0]}]." if ids else "nothing"

    llm = FakeLLM(answerer)
    mesh = rr_mesh(FakeLaya({"identifier": 30.0}, {"identifier": "Excellent"}), answer_llm=llm)
    out = mesh.answer("how is a customer identified", evidence_types=["documents"])
    assert out["grounded"] and len(re.findall(r"^\[D\d+\]", llm.calls[-1]["prompt"], re.M)) == 1
    calls = len(llm.calls)
    mesh.reranker._agent = FakeLaya({"identifier": 1.0, "parcels": 1.0, "invoices": 1.0}, {})
    none = mesh.answer("how is a customer identified", evidence_types=["documents"])
    assert none["answer"] == "I could not find a response." and len(llm.calls) == calls


GUIDE = "# Guide\n" + " ".join(f"Rule {i}: customers place orders and receive invoices." for i in range(300))


def _leaf_texts(mesh) -> set[str]:
    docs = mesh.doc_index._ensure().storage_context.docstore.docs.values()
    return {n.get_content() for n in docs if n.metadata.get("is_leaf")}


def test_laya_ranks_leaf_chunks_and_the_selected_chunk_gets_its_parents(rr_mesh):
    agent = FakeLaya({"rule": 30.0}, {"rule": "Clear"})
    mesh = rr_mesh(agent, reranker={"min_probability": 0.1})  # several guide chunks tie
    mesh.ingest_document(content=GUIDE.encode(), filename="guide.md", wait_seconds=30)
    res = mesh.retrieve_context("customers place orders", top_k=5, evidence_types=["documents"])
    candidates = agent.calls[0][2]["description"]["criteria"].values()
    leaves = _leaf_texts(mesh)
    assert all(text in leaves for text in candidates)  # whole leaf chunks, never parent text
    assert all(text in leaves for text in (s["chosen_response"] for s in agent.calls[1][1]))
    chosen = res["results"][0]
    assert res["reranking"]["outcome"] == "selected" and chosen["text"] in leaves
    assert [p["level"] for p in chosen["parents"]] == [1, 0]
    assert all(len(p["text"]) > len(chosen["text"]) for p in chosen["parents"])


def test_answer_prompt_has_matched_chunk_and_every_parent_once(rr_mesh):
    llm = FakeLLM(lambda prompt, system, purpose: "Rules apply [D1].")
    mesh = rr_mesh(FakeLaya({}, {}), enabled=False, answer_llm=llm)
    mesh.ingest_document(content=GUIDE.encode(), filename="guide.md", wait_seconds=30)
    mesh.retrieval.cfg.max_context_tokens = 20000  # room for every parent of the top 5
    out = mesh.answer("customers place orders and receive invoices", top_k=5, evidence_types=["documents"])
    prompt = llm.calls[-1]["prompt"]
    first = out["evidence"]["results"][0]
    assert out["grounded"] and f"Matched chunk:\n{first['text']}" in prompt
    for p in first["parents"]:
        assert f"Parent context (level {p['level']}, {p['node_id']}):\n{p['text']}" in prompt
    owners: dict[str, list[int]] = {}
    for i, d in enumerate(out["evidence"]["results"], start=1):
        for p in d["parents"]:
            owners.setdefault(p["node_id"], []).append(i)
    shared = {nid: ids for nid, ids in owners.items() if len(ids) > 1}
    assert shared  # guide chunks share parents
    for nid, ids in shared.items():  # text printed once, later blocks point back to it
        assert prompt.count(f"{nid}):\n") == 1
        assert prompt.count(f"{nid}): same as in [D{ids[0]}]") == len(ids) - 1


def test_search_api_returns_reranking(rr_mesh):
    mesh = rr_mesh(FakeLaya({"identifier": 30.0}, {"identifier": "Clear"}))
    with TestClient(create_app(mesh)) as c:
        body = c.post("/api/search", json={"query": "customer identifier", "rerank": True}).json()
        assert body["reranking"]["outcome"] == "selected" and body["response"]["found"]
        assert c.get("/api/health").json()["checks"]["reranker"]["enabled"] is True

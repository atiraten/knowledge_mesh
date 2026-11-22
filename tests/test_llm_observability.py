"""Observability of model calls: why each answer ended (finish reason), thinking tokens, answers cut off at
max_output_tokens (span attributes, metric, log line without content, term run issue), one class word per
name in the term prompt, and the Phoenix trace link on the Jobs page."""

import logging
import re
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import types as gtypes

from knowledge_mesh.api.app import STATIC_DIR, create_app
from knowledge_mesh.llm.client import FakeLLM, InstrumentedLLM, LLMResponse, _GoogleClient, _OpenAIClient
from knowledge_mesh.structured.terms import render_prompt
from tests.conftest import fake_term_responder, prompt_table
from tests.test_observability_agent import PhoenixRecorder, by_name, spans  # noqa: F401


def cut_off(text="Look at that list: `Code` is listed twice"):
    return LLMResponse(text, "fake", 2238, 325, finish_reason="MAX_TOKENS", thinking_tokens=7867, truncated=True)


def gemini(response):
    client = _GoogleClient.__new__(_GoogleClient)  # no SDK client: generate_content is faked
    client._types = SimpleNamespace(GenerateContentConfig=lambda **kw: kw)
    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: response))
    client.model, client.settings = "gemini-test", SimpleNamespace(temperature=0.0, max_output_tokens=8192)
    return client


def openai(response):
    client = _OpenAIClient.__new__(_OpenAIClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    client.model, client.settings = "gpt-test", SimpleNamespace(temperature=0.0)
    return client


def test_gemini_reports_finish_reason_and_thinking_tokens():
    usage = SimpleNamespace(prompt_token_count=2238, candidates_token_count=325, thoughts_token_count=7867)
    r = gemini(SimpleNamespace(text="Look at that", usage_metadata=usage,
                               candidates=[SimpleNamespace(finish_reason=gtypes.FinishReason.MAX_TOKENS)])).generate("p")
    assert (r.finish_reason, r.output_tokens, r.thinking_tokens, r.truncated) == ("MAX_TOKENS", 325, 7867, True)
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=None)
    r = gemini(SimpleNamespace(text="a,b", usage_metadata=usage,
                               candidates=[SimpleNamespace(finish_reason=gtypes.FinishReason.STOP)])).generate("p")
    assert (r.finish_reason, r.thinking_tokens, r.truncated) == ("STOP", None, False)
    r = gemini(SimpleNamespace(text=None, usage_metadata=None, candidates=None)).generate("p")
    assert (r.text, r.finish_reason, r.truncated) == ("", None, False)


def test_openai_reports_finish_reason_and_reasoning_tokens():
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=100))
    r = openai(SimpleNamespace(usage=usage, choices=[SimpleNamespace(finish_reason="length",
                                                                     message=SimpleNamespace(content="part"))])).generate("p")
    assert (r.finish_reason, r.output_tokens, r.thinking_tokens, r.truncated) == ("length", 5, 100, True)
    r = openai(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                               choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))])).generate("p")
    assert (r.finish_reason, r.thinking_tokens, r.truncated) == ("stop", None, False)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_cut_off_answer_is_on_the_span_in_metrics_and_in_logs(spans, mesh_factory, config_factory):
    cfg = config_factory({"term_generation": {"input_cost_per_1k_tokens": 0.0, "output_cost_per_1k_tokens": 1.0}})
    mesh = mesh_factory(cfg)
    capture = Capture()
    logger = logging.getLogger("knowledge_mesh.llm.client")
    logger.addHandler(capture)
    try:
        r = InstrumentedLLM(FakeLLM(lambda p, s, purpose: cut_off()), cfg.term_generation).generate(
            "the prompt", purpose="term_generation")
    finally:
        logger.removeHandler(capture)
    assert r.cost_usd == pytest.approx((325 + 7867) / 1000)  # thinking tokens are billed as output
    attrs = by_name(spans, "llm.term_generation")[-1].attributes
    assert attrs["llm.finish_reason"] == "MAX_TOKENS" and attrs["llm.output_truncated"] is True
    assert attrs["llm.token_count.completion"] == 325
    assert attrs["llm.token_count.completion_details.reasoning"] == 7867
    assert attrs["llm.token_count.total"] == 2238 + 325 + 7867
    warning = [x for x in capture.records if x.getMessage() == "LLM output cut off at max_output_tokens"]
    assert warning and warning[0].thinking_tokens == 7867 and warning[0].max_output_tokens == 8192
    assert not any("Look at that" in str(vars(x)) for x in capture.records)  # never the content in logs
    text = TestClient(create_app(mesh)).get("/metrics").text
    assert re.search(r'mesh_llm_truncated_responses_total\{[^}]*purpose="term_generation"', text)
    assert re.search(r'mesh_llm_tokens_total\{[^}]*direction="thinking"', text)

    InstrumentedLLM(FakeLLM(lambda p, s, purpose: "fine"), cfg.term_generation).generate("p", purpose="term_review")
    attrs = by_name(spans, "llm.term_review")[-1].attributes
    assert attrs["llm.output_truncated"] is False and "llm.finish_reason" not in attrs


def test_term_run_names_a_cut_off_answer(mesh_factory, config_factory, sqlite_source):
    def responder(prompt, system, purpose):
        if purpose == "term_generation" and prompt_table(prompt) == "orders":
            return cut_off()
        return fake_term_responder(prompt, system, purpose)

    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source]}), term_llm=FakeLLM(responder))
    mesh.catalog_source("crm", wait_seconds=120)
    ds = mesh.db.query_one("SELECT dataset_id FROM datasets WHERE name = 'orders'")["dataset_id"]
    run = mesh.db.query_one("SELECT * FROM term_generation_runs WHERE dataset_id = ?", (ds,))
    assert run["generated"] == 0
    assert "term_generation response cut off at max_output_tokens (325 visible + 7867 thinking tokens)" in run["error"]


def test_term_prompt_lists_each_class_word_once():
    settings = SimpleNamespace(industry_name="Banking", language_name="English",
                               class_words=["Code", "Name", "Code", "Percent", "Percent"])
    out = render_prompt("{class_words_section}", "customer", ["id"], settings, {})
    assert out == "   Allowed class words: Code, Name, Percent"


def test_job_links_to_its_trace_in_phoenix(spans, mesh_factory, config_factory, sqlite_source, fake_term_llm):
    mesh = mesh_factory(config_factory({"structured_sources": [sqlite_source]}), term_llm=fake_term_llm,
                        phoenix_transport=httpx.MockTransport(PhoenixRecorder()))
    client = TestClient(create_app(mesh))
    job_id = client.post("/api/structured-sources/crm/catalog", json={}).json()["job_id"]  # traced like the UI
    for _ in range(240):
        out = client.get(f"/api/jobs/{job_id}").json()
        if out["status"] not in ("queued", "running"):
            break
        time.sleep(0.5)
    base = mesh.config.observability.phoenix_url.rstrip("/")
    assert out["trace_id"] and out["observability"]["trace_url"] == f"{base}/projects/UHJvamVjdDoy/traces/{out['trace_id']}"
    js = (STATIC_DIR / "app.js").read_text()
    assert "job.observability && job.observability.trace_url" in js and js.count("open trace in Phoenix") == 2


def test_provider_failures_carry_status_and_message():
    from knowledge_mesh.common.errors import TransientError

    class ServerError(Exception):
        pass

    def deadline(**kwargs):
        raise ServerError("504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before "
                          "operation could complete.', 'status': 'DEADLINE_EXCEEDED'}}")

    client = gemini(None)
    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=deadline))
    with pytest.raises(TransientError) as err:
        client.generate("p")
    assert str(err.value.message if hasattr(err.value, "message") else err.value).startswith(
        "google call failed: ServerError: 504 DEADLINE_EXCEEDED")
    from knowledge_mesh.llm.client import error_type
    assert error_type(err.value) == "timeout"  # classified for metrics and alerts

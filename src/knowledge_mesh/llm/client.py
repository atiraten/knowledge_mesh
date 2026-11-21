"""LLM clients for term generation, answer generation and the search agent (configured independently).

Providers: ``google`` (Gemini via ``google-genai``), ``openai`` (or any OpenAI-compatible
endpoint through ``base_url``), and ``fake`` (tests). API keys are read from the environment
variable ``api_key_env`` first, then from ``api_key_file``; they are never logged.

Every call is traced as an OpenInference LLM span (model, token counts, latency, failures;
prompt/response text only when ``observability.capture_sensitive_content`` is true) and
counted in Prometheus metrics, including an estimated cost.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from ..common.errors import NotConfigured, OperationFailed, TransientError
from ..common.retry import RetryPolicy, retry_call
from ..observability import get_metrics, span
from ..observability.telemetry import capture_sensitive

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float = 0.0


class LLMClient(Protocol):
    model: str
    provider: str

    def generate(self, prompt: str, *, system: str | None = None, purpose: str = "generic",
                 json_schema: dict | None = None, json_output: bool = False) -> LLMResponse: ...


def resolve_api_key(settings) -> str:
    if settings.api_key_env and os.environ.get(settings.api_key_env):
        return os.environ[settings.api_key_env].strip()
    if settings.api_key_file and os.path.exists(settings.api_key_file):
        key = open(settings.api_key_file, encoding="utf-8").read().strip()
        if key:
            return key
    raise NotConfigured(
        f"no API key for {settings.provider}: set ${settings.api_key_env or '<api_key_env>'} or api_key_file"
    )


class _GoogleClient:
    provider = "google"

    def __init__(self, settings):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise NotConfigured("google-genai is not installed (pip install 'knowledge-mesh[google]')") from exc
        self._types = types
        self._client = genai.Client(api_key=resolve_api_key(settings),
                                    http_options=types.HttpOptions(timeout=int(settings.timeout_seconds * 1000)))
        self.model = settings.model
        self.settings = settings

    def generate(self, prompt: str, *, system: str | None = None, purpose: str = "generic",
                 json_schema: dict | None = None, json_output: bool = False) -> LLMResponse:
        extra: dict = {}
        if json_output or json_schema:
            extra["response_mime_type"] = "application/json"
        if json_schema:
            extra["response_json_schema"] = json_schema
        cfg = self._types.GenerateContentConfig(
            temperature=self.settings.temperature, max_output_tokens=self.settings.max_output_tokens,
            system_instruction=system, **extra,
        )
        try:
            resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        except Exception as exc:
            if _transient(exc):
                raise TransientError(f"{self.provider} call failed: {type(exc).__name__}") from exc
            raise OperationFailed(f"{self.provider} call failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(resp.text or "", self.model,
                           getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None))


class _OpenAIClient:
    provider = "openai"

    def __init__(self, settings):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise NotConfigured("openai is not installed (pip install 'knowledge-mesh[openai]')") from exc
        self._client = OpenAI(api_key=resolve_api_key(settings), base_url=settings.base_url,
                              timeout=settings.timeout_seconds, max_retries=0)
        self.model = settings.model
        self.settings = settings

    def generate(self, prompt: str, *, system: str | None = None, purpose: str = "generic",
                 json_schema: dict | None = None, json_output: bool = False) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        extra: dict = {}
        if json_output or json_schema:  # the schema itself is in the prompt text
            extra["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(model=self.model, messages=messages,
                                                        temperature=self.settings.temperature, **extra)
        except Exception as exc:
            if _transient(exc):
                raise TransientError(f"{self.provider} call failed: {type(exc).__name__}") from exc
            raise OperationFailed(f"{self.provider} call failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        usage = getattr(resp, "usage", None)
        return LLMResponse(resp.choices[0].message.content or "", self.model,
                           getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))


class FakeLLM:
    """Deterministic client for tests: ``responder(prompt, system, purpose) -> text``."""

    provider = "fake"

    def __init__(self, responder: Callable[[str, str | None, str], str], model: str = "fake-model"):
        self.responder = responder
        self.model = model
        self.calls: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None, purpose: str = "generic",
                 json_schema: dict | None = None, json_output: bool = False,
                 prompt_version: str | None = None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system, "purpose": purpose, "json_schema": json_schema,
                           "json_output": json_output, "prompt_version": prompt_version})
        text = self.responder(prompt, system, purpose)
        return LLMResponse(text, self.model, len(prompt) // 4, len(text) // 4)


def _transient(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(s in name for s in ("timeout", "connection", "ratelimit", "unavailable", "servererror")) or \
        any(s in text for s in ("429", "503", "resource_exhausted", "deadline", "temporarily"))


class InstrumentedLLM:
    """Retries transient failures and records traces/metrics for every call."""

    def __init__(self, client: LLMClient, settings, prompt_version: str | None = None):
        self.client = client
        self.settings = settings
        self.model = client.model
        self.provider = client.provider
        self.prompt_version = prompt_version
        self.policy = RetryPolicy(max_attempts=settings.max_retries + 1)

    def generate(self, prompt: str, *, system: str | None = None, purpose: str = "generic",
                 json_schema: dict | None = None, json_output: bool = False,
                 prompt_version: str | None = None) -> LLMResponse:
        metrics = get_metrics()
        params = {"temperature": self.settings.temperature, "max_output_tokens": self.settings.max_output_tokens,
                  "json_output": bool(json_output or json_schema), "response_schema": bool(json_schema),
                  "max_retries": self.settings.max_retries, "timeout_seconds": self.settings.timeout_seconds}
        attrs = {"llm.provider": self.provider, "llm.system": self.provider, "llm.model_name": self.model,
                 "llm.purpose": purpose, "llm.prompt_template.version": prompt_version or self.prompt_version,
                 "llm.prompt.chars": len(prompt) + len(system or ""), "llm.json_output": params["json_output"],
                 "llm.invocation_parameters": json.dumps(params)}
        if capture_sensitive():  # OpenInference messages: Phoenix shows the call as a conversation
            attrs["input.value"] = prompt[:20000]
            messages = ([("system", system)] if system else []) + [("user", prompt)]
            for i, (role, content) in enumerate(messages):
                attrs[f"llm.input_messages.{i}.message.role"] = role
                attrs[f"llm.input_messages.{i}.message.content"] = content[:20000]
        with span(f"llm.{purpose}", "LLM", attrs) as sp:
            start = time.perf_counter()
            status = "ok"
            attempts = 0

            def on_retry(attempt, exc):
                nonlocal attempts
                attempts += 1
                metrics.retries.add(1, {"operation": f"llm.{purpose}"})

            try:
                resp = retry_call(
                    lambda: self.client.generate(prompt, system=system, purpose=purpose, json_schema=json_schema,
                                                 json_output=json_output), self.policy, on_retry=on_retry,
                )
            except Exception as exc:
                status = "error"
                kind = error_type(exc)
                metrics.llm_requests.add(1, {"purpose": purpose, "model": self.model, "status": status})
                metrics.llm_errors.add(1, {"purpose": purpose, "model": self.model, "error_type": kind})
                sp.set_attribute("llm.error_type", kind)
                sp.set_attribute("llm.retries", attempts)
                log.warning("LLM call failed", extra={"purpose": purpose, "model": self.model,
                                                      "error": type(exc).__name__, "error_type": kind})
                raise
            finally:
                elapsed = time.perf_counter() - start
                metrics.llm_latency.record(elapsed, {"purpose": purpose, "model": self.model})
            metrics.llm_requests.add(1, {"purpose": purpose, "model": self.model, "status": status})
            tin, tout = resp.input_tokens or 0, resp.output_tokens or 0
            metrics.llm_tokens.add(tin, {"purpose": purpose, "direction": "input"})
            metrics.llm_tokens.add(tout, {"purpose": purpose, "direction": "output"})
            cost = tin / 1000 * self.settings.input_cost_per_1k_tokens + tout / 1000 * self.settings.output_cost_per_1k_tokens
            if cost:
                metrics.llm_cost.add(cost, {"purpose": purpose, "model": self.model})
            sp.set_attribute("llm.token_count.prompt", tin)
            sp.set_attribute("llm.token_count.completion", tout)
            sp.set_attribute("llm.token_count.total", tin + tout)
            sp.set_attribute("llm.estimated_cost_usd", cost)
            sp.set_attribute("llm.retries", attempts)
            if capture_sensitive():
                sp.set_attribute("output.value", resp.text[:20000])
                sp.set_attribute("llm.output_messages.0.message.role", "assistant")
                sp.set_attribute("llm.output_messages.0.message.content", resp.text[:20000])
            resp.cost_usd = cost
            return resp


def error_type(exc: Exception) -> str:
    """timeout | rate_limit | not_configured | unavailable | other, for metrics and alerts."""
    text = f"{type(exc).__name__} {getattr(exc, 'message', '')} {exc}".lower()
    if isinstance(exc, NotConfigured):
        return "not_configured"
    if any(s in text for s in ("timeout", "deadline", "timed out")):
        return "timeout"
    if any(s in text for s in ("429", "ratelimit", "rate limit", "resource_exhausted", "quota")):
        return "rate_limit"
    if any(s in text for s in ("503", "unavailable", "connection")):
        return "unavailable"
    return "other"


def build_llm(settings, prompt_version: str | None = None) -> InstrumentedLLM:
    if not settings.configured:
        raise NotConfigured("LLM provider/model is not configured")
    if settings.provider == "google":
        client: LLMClient = _GoogleClient(settings)
    elif settings.provider == "openai":
        client = _OpenAIClient(settings)
    else:
        raise NotConfigured(f"provider {settings.provider!r} cannot be built from configuration")
    return InstrumentedLLM(client, settings, prompt_version)

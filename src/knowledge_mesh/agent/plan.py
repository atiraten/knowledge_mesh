"""Planner LLM: turns the user question into an execution plan of intents (an ``IntentPlan``).

The planner call uses ``prompts/planner.txt`` as the system prompt and sends the user question
plus the ``IntentPlan`` JSON schema. It never receives the tool list. Each intent carries the
retrieval queries the harness passes to retrieval. An unusable plan (not JSON, schema mismatch,
unknown dependency, cycle, too many intents) falls back to one intent equal to the question.
The same prompt is used to replan when a task comes back unanswered.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from ..common.errors import MeshError, NotConfigured, OperationFailed

log = logging.getLogger(__name__)
MAX_QUERY_CHARS = 2000  # retrieval query length limit


class Intent(BaseModel):
    id: str = Field(description="Short unique id, e.g. I1.")
    intent: str = Field(description="One user objective, written in the user's own terms.")
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="1 to 3 short search queries for this intent, in the user's own terms. Synonyms and "
                    "abbreviations are allowed; added scope is not.")
    depends_on: list[str] = Field(default_factory=list,
                                  description="IDs of the intents whose results this intent needs.")


class IntentPlan(BaseModel):
    original_query: str = Field(description="The user question, verbatim.")
    decompose: bool = Field(description="True when the question was split into several intents.")
    intents: list[Intent]
    reasoning: str = Field(description="Why the question was or was not decomposed.")
    execution: Literal["single", "parallel", "sequential", "mixed"] = Field(
        description="single: one intent; parallel: independent intents; sequential: each intent depends on the "
                    "previous one; mixed: some intents depend on others.")


class AgentUnavailable(MeshError):
    """The agent model cannot be used for this search (not configured, no key, failing calls)."""

    code = "agent_unavailable"
    http_status = 503


class PlanError(ValueError):
    pass


def inline_schema(model: type[BaseModel], require_all: bool = False) -> dict[str, Any]:
    """The model's JSON schema with ``$ref``s inlined and ``title``/``default`` removed, so it can be
    passed to providers that accept only a JSON Schema subset. ``require_all`` marks every property
    required (the parser stays lenient; this only tells the model to fill every field)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]]))
            out = {k: walk(v) for k, v in node.items() if k not in {"title", "default"}}
            if require_all and out.get("type") == "object" and isinstance(out.get("properties"), dict):
                out["required"] = list(out["properties"])
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


PLAN_SCHEMA = inline_schema(IntentPlan, require_all=True)


_VALID_ESCAPES = frozenset('"\\/bfnrtu')
_DECODER = json.JSONDecoder(strict=False)  # tolerate raw newlines/tabs inside strings


def _escape_invalid_backslashes(s: str) -> str:
    """Double every backslash that does not start a valid JSON escape (e.g. LaTeX ``\\{`` in an answer).
    Escape pairs are consumed left to right, so an already escaped backslash (``\\\\{``) is left alone."""
    return re.sub(r"\\(.)", lambda m: m.group(0) if m.group(1) in _VALID_ESCAPES else "\\\\" + m.group(1), s,
                  flags=re.S)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply. Tolerates code fences, surrounding prose and
    invalid backslash escapes (models often write LaTeX such as ``$\\{x\\}$`` inside answers)."""
    s = (text or "").strip()
    if not s.startswith("{"):
        fenced = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if fenced:
            s = fenced.group(1).strip()
    error = "the reply contains no JSON object"
    for candidate in (s, _escape_invalid_backslashes(s)):
        start = candidate.find("{")
        if start < 0:
            break
        try:
            value, _ = _DECODER.raw_decode(candidate[start:])
        except json.JSONDecodeError as exc:
            error = f"the reply is not valid JSON: {exc.msg}"
            continue
        if not isinstance(value, dict):
            raise ValueError("the reply must be a JSON object")
        return value
    raise ValueError(error)


def _with_defaults(raw: dict[str, Any], query: str) -> dict[str, Any]:
    """Fill fields a model may leave out, so a usable plan is not rejected for a missing detail."""
    raw = dict(raw)
    intents = raw.get("intents")
    if isinstance(intents, list):
        fixed = []
        for item in intents:
            if isinstance(item, dict):
                item = dict(item)
                if "intent" not in item and "description" in item:  # earlier plan format
                    item["intent"] = item.pop("description")
            fixed.append(item)
        raw["intents"] = fixed
        raw.setdefault("decompose", len(fixed) > 1)
        raw.setdefault("execution", "single" if len(fixed) == 1 else "mixed")
    raw.setdefault("original_query", query)
    raw.setdefault("reasoning", "")
    return raw


def normalize_queries(intent: Intent, max_queries: int) -> None:
    """Strip, de-duplicate (ignoring case) and cap the retrieval queries; none -> the intent itself."""
    seen: set[str] = set()
    queries: list[str] = []
    for q in intent.retrieval_queries:
        q = " ".join(str(q).split())[:MAX_QUERY_CHARS]
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            queries.append(q)
    intent.retrieval_queries = queries[:max_queries] or [intent.intent.strip()[:MAX_QUERY_CHARS]]


def validate_plan(plan: IntentPlan, max_intents: int) -> list[list[str]]:
    """Check the plan and return its dependency layers (each layer depends only on earlier layers)."""
    if not plan.intents:
        raise PlanError("the plan has no intents")
    if len(plan.intents) > max_intents:
        raise PlanError(f"the plan has {len(plan.intents)} intents; at most {max_intents} are allowed")
    ids = [i.id.strip() for i in plan.intents]
    if any(not i for i in ids):
        raise PlanError("an intent has an empty id")
    if len(set(ids)) != len(ids):
        raise PlanError("intent ids are not unique")
    if any(not i.intent.strip() for i in plan.intents):
        raise PlanError("an intent has an empty description")
    known = set(ids)
    for intent in plan.intents:
        for dep in intent.depends_on:
            if dep == intent.id:
                raise PlanError(f"intent {intent.id} depends on itself")
            if dep not in known:
                raise PlanError(f"intent {intent.id} depends on unknown intent {dep}")
    layers: list[list[str]] = []
    done: set[str] = set()
    pending = list(plan.intents)
    while pending:
        ready = [i for i in pending if set(i.depends_on) <= done]
        if not ready:
            raise PlanError("the intent dependencies form a cycle: " + ", ".join(i.id for i in pending))
        layers.append([i.id for i in ready])
        done |= {i.id for i in ready}
        pending = [i for i in pending if i.id not in done]
    return layers


@dataclass
class PlanResult:
    plan: IntentPlan
    layers: list[list[str]]
    fallback_reason: str | None = None
    raw: str | None = None


def fallback_plan(query: str, reason: str) -> PlanResult:
    plan = IntentPlan(original_query=query, decompose=False,
                      intents=[Intent(id="I1", intent=query, retrieval_queries=[query[:MAX_QUERY_CHARS]])],
                      reasoning=f"Fallback to a single intent: {reason}", execution="single")
    return PlanResult(plan, [["I1"]], reason)


class Planner:
    def __init__(self, settings, llm: Callable[[], Any], read_prompt: Callable[[Any], str]):
        self.cfg = settings
        self._llm = llm
        self._read_prompt = read_prompt

    @staticmethod
    def _schema_text() -> str:
        return ("IntentPlan JSON schema:\n" + json.dumps(PLAN_SCHEMA, indent=2)
                + "\n\nReturn only the IntentPlan JSON object.")

    def message(self, query: str) -> str:
        return "User request:\n" + query + "\n\n" + self._schema_text()

    def replan_message(self, query: str, plan: IntentPlan, task_results: list[dict[str, Any]], reason: str) -> str:
        results = "\n".join(
            f"{r['id']} ({r['status']}, attempt {r['attempt']}): {r['intent']}\n"
            f"Result: {(r.get('answer') or r.get('error') or 'none')[:800]}" for r in task_results)
        return ("User request:\n" + query
                + "\n\nCurrent plan:\n" + plan.model_dump_json(indent=2)
                + "\n\nTask results so far:\n" + (results or "(none)")
                + "\n\nWhy a new plan is needed: " + reason
                + "\n\nRevise the plan. Keep every answered intent unchanged, with the same id. For the intents "
                  "that were not answered, write better retrieval queries, reformulate the intent or split it; "
                  "you may add intents. Do not add scope the user did not state.\n\n" + self._schema_text())

    def _call(self, prompt: str, system: str, purpose: str):
        llm = self._llm()
        version = prompt_version(system)
        try:
            return llm.generate(prompt, system=system, purpose=purpose, json_schema=PLAN_SCHEMA,
                                prompt_version=version)
        except NotConfigured as exc:
            raise AgentUnavailable(f"agent model unavailable: {exc.message}") from exc
        except OperationFailed as exc:
            # Some models reject a response schema: retry once in plain JSON mode (schema stays in the prompt).
            log.warning("planner call with response schema failed; retrying in JSON mode",
                        extra={"error": exc.message[:200]})
            try:
                return llm.generate(prompt, system=system, purpose=purpose, json_output=True,
                                    prompt_version=version)
            except Exception as exc2:
                raise AgentUnavailable(f"agent model call failed: {error_text(exc2)}") from exc2
        except Exception as exc:
            raise AgentUnavailable(f"agent model call failed: {error_text(exc)}") from exc

    def _parse(self, text: str, query: str) -> PlanResult:
        plan = IntentPlan.model_validate(_with_defaults(parse_json_object(text), query))
        for intent in plan.intents:
            normalize_queries(intent, self.cfg.max_retrieval_queries)
        return PlanResult(plan, validate_plan(plan, self.cfg.max_intents), None, text[:4000])

    def plan(self, query: str) -> PlanResult:
        system = self._read_prompt(self.cfg.planner_prompt_path)
        resp = self._call(self.message(query), system, "agent_planner")
        try:
            return self._parse(resp.text, query)
        except (ValueError, ValidationError) as exc:
            reason = short_error(exc)
            log.warning("unusable plan; using a single intent", extra={"reason": reason})
            result = fallback_plan(query, f"unusable plan ({reason})")
            result.raw = resp.text[:4000]
            return result

    def replan(self, query: str, plan: IntentPlan, task_results: list[dict[str, Any]],
               reason: str) -> tuple[PlanResult | None, str | None]:
        """A revised plan, or (None, why) when the replan call fails or returns an unusable plan."""
        system = self._read_prompt(self.cfg.planner_prompt_path)
        try:
            resp = self._call(self.replan_message(query, plan, task_results, reason), system, "agent_replanner")
            return self._parse(resp.text, query), None
        except AgentUnavailable as exc:
            return None, exc.message
        except (ValueError, ValidationError) as exc:
            return None, f"unusable plan ({short_error(exc)})"


def prompt_version(text: str) -> str:
    """Short content hash of a prompt, recorded as ``llm.prompt_template.version`` on every LLM span."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", ()))
        return f"{loc}: {first.get('msg')}" if loc else str(first.get("msg"))
    return str(exc)[:200]


def error_text(exc: Exception) -> str:
    return exc.message if isinstance(exc, MeshError) else f"{type(exc).__name__}: {str(exc)[:200]}"

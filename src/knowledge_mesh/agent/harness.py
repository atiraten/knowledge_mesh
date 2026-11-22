"""Search agent harness.

    user question
      -> planner LLM (prompts/planner.txt, no tools) -> execution plan (intents with retrieval queries)
      -> HARNESS ORCHESTRATION LOOP
           select the next ready task(s)  (dependencies done; ready tasks of a parallel/mixed plan
                                           run at the same time)
           -> EXECUTOR AGENT LOOP per task (prompts/executor.txt): the executor LLM receives the tool
              list, the plan's intents, its task and its earlier steps; it evaluates the latest tool
              results and returns tool calls (run in parallel) or completes the task with a status
              (answered / partial / not_found). The harness runs the tools, adds the results to the
              context and loops, up to agent.max_iterations calls per task.
           -> evaluate the task result (rules): answered -> next task; otherwise replan with the
              planner LLM (at most agent.max_replans times), then continue; nothing ready -> finish
      -> final response (prompts/synthesis.txt): one grounded answer from every task result and
         the gathered evidence; citations are checked against the evidence supplied.

The first retrieve_documents call of each task always searches the task's planned retrieval queries;
a query_knowledge_graph call without intent_id / intent / retrieval_queries gets them from its task.
The raw user question is not shown to the executor, so the work follows the planner's intents.

Clarification (at the response boundary): when, after the replans, a task is still partial / not
found (the executor may say what it needs in ``clarification_needed``) or nothing was found, synthesis
writes one question for the user and the search ends with ``outcome: needs_clarification``. Its state
(plan, task results, evidence) is returned for storage; ``resume`` reloads it with the user's reply,
calls the existing replan step with the reply as the reason (answered tasks are kept), and continues
with the same orchestration loop and synthesis.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..observability import (add_event, capture_sensitive, current_span_context, current_span_id, current_trace_id,
                             get_metrics, span)
from ..observability.audit import RequestContext
from ..observability.quality import run_checks
from ..retrieval.evidence import CITE, check_citations
from .ledger import EvidenceLedger
from .plan import (AgentUnavailable, Intent, IntentPlan, Planner, PlanResult, error_text, parse_json_object,
                   prompt_version, short_error, validate_plan)
from .mcp_client import MAX_QUERIES, McpToolBox, SearchScope, ToolResult

log = logging.getLogger(__name__)
RETRIEVE_TOOL = "retrieve_documents"
GRAPH_TOOL = "query_knowledge_graph"
TASK_STATUSES = ("answered", "partial", "not_found")
UNSURE_STATUSES = ("partial", "not_found", "max_iterations")   # the executor could not answer with confidence
CLARIFY_LINE = re.compile(r"^\s*\**\s*clarification question\s*\**\s*:\s*(.+?)\s*$", re.I | re.M)
MEMORY_NOTE = "none: each search starts with an empty context; nothing is kept between searches"


class _Usage:
    """Tokens and estimated cost of the LLM calls made for one search (shared by its threads)."""

    def __init__(self) -> None:
        self.input_tokens = self.output_tokens = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def add(self, resp) -> None:
        with self._lock:
            self.input_tokens += resp.input_tokens or 0
            self.output_tokens += resp.output_tokens or 0
            self.cost_usd += getattr(resp, "cost_usd", 0.0) or 0.0


_USAGE: contextvars.ContextVar[_Usage | None] = contextvars.ContextVar("agent_usage", default=None)


class _Metered:
    """Wraps the agent model so every call's tokens and cost are added to the current search."""

    def __init__(self, llm):
        self._llm = llm
        self.provider = getattr(llm, "provider", None)
        self.model = getattr(llm, "model", None)

    def generate(self, *args, **kwargs):
        resp = self._llm.generate(*args, **kwargs)
        usage = _USAGE.get()
        if usage is not None:
            usage.add(resp)
        return resp


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tool: str
    arguments: dict[str, Any] | None = None


class AgentAction(BaseModel):
    """One executor reply: evaluate the latest results, then request tool calls or complete the task."""

    model_config = ConfigDict(extra="ignore")
    evaluation: str = ""
    action: Literal["tools", "tool", "complete"]
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool: str | None = None              # single-call form: {"action": "tool", "tool", "arguments"}
    arguments: dict[str, Any] | None = None
    status: str = "answered"
    answer: str | None = None
    clarification_needed: str | None = None   # with partial / not_found: what the user could add

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> str:
        v = str(v or "answered").strip().lower().replace(" ", "_")
        return v if v in TASK_STATUSES else "partial"

    @model_validator(mode="after")
    def _normalize(self) -> "AgentAction":
        if self.action in ("tool", "tools") and not self.tool_calls:
            if not self.tool:
                raise ValueError('"tool_calls" must list at least one tool call when "action" is "tools"')
            self.tool_calls = [ToolCall(tool=self.tool, arguments=self.arguments)]
        if self.action == "tool":
            self.action = "tools"
        if self.action == "complete" and not (self.answer or "").strip():
            raise ValueError('"answer" is required when "action" is "complete"')
        return self


@dataclass
class CallRecord:
    tool: str
    arguments: dict[str, Any]
    result: ToolResult
    note: str | None = None
    planned: bool = False  # carried the task's planned retrieval queries

    def public(self) -> dict[str, Any]:
        out = {"tool": self.tool, "arguments": self.arguments, "status": self.result.status,
               "summary": self.result.summary, "evidence_ids": self.result.evidence_ids}
        if self.note:
            out["note"] = self.note
        if self.result.status == "error":
            out["error"] = self.result.observation.get("error")
        return out


@dataclass
class Step:
    iteration: int
    action: str  # tools | complete | invalid
    evaluation: str = ""
    calls: list[CallRecord] = field(default_factory=list)
    error: dict[str, Any] | None = None   # invalid reply
    cited: list[str] = field(default_factory=list)

    def evidence_ids(self) -> list[str]:
        return [i for c in self.calls for i in c.result.evidence_ids] + self.cited

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {"iteration": self.iteration, "action": self.action, "evaluation": self.evaluation,
                               "tool_calls": [c.public() for c in self.calls]}
        if self.error:
            out["error"] = self.error
        if self.cited:
            out["cited"] = self.cited
        return out


@dataclass
class TaskRun:
    intent: Intent
    attempt: int
    plan_version: int
    status: str  # answered | partial | not_found | max_iterations | failed
    iterations: int
    steps: list[Step]
    answer: str | None = None
    dropped_citations: list[str] = field(default_factory=list)
    error: str | None = None
    planned_searched: bool = False
    replies: int = 0
    invalid_replies: int = 0
    injection_flags: int = 0
    span_context: Any = None  # for span links from the tasks that depend on this one
    clarification_needed: str | None = None
    restored_evidence: list[str] = field(default_factory=list)  # evidence of a task restored after a clarification

    def evidence_ids(self) -> list[str]:
        return list(dict.fromkeys(self.restored_evidence + [i for s in self.steps for i in s.evidence_ids()]))

    def summary(self) -> dict[str, Any]:
        out = {"id": self.intent.id, "intent": self.intent.intent, "status": self.status, "attempt": self.attempt,
               "answer": self.answer, "error": self.error}
        if self.clarification_needed:
            out["clarification_needed"] = self.clarification_needed
        return out

    def state(self) -> dict[str, Any]:
        """What a clarification round keeps of this task run."""
        return {"intent": self.intent.model_dump(), "attempt": self.attempt, "plan_version": self.plan_version,
                "status": self.status, "iterations": self.iterations, "answer": self.answer, "error": self.error,
                "dropped_citations": self.dropped_citations, "planned_searched": self.planned_searched,
                "clarification_needed": self.clarification_needed, "evidence_ids": self.evidence_ids(),
                "steps": [s.public() for s in self.steps]}

    @classmethod
    def from_state(cls, st: dict[str, Any]) -> "TaskRun":
        run = cls(Intent.model_validate(st["intent"]), st["attempt"], st["plan_version"], st["status"],
                  st["iterations"], [], st.get("answer"), st.get("dropped_citations") or [], st.get("error"),
                  st.get("planned_searched", False), clarification_needed=st.get("clarification_needed"),
                  restored_evidence=st.get("evidence_ids") or [])
        run.restored_steps = st.get("steps") or []
        return run

    def public(self) -> dict[str, Any]:
        out = {"id": self.intent.id, "intent": self.intent.intent,
               "retrieval_queries": self.intent.retrieval_queries, "depends_on": self.intent.depends_on,
               "attempt": self.attempt, "plan_version": self.plan_version, "status": self.status,
               "iterations": self.iterations, "answer": self.answer, "dropped_citations": self.dropped_citations,
               "error": self.error, "planned_queries_searched": self.planned_searched,
               "evidence_ids": self.evidence_ids(),
               "steps": [s.public() for s in self.steps] or list(getattr(self, "restored_steps", []))}
        if self.clarification_needed:
            out["clarification_needed"] = self.clarification_needed
        return out


class _Run:
    def __init__(self, query: str, planned: PlanResult, tools: McpToolBox, ledger: EvidenceLedger,
                 executor_prompt: str):
        self.query, self.tools, self.ledger, self.executor_prompt = query, tools, ledger, executor_prompt
        self.executor_version = prompt_version(executor_prompt)
        self.plan: IntentPlan = planned.plan
        self.tool_specs = "\n".join(json.dumps(s, ensure_ascii=False) for s in tools.specs())
        self.plans: list[dict[str, Any]] = [{"version": 1, **planned.plan.model_dump(), "layers": planned.layers,
                                             "fallback_reason": planned.fallback_reason, "replan_reason": None}]
        self.decisions: list[dict[str, Any]] = []
        self.task_runs: list[TaskRun] = []
        self.latest: dict[str, TaskRun] = {}
        self.status: dict[str, str] = {}  # task id -> running / final status (shown to the executor)
        self.llm_calls = 0
        self.tool_calls = 0
        self.output_format = "text"
        self.clarifications: list[dict[str, str]] = []   # earlier rounds: {"question", "answer"}
        self.round = 0                                  # clarification rounds already answered
        self.root_trace_id: str | None = None
        self.continues: str | None = None
        self._lock = threading.Lock()

    def count(self, attr: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + n)

    def set_status(self, task_id: str, status: str) -> None:
        with self._lock:
            self.status[task_id] = status


def _scope_state(scope: SearchScope | None) -> dict[str, Any] | None:
    if scope is None:
        return None
    from dataclasses import asdict

    return asdict(scope)


class SearchHarness:
    def __init__(self, settings, llm_factory: Callable[[], Any] | None,
                 tools_factory: Callable[[EvidenceLedger, SearchScope | None], McpToolBox]):
        self.cfg = settings
        self._factory = llm_factory
        self._tools_factory = tools_factory
        self._llm = None
        self._lock = threading.Lock()
        self.planner = Planner(settings, self._client, self._prompt)

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self._factory is not None)

    def disabled_reason(self) -> str | None:
        if self.enabled:
            return None
        if not self.cfg.enabled:
            return "the search agent is disabled (agent.enabled is false in config/mesh.yaml)"
        return "no agent model is configured (agent.provider and agent.model in config/mesh.yaml)"

    def _client(self):
        with self._lock:
            if self._llm is None:
                if self._factory is None:
                    raise AgentUnavailable(self.disabled_reason() or "agent model unavailable")
                try:
                    self._llm = _Metered(self._factory())
                except Exception as exc:
                    raise AgentUnavailable(f"agent model unavailable: {error_text(exc)}") from exc
            return self._llm

    @staticmethod
    def _prompt(path) -> str:
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise AgentUnavailable(f"agent prompt not readable: {path}") from exc
        if not text:
            raise AgentUnavailable(f"agent prompt is empty: {path}")
        return text

    # -- run -------------------------------------------------------------------------------------
    def run(self, query: str, scope: SearchScope | None = None,
            context: RequestContext | None = None) -> dict[str, Any]:
        context = context or RequestContext()
        started = time.perf_counter()
        usage_token = _USAGE.set(_Usage())
        try:
            return self._run(query, scope, context, started)
        finally:
            _USAGE.reset(usage_token)

    def resume(self, state: dict[str, Any], reply: str, scope: SearchScope | None = None,
               context: RequestContext | None = None) -> dict[str, Any]:
        """Continue a search that asked for clarification, with the user's reply (state from ``_clarification_state``)."""
        context = context or RequestContext()
        started = time.perf_counter()
        usage_token = _USAGE.set(_Usage())
        try:
            return self._run(state["query"], scope, context, started, resume=(state, reply))
        finally:
            _USAGE.reset(usage_token)

    def _run(self, query: str, scope: SearchScope | None, context: RequestContext, started: float,
             resume: tuple[dict[str, Any], str] | None = None) -> dict[str, Any]:
        metrics = get_metrics()
        usage = _USAGE.get()
        root_attrs = {"input.value": query[:500], "agent.max_iterations": self.cfg.max_iterations,
                      "user.id": context.caller, "session.id": context.session_id, "agent.memory": MEMORY_NOTE}
        if resume is not None:
            root_attrs["agent.clarification.round"] = resume[0]["round"] + 1
            root_attrs["agent.continues"] = resume[0]["trace_id"]
        with span("agent.search", "AGENT", root_attrs) as sp:
            executor_prompt = self._prompt(self.cfg.executor_prompt_path)
            synthesis_prompt = self._prompt(self.cfg.synthesis_prompt_path)
            versions = {"planner": prompt_version(self._prompt(self.cfg.planner_prompt_path)),
                        "executor": prompt_version(executor_prompt), "synthesis": prompt_version(synthesis_prompt)}
            sp.set_attribute("agent.prompt_versions", json.dumps(versions))
            if resume is not None:
                run, answered = self._restore(resume[0], resume[1], scope, executor_prompt)
                return self._finish(run, answered, scope, synthesis_prompt, versions, sp, started, usage)
            with span("agent.plan", "CHAIN", {"input.value": query[:500]}) as psp:
                planned = self.planner.plan(query)
                psp.set_attribute("agent.plan", planned.plan.model_dump_json())
                psp.set_attribute("agent.plan.intents", len(planned.plan.intents))
                psp.set_attribute("agent.plan.execution", planned.plan.execution)
                psp.set_attribute("agent.plan.decompose", planned.plan.decompose)
                if planned.fallback_reason:
                    psp.set_attribute("agent.plan.fallback_reason", planned.fallback_reason)
            metrics.agent_plans.add(1, {"kind": "initial", "outcome": "fallback" if planned.fallback_reason else "ok"})
            metrics.agent_intents.record(len(planned.plan.intents))
            if planned.fallback_reason:
                metrics.agent_recoveries.add(1, {"kind": "plan_fallback"})
            ledger = EvidenceLedger()
            run = _Run(query, planned, self._tools_factory(ledger, scope), ledger, executor_prompt)
            run.count("llm_calls")  # the planner call
            return self._finish(run, set(), scope, synthesis_prompt, versions, sp, started, usage)

    def _finish(self, run: _Run, answered: set[str], scope: SearchScope | None, synthesis_prompt: str,
                versions: dict[str, str], sp, started: float, usage: _Usage) -> dict[str, Any]:
        """Orchestration loop, final response (or clarification question) and the search result."""
        metrics = get_metrics()
        query, ledger = run.query, run.ledger
        run.output_format = (scope.output_format if scope is not None else None) or run.output_format
        self._orchestrate(run, answered)
        notes = self._clarification_notes(run)
        final = self._synthesize(run, synthesis_prompt, notes)
        question = final.get("clarification_question")
        if notes is not None and not question:
            question = self._fallback_question(notes)
        if notes is None:
            question = None
        outcome = "needs_clarification" if question else ("answered" if final.get("answer") else "failed")
        if question:
            sp.set_attribute("agent.clarification.question_chars", len(question))
        evaluation = run_checks(self._facts(run, final))
        failed = [c["name"] for c in evaluation["checks"] if not c["passed"]]
        sp.set_attribute("agent.outcome", outcome)
        sp.set_attribute("agent.decisions", json.dumps(run.decisions)[:4000])
        sp.set_attribute("agent.plan_versions", len(run.plans))
        sp.set_attribute("agent.task_statuses", [f"{t.intent.id}:{t.status}" for t in run.task_runs])
        sp.set_attribute("agent.llm_calls", run.llm_calls)
        sp.set_attribute("agent.tool_calls", run.tool_calls)
        sp.set_attribute("agent.eval.compliance", evaluation["compliance"] if evaluation["compliance"] is not None
                         else -1.0)
        sp.set_attribute("agent.eval.failed_checks", failed)
        sp.set_attribute("agent.eval.violations", evaluation["violations"])
        if evaluation["citation_coverage"] is not None:
            sp.set_attribute("agent.eval.citation_coverage", evaluation["citation_coverage"])
        if capture_sensitive() and final.get("answer"):
            sp.set_attribute("output.value", final["answer"])
        elapsed = time.perf_counter() - started
        sp.set_attribute("llm.token_count.prompt", usage.input_tokens)
        sp.set_attribute("llm.token_count.completion", usage.output_tokens)
        sp.set_attribute("agent.estimated_cost_usd", round(usage.cost_usd, 6))
        metrics.agent_runs.add(1, {"outcome": outcome})
        metrics.agent_search_duration.record(elapsed, {"outcome": outcome})
        metrics.search_tokens.record(usage.input_tokens + usage.output_tokens)
        metrics.search_cost.record(usage.cost_usd)
        metrics.search_evidence.record(len(ledger))
        llm = self._client()
        model = getattr(llm, "model", None)
        trace_id, span_id = current_trace_id(), current_span_id()
        latest = [run.latest[i] for i in run.latest]
        result = {
            "query": query, "mode": "agent",
            "agent": {"used": True, "provider": getattr(llm, "provider", None), "model": model,
                      "max_iterations": self.cfg.max_iterations, "max_replans": self.cfg.max_replans},
            "models": {"planner": model, "executor": model, "synthesis": model},
            "prompt_versions": versions,
            "plan": run.plans[-1], "plans": run.plans, "decisions": run.decisions,
            "tasks": [t.public() for t in run.task_runs],
            "answer": final.get("answer"), "grounded": bool(final.get("citations")),
            "citations": final.get("citations", []), "dropped_citations": final.get("dropped_citations", []),
            "error": final.get("error"), "outcome": outcome,
            "evidence": ledger.public(), "insufficient_evidence": len(ledger) == 0,
            "evaluation": evaluation,
            "usage": {"llm_calls": run.llm_calls, "tool_calls": run.tool_calls,
                      "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                      "cost_usd": round(usage.cost_usd, 6), "latency_ms": int(elapsed * 1000)},
            "observability": {"trace_id": trace_id, "span_id": span_id, "memory": MEMORY_NOTE},
            "trace_id": trace_id,
            # internal: popped by KnowledgeMesh.search (judge input and audit resources)
            "_judge_input": {"trace_id": trace_id, "span_id": span_id, "query": query,
                             "intents": [{"id": i.id, "intent": i.intent} for i in run.plan.intents],
                             "tasks": [{"id": t.intent.id, "status": t.status, "answer": t.answer} for t in latest],
                             "evidence": final.get("evidence_text", ""), "answer": final.get("answer")},
            "_resources": self._resources(run),
        }
        result["html"] = [{"id": i, "title": it.data.get("title"), "html": it.data["html"],
                           "truncated": it.data.get("truncated", False)}
                          for i in ledger.ids() if (it := ledger.item(i)) is not None and it.kind == "html_fragment"]
        result["output_format"] = run.output_format
        if run.continues:
            result["continues"] = run.continues
            result["clarifications"] = run.clarifications
        if question:
            round_ = run.round + 1
            result["clarification"] = {"question": question, "round": round_,
                                       "max_rounds": self.cfg.max_clarifications,
                                       "reply_ref": f"/api/searches/{trace_id}/clarification"}
            # internal: stored by KnowledgeMesh.search until the user replies (or it expires)
            result["_clarification_state"] = {
                "query": query, "trace_id": trace_id, "root_trace_id": run.root_trace_id or trace_id,
                "round": run.round, "question": question, "output_format": run.output_format,
                "scope": _scope_state(scope), "plan": run.plan.model_dump(), "plans": run.plans,
                "decisions": run.decisions, "tasks": [t.state() for t in run.latest.values()],
                "ledger": ledger.export(), "clarifications": run.clarifications + [{"question": question,
                                                                                    "answer": None}]}
        return result

    # -- clarification ---------------------------------------------------------------------------
    def _clarification_notes(self, run: _Run) -> list[dict[str, Any]] | None:
        """Why the user should be asked (None: no question). Triggers, after the replans: a task the executor
        could not answer with confidence (partial / not_found / iteration limit), or nothing found at all."""
        if not self.cfg.clarification_enabled or run.round >= self.cfg.max_clarifications:
            return None
        final_ids = [i.id for i in run.plan.intents]
        tasks = [run.latest[i] for i in final_ids if i in run.latest]
        unsure = [t for t in tasks if t.status in UNSURE_STATUSES]
        nothing = len(run.ledger) == 0 and not any(t.status == "failed" for t in tasks)
        if not unsure and not nothing:
            return None
        notes = [{"id": t.intent.id, "intent": t.intent.intent, "status": t.status,
                  "needed": t.clarification_needed, "result": (t.answer or t.error or "")[:400]} for t in unsure]
        if nothing and not notes:
            notes = [{"id": t.intent.id, "intent": t.intent.intent, "status": t.status,
                      "needed": t.clarification_needed, "result": "no information found"} for t in tasks]
        return notes

    @staticmethod
    def _fallback_question(notes: list[dict[str, Any]]) -> str:
        needs = [n["needed"] for n in notes if n.get("needed")]
        if needs:
            return "To answer, I need more information: " + " ".join(dict.fromkeys(needs))
        intents = "; ".join(n["intent"] for n in notes)
        return (f"I could not find information for: {intents}. Can you add details (other names, the dataset or "
                "document to use, or the time period) so I can continue?")

    def _restore(self, state: dict[str, Any], reply: str, scope: SearchScope | None,
                 executor_prompt: str) -> tuple[_Run, set[str]]:
        """Rebuild the search from its saved state, then revise the plan with the user's reply (existing replan)."""
        plan = IntentPlan.model_validate(state["plan"])
        ledger = EvidenceLedger.restore(state["ledger"])
        run = _Run(state["query"], PlanResult(plan, validate_plan(plan, max(self.cfg.max_intents, len(plan.intents)))),
                   self._tools_factory(ledger, scope), ledger, executor_prompt)
        run.plans = list(state.get("plans") or run.plans)
        run.decisions = list(state.get("decisions") or [])
        for st in state.get("tasks") or []:
            t = TaskRun.from_state(st)
            run.task_runs.append(t)
            run.latest[t.intent.id] = t
            run.set_status(t.intent.id, t.status)
        run.output_format = state.get("output_format") or "text"
        run.clarifications = [dict(c) for c in state.get("clarifications") or []]
        reply = " ".join(str(reply or "").split())[:4000]
        if run.clarifications:
            run.clarifications[-1]["answer"] = reply
        run.round = int(state.get("round", 0)) + 1
        run.root_trace_id = state.get("root_trace_id")
        run.continues = state.get("trace_id")
        question = run.clarifications[-1]["question"] if run.clarifications else ""
        reason = f"the user was asked: {question!r}; the user answered: {reply!r}"
        get_metrics().agent_recoveries.add(1, {"kind": "clarification"})
        with span("agent.replan", "CHAIN", {"agent.replan.reason": "user clarification"}) as rsp:
            revised, why_not = self.planner.replan(run.query, plan, [t.summary() for t in run.latest.values()], reason)
            run.count("llm_calls")
            if revised is None:
                rsp.set_attribute("agent.replan.error", (why_not or "")[:500])
                get_metrics().agent_plans.add(1, {"kind": "replan", "outcome": "failed"})
                self._decide(run, {"after": [], "decision": "next",
                                   "reason": f"user clarification; replan failed: {why_not}"})
            else:
                get_metrics().agent_plans.add(1, {"kind": "replan", "outcome": "ok"})
                run.plan = revised.plan
                run.plans.append({"version": len(run.plans) + 1, **revised.plan.model_dump(), "layers": revised.layers,
                                  "fallback_reason": None, "replan_reason": f"user clarification: {reply[:300]}"})
                self._decide(run, {"after": [], "decision": "replan", "reason": "user clarification",
                                   "plan_version": len(run.plans)})
        answered = {tid for tid, t in run.latest.items() if t.status == "answered"}
        return run, answered

    @staticmethod
    def _facts(run: _Run, final: dict[str, Any]) -> dict[str, Any]:
        scores = [it.data.get("score") for i in run.ledger.ids() if (it := run.ledger.item(i)) is not None
                  and it.kind in ("document_passage", "glossary") and isinstance(it.data.get("score"), (int, float))]
        flagged = [c["id"] for c in final.get("citations", [])
                   if (it := run.ledger.item(c["id"])) is not None and it.kind == "document_passage"
                   and "suspicious_instruction" in (it.data.get("flags") or [])]
        return {
            "plan_fallback": run.plans[0]["fallback_reason"],
            "task_runs": [{"id": t.intent.id, "status": t.status, "planned_searched": t.planned_searched}
                          for t in run.task_runs],
            "executor_replies": sum(t.replies for t in run.task_runs),
            "invalid_replies": sum(t.invalid_replies for t in run.task_runs),
            "latest_statuses": {tid: t.status for tid, t in run.latest.items()},
            "answer": final.get("answer"), "evidence_ids": run.ledger.ids(),
            "dropped_citations": final.get("dropped_citations", []), "evidence_count": len(run.ledger),
            "top_score": round(max(scores), 4) if scores else None, "flagged_cited": flagged,
            "injection_flags": sum(t.injection_flags for t in run.task_runs),
        }

    @staticmethod
    def _resources(run: _Run) -> dict[str, list[str]]:
        documents, datasets = [], []
        for ident in run.ledger.ids():
            it = run.ledger.item(ident)
            if it is None:
                continue
            if it.kind in ("document_passage", "document_text") and it.data.get("document_id"):
                documents.append(it.data["document_id"])
            elif it.kind == "dataset":
                datasets.append(it.data["dataset_id"])
            elif it.kind == "glossary" and it.data.get("dataset_id"):
                datasets.append(it.data["dataset_id"])
        tools = [c.tool for t in run.task_runs for s in t.steps for c in s.calls]
        return {"documents": list(dict.fromkeys(documents)), "datasets": list(dict.fromkeys(datasets)),
                "tools": list(dict.fromkeys(tools))}

    # -- orchestration loop ----------------------------------------------------------------------
    def _orchestrate(self, run: _Run, answered: set[str] | None = None) -> None:
        answered = set(answered or ())  # tasks answered before a clarification round are not rerun
        ran: set[str] = set()  # tasks with a result in the current plan version
        replans = 0
        while True:
            plan = run.plan
            ready = [i for i in plan.intents if i.id not in answered and i.id not in ran
                     and all(d in answered or d in ran for d in i.depends_on)]
            if not ready:
                self._decide(run, {"after": [], "decision": "finish", "reason": "every task of the plan has a result"})
                break
            batch = ready if plan.execution in ("parallel", "mixed") else ready[:1]
            batch = batch[: self.cfg.max_parallel_intents]
            results = self._run_batch(run, batch)
            for r in results:
                run.task_runs.append(r)
                run.latest[r.intent.id] = r
                ran.add(r.intent.id)
                if r.status == "answered":
                    answered.add(r.intent.id)
            after = [r.intent.id for r in results]
            unanswered = [r for r in results if r.status != "answered"]
            # -- evaluate the task results ------------------------------------------------------
            if not unanswered:
                self._decide(run, {"after": after, "decision": "next", "reason": "answered"})
                continue
            reason = "; ".join(f"{r.intent.id} {r.status}" + (f" ({r.error})" if r.error else "") for r in unanswered)
            if replans >= self.cfg.max_replans:
                self._decide(run, {"after": after, "decision": "next",
                                   "reason": f"{reason}; no replan left (agent.max_replans = {self.cfg.max_replans})"})
                continue
            replans += 1
            get_metrics().agent_recoveries.add(1, {"kind": "replan"})
            with span("agent.replan", "CHAIN", {"agent.replan.reason": reason[:500]}) as rsp:
                revised, why_not = self.planner.replan(run.query, plan, [t.summary() for t in run.latest.values()],
                                                       reason)
                run.count("llm_calls")
                if revised is None:
                    rsp.set_attribute("agent.replan.error", (why_not or "")[:500])
                    get_metrics().agent_plans.add(1, {"kind": "replan", "outcome": "failed"})
                    self._decide(run, {"after": after, "decision": "next",
                                       "reason": f"{reason}; replan failed: {why_not}"})
                    continue
                rsp.set_attribute("agent.plan", revised.plan.model_dump_json())
                get_metrics().agent_plans.add(1, {"kind": "replan", "outcome": "ok"})
                get_metrics().agent_intents.record(len(revised.plan.intents))
            run.plan = revised.plan
            ran = set()
            run.plans.append({"version": len(run.plans) + 1, **revised.plan.model_dump(), "layers": revised.layers,
                              "fallback_reason": None, "replan_reason": reason})
            self._decide(run, {"after": after, "decision": "replan", "reason": reason,
                               "plan_version": len(run.plans)})

    @staticmethod
    def _decide(run: _Run, decision: dict[str, Any]) -> None:
        """Record an orchestration decision (next / replan / finish) in the response and as a span event."""
        run.decisions.append(decision)
        add_event("agent.decision", {"decision": decision["decision"], "after": decision["after"],
                                     "reason": decision.get("reason"), "plan_version": decision.get("plan_version")})

    def _run_batch(self, run: _Run, batch: list[Intent]) -> list[TaskRun]:
        def deps(task: Intent) -> list[TaskRun]:
            return [run.latest[d] for d in task.depends_on if d in run.latest]

        version = len(run.plans)
        attempts = {t.id: 1 + sum(1 for r in run.task_runs if r.intent.id == t.id) for t in batch}
        if len(batch) == 1:
            t = batch[0]
            return [self._run_task(run, t, attempts[t.id], version, deps(t))]
        with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="agent-task") as pool:
            futures = [pool.submit(contextvars.copy_context().run, self._run_task, run, t, attempts[t.id], version,
                                   deps(t)) for t in batch]
            return [f.result() for f in futures]

    # -- executor agent loop (one task) ----------------------------------------------------------
    def _run_task(self, run: _Run, task: Intent, attempt: int, version: int, deps: list[TaskRun]) -> TaskRun:
        metrics = get_metrics()
        steps: list[Step] = []
        status, answer, error, dropped, calls = "max_iterations", None, None, [], 0
        clarification: str | None = None
        planned_searched = False   # have the task's planned retrieval queries been searched yet?
        seen: set[str] = set()     # evidence IDs already shown in this task's context
        invalid = flags = 0
        run.set_status(task.id, "running")
        links = [d.span_context for d in deps if d.span_context is not None]
        with span("agent.intent", "CHAIN", {"agent.intent.id": task.id, "agent.intent.attempt": attempt,
                                            "agent.intent.plan_version": version,
                                            "agent.intent.depends_on": task.depends_on,
                                            "agent.intent.retrieval_queries": task.retrieval_queries,
                                            "input.value": task.intent[:500]}, links=links) as sp:
            span_context = current_span_context()
            for iteration in range(1, self.cfg.max_iterations + 1):
                prompt, ctx = self._step_prompt(run, task, deps, steps, iteration)
                add_event("agent.context", {"iteration": iteration, **ctx})
                metrics.agent_context_chars.record(ctx["context_chars"])
                if ctx["stubbed_results"]:
                    metrics.agent_context_pruned.add(ctx["stubbed_results"], {"kind": "stubbed"})
                try:
                    resp = self._client().generate(prompt, system=run.executor_prompt, purpose="agent_step",
                                                   json_output=True, prompt_version=run.executor_version)
                except Exception as exc:
                    status, error = "failed", f"agent model call failed: {error_text(exc)}"
                    metrics.agent_recoveries.add(1, {"kind": "llm_failure"})
                    log.warning("agent step failed", extra={"intent": task.id, "error": error[:200]})
                    break
                calls += 1
                run.count("llm_calls")
                try:
                    action = AgentAction.model_validate(parse_json_object(resp.text))
                    if len(action.tool_calls) > self.cfg.max_tool_calls_per_step:
                        raise ValueError(f"at most {self.cfg.max_tool_calls_per_step} tool calls per turn")
                except (ValueError, ValidationError) as exc:
                    msg = short_error(exc)
                    invalid += 1
                    metrics.agent_recoveries.add(1, {"kind": "invalid_reply"})
                    steps.append(Step(iteration, "invalid", error={
                        "code": "invalid_action",
                        "message": f"Your reply was not a valid action ({msg}). Reply with one JSON object as specified."}))
                    add_event("agent.step", {"iteration": iteration, "action": "invalid", "error": msg})
                    continue
                explanation = action.evaluation[:1000] if capture_sensitive() else None
                if action.action == "complete":
                    known = dict.fromkeys(run.ledger.ids())
                    answer, cited, dropped = check_citations(action.answer or "", known)
                    status = action.status
                    if status != "answered" and action.clarification_needed:
                        clarification = " ".join(action.clarification_needed.split())[:500]
                    steps.append(Step(iteration, "complete", action.evaluation, cited=cited))
                    add_event("agent.step", {"iteration": iteration, "action": "complete", "status": status,
                                             "cited": cited, "evaluation": explanation})
                    break
                records = self._run_calls(run, task, action.tool_calls, planned_searched, frozenset(seen))
                for c in records:
                    if c.planned and c.result.status == "ok":
                        planned_searched = True
                    if c.result.status == "error":
                        metrics.agent_recoveries.add(1, {"kind": "tool_error"})
                    flags += c.result.injection_flags
                    seen.update(c.result.evidence_ids)
                steps.append(Step(iteration, "tools", action.evaluation, records))
                add_event("agent.step", {"iteration": iteration, "action": "tools",
                                         "tools": [c.tool for c in records],
                                         "statuses": [c.result.status for c in records],
                                         "evidence_ids": [i for c in records for i in c.result.evidence_ids],
                                         "evaluation": explanation})
            if status == "max_iterations":
                metrics.agent_recoveries.add(1, {"kind": "iteration_cap"})
            if flags:
                metrics.guardrail_events.add(flags, {"kind": "suspicious_tool_result"})
            sp.set_attribute("agent.intent.status", status)
            sp.set_attribute("agent.intent.iterations", calls)
            sp.set_attribute("agent.intent.invalid_replies", invalid)
            sp.set_attribute("agent.intent.planned_queries_searched", planned_searched)
            sp.set_attribute("agent.intent.tools", [c.tool for s in steps for c in s.calls])
            if capture_sensitive() and answer:
                sp.set_attribute("output.value", answer)
        run.set_status(task.id, status)
        metrics.agent_iterations.record(calls, {"status": status})
        metrics.agent_tasks.add(1, {"status": status})
        return TaskRun(task, attempt, version, status, calls, steps, answer, dropped, error, planned_searched,
                       calls, invalid, flags, span_context, clarification)

    def _run_calls(self, run: _Run, task: Intent, tool_calls: list[ToolCall], planned_searched: bool,
                   seen: frozenset) -> list[CallRecord]:
        """Run one turn's tool calls in parallel. The task's first retrieval always includes its planned
        retrieval queries (first), followed by any extra queries the model asked for."""
        prepared: list[tuple[str, dict[str, Any], str | None, bool]] = []
        planned_assigned = planned_searched
        for c in tool_calls:
            args, note, planned = dict(c.arguments or {}), None, False
            if c.tool == RETRIEVE_TOOL and not planned_assigned:
                args, note = self._with_planned_queries(task, args)
                planned_assigned = planned = True
            elif c.tool == GRAPH_TOOL:
                args, note = self._with_intent(task, args)
            prepared.append((c.tool, args, note, planned))
        if len(prepared) == 1:
            results = [run.tools.call(prepared[0][0], prepared[0][1], seen)]
        else:
            workers = min(self.cfg.max_parallel_tool_calls, len(prepared))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent-tool") as pool:
                futures = [pool.submit(contextvars.copy_context().run, run.tools.call, tool, args, seen)
                           for tool, args, _, _ in prepared]
                results = [f.result() for f in futures]
        run.count("tool_calls", len(prepared))
        metrics = get_metrics()
        for result in results:
            if result.truncated:
                metrics.agent_context_pruned.add(1, {"kind": "truncated"})
            if result.compacted:
                metrics.agent_context_pruned.add(result.compacted, {"kind": "seen_before"})
        return [CallRecord(tool, args, result, note, planned)
                for (tool, args, note, planned), result in zip(prepared, results)]

    @staticmethod
    def _with_planned_queries(task: Intent, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        proposed = args.pop("queries", None)
        single = args.pop("query", None)
        proposed = [q for q in (proposed if isinstance(proposed, list) else []) if isinstance(q, str)]
        if not proposed and isinstance(single, str):
            proposed = [single]
        keys = {q.casefold() for q in proposed}
        missing = [q for q in task.retrieval_queries if q.casefold() not in keys]
        merged: list[str] = []
        for q in task.retrieval_queries + proposed:
            if q.strip() and q.casefold() not in {m.casefold() for m in merged}:
                merged.append(q)
        args["queries"] = merged[:MAX_QUERIES]
        note = f"planned retrieval queries added by the harness: {missing}" if missing else None
        return args, note

    @staticmethod
    def _with_intent(task: Intent, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Fill only the missing intent fields of a query_knowledge_graph call from the current task."""
        added = []
        if not args.get("intent_id"):
            args["intent_id"] = task.id
            added.append("intent_id")
        if not str(args.get("intent") or "").strip():
            args["intent"] = task.intent
            added.append("intent")
        if not args.get("retrieval_queries"):
            args["retrieval_queries"] = list(task.retrieval_queries)
            added.append("retrieval_queries")
        return args, (f"intent fields added by the harness: {added}" if added else None)

    def _step_prompt(self, run: _Run, task: Intent, deps: list[TaskRun], steps: list[Step],
                     iteration: int) -> tuple[str, dict[str, Any]]:
        intents = []
        for i in run.plan.intents:
            intents += [f"- {i.id}: {i.intent}",
                        "  retrieval queries: " + "; ".join(json.dumps(q, ensure_ascii=False) for q in i.retrieval_queries),
                        f"  depends on: {', '.join(i.depends_on) or 'none'}",
                        f"  status: {run.status.get(i.id, 'pending')}"]
        head = [
            "## Tools",
            "Call tools by name, with arguments matching their parameters (JSON Schema). Several tool calls in one "
            "turn run in parallel.",
            run.tool_specs,
            "", "## Intents (from the planner)",
            f"Execution: {run.plan.execution}", *intents,
            "", "## Current task", f"{task.id}: {task.intent}",
            "Retrieval queries: " + "; ".join(json.dumps(q, ensure_ascii=False) for q in task.retrieval_queries),
        ]
        if run.output_format == "html":
            head += ["Output format requested: HTML for web display. When the task's result is ready, call "
                     "generate_html with the relevant content (result_ref for execute_sql rows) before completing."]
        answered_clarifications = [c for c in run.clarifications if c.get("answer")]
        if answered_clarifications:
            head += ["", "## Clarifications from the user"]
            for c in answered_clarifications:
                head += [f"Question: {c['question']}", f"Answer: {c['answer']}"]
        if deps:
            head += ["", "## Results of the tasks this one depends on"]
            for d in deps:
                findings = d.answer or f"not completed ({d.error or d.status})"
                head += [f"{d.intent.id} ({d.status}): {d.intent.intent}", f"Result: {findings}",
                         f"Evidence gathered: {', '.join(d.evidence_ids()) or 'none'}"]
        last = iteration == self.cfg.max_iterations
        tail = ["", "## Your turn",
                f"Iteration {iteration} of {self.cfg.max_iterations}."
                + (" This is the last iteration: complete the task now with the evidence gathered." if last else "")
                + " Reply with one JSON object."]
        fixed = sum(len(x) + 1 for x in head + tail)
        rendered, stubbed = self._render_steps(steps, max(1000, self.cfg.max_context_chars - fixed))
        body = ["", "## Steps so far", *(rendered or ["(none yet)"])]
        prompt = "\n".join(head + body + tail)
        calls = [c for s in steps for c in s.calls]
        stats = {"context_chars": len(prompt) + len(run.executor_prompt),
                 "estimated_tokens": (len(prompt) + len(run.executor_prompt)) // 4,
                 "system_prompt_version": run.executor_version, "steps_shown": len(steps),
                 "results_shown": len(calls), "stubbed_results": stubbed,
                 "truncated_results": sum(1 for c in calls if c.result.truncated),
                 "seen_before_items": sum(c.result.compacted for c in calls), "dependency_results": len(deps)}
        return prompt, stats

    @staticmethod
    def _render_step(s: Step, full: bool) -> str:
        lines = [f"### Step {s.iteration}"]
        if s.evaluation:
            lines.append(f"Evaluation: {s.evaluation}")
        if s.action == "invalid":
            lines.append("Action: (invalid reply)")
            lines.append(f"Result: {json.dumps({'ok': False, 'error': s.error}, ensure_ascii=False)}")
        for n, c in enumerate(s.calls, start=1):
            lines.append(f"Call {n}: {c.tool} {json.dumps(c.arguments, ensure_ascii=False)}")
            if c.note:
                lines.append(f"Note {n}: {c.note}")
            if full:
                lines.append(f"Result {n}: {json.dumps(c.result.observation, ensure_ascii=False, default=str)}")
            else:
                ids = ", ".join(c.result.evidence_ids) or "none"
                lines.append(f"Result {n}: [omitted to save space: {c.result.summary}; evidence: {ids}]")
        return "\n".join(lines)

    def _render_steps(self, steps: list[Step], budget: int) -> tuple[list[str], int]:
        """Every step, newest in full; the oldest results become one-line stubs when over budget.
        Returns the rendered steps and how many tool results were replaced by stubs."""
        rendered = [self._render_step(s, True) for s in steps]
        total = sum(len(r) + 1 for r in rendered)
        stubbed = 0
        for i in range(len(steps) - 1):
            if total <= budget:
                break
            short = self._render_step(steps[i], False)
            total -= len(rendered[i]) - len(short)
            rendered[i] = short
            stubbed += len(steps[i].calls)
        return rendered, stubbed

    # -- final response --------------------------------------------------------------------------
    def _synthesize(self, run: _Run, system: str, notes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        final_ids = [i.id for i in run.plan.intents]
        tasks = [run.latest[i] for i in final_ids if i in run.latest]
        # answered tasks dropped by a replan still count; unanswered ones were superseded
        tasks += [t for tid, t in run.latest.items() if tid not in final_ids and t.status == "answered"]
        cited_first = [i for t in tasks for i in CITE.findall(t.answer or "") if i in run.ledger]
        ordered = list(dict.fromkeys(cited_first + run.ledger.ids()))
        blocks, catalog = run.ledger.render(ordered[: self.cfg.max_evidence_items])
        kept, used = [], 0
        for block in blocks:  # bound the evidence by size too (the cited items come first)
            if kept and used + len(block) > self.cfg.max_context_chars:
                break
            kept.append(block)
            used += len(block)
        catalog = {k: v for k, v in catalog.items() if any(b.startswith(f"[{k}]") for b in kept)}
        findings = "\n\n".join(
            f"{t.intent.id} ({t.status}): {t.intent.intent}\n"
            f"Result: {t.answer or 'not completed — ' + (t.error or 'the iteration limit was reached')}"
            for t in tasks)
        clarified = "".join(f"\nQuestion: {c['question']}\nAnswer: {c['answer']}" for c in run.clarifications
                            if c.get("answer"))
        prompt = (f"User request: {run.query}\n"
                  + (f"\n<user_clarifications>{clarified}\n</user_clarifications>\n" if clarified else "")
                  + f"\n<task_results>\n{findings or '(no task completed)'}\n"
                  "</task_results>\n\n<evidence>\n"
                  + ("\n\n".join(kept) if kept else "(no evidence was found)")
                  + "\n</evidence>\n\n")
        if notes:
            lines = "\n".join(f"- {n['id']} ({n['status']}): {n['intent']}"
                              + (f"; missing: {n['needed']}" if n.get("needed") else "") for n in notes)
            prompt += (f"<clarification_notes>\n{lines}\n</clarification_notes>\n\n"
                       "The request is not fully answered. Write the answer from the available evidence, then end "
                       "with one line starting with \"Clarification question:\" that asks the user for the specific "
                       "information needed to answer.\n\n")
        prompt += "Write the final answer with citations:"
        with span("agent.synthesis", "CHAIN", {"agent.evidence_ids": list(catalog)}) as sp:
            try:
                resp = self._client().generate(prompt, system=system, purpose="agent_synthesis",
                                               prompt_version=prompt_version(system))
            except Exception as exc:
                msg = f"answer synthesis failed: {error_text(exc)}"
                sp.set_attribute("agent.synthesis.error", msg[:500])
                return {"answer": None, "error": msg, "citations": [], "dropped_citations": []}
            run.count("llm_calls")
            raw_text, question = resp.text, None
            if notes:
                found = CLARIFY_LINE.findall(raw_text)
                if found:
                    question = found[-1].strip()[:1000]
                    raw_text = CLARIFY_LINE.sub("", raw_text).strip()
            text, cited, dropped = check_citations(raw_text, catalog)
            sp.set_attribute("agent.citations", cited)
            sp.set_attribute("agent.invalid_citations", dropped)
        return {"answer": text, "clarification_question": question,
                "citations": [{"id": i, **catalog[i]} for i in cited], "dropped_citations": dropped,
                "evidence_text": "\n\n".join(kept)}

"""Online quality evaluation of agent searches.

* ``scan_sensitive``: PII and secrets in text (emails, phone numbers, card numbers with a Luhn check,
  IBANs with the mod-97 check, US SSNs, API-key-like strings). Findings are reported by kind with a
  masked sample, never the raw value.
* ``citation_coverage``: share of answer sentences that carry a valid evidence citation.
* ``run_checks``: deterministic rule checks on one search (no LLM call), a compliance score and the
  policy violations; recorded on the trace and in Prometheus.
* ``SearchJudge``: an LLM judge (the agent model, ``prompts/judge.txt``) scoring every search in the
  background: faithfulness, unsupported claims (hallucinations), answer relevance, completeness,
  context relevance and plan adherence. Scores are stored, sent to Phoenix as span annotations and
  exported as metrics.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from ..common.ids import utcnow
from .telemetry import get_metrics, span

log = logging.getLogger(__name__)

CITATION_COVERAGE_MIN = 0.8   # share of answer sentences with a valid citation
RELEVANCE_MIN = 0.5           # best retrieval similarity expected for a well-supported answer
JUDGE_METRICS = ("faithfulness", "answer_relevance", "completeness", "context_relevance", "plan_adherence")

# -- sensitive data --------------------------------------------------------------------------------
_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<![\w-])\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]\d{3,4}[ .-]\d{3,4}(?![\w-])"),
    "card_number": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
    "us_ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "secret": re.compile(r"\b(?:AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,})\b"
                         r"|(?i:\b(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*[^\s\"']{8,})"),
}


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total, alt = total + n, not alt
    return total % 10 == 0


def _iban_ok(value: str) -> bool:
    s = value.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


def _mask(value: str) -> str:
    v = value.strip()
    return v[:2] + "…" + v[-2:] if len(v) > 6 else "…"


def scan_sensitive(text: str | None) -> list[dict[str, Any]]:
    """PII/secret findings in ``text``: ``[{"kind", "count", "sample"}]`` (samples are masked)."""
    if not text:
        return []
    found: dict[str, list[str]] = {}
    for kind, pattern in _PATTERNS.items():
        for m in pattern.finditer(text):
            value = m.group(0)
            if kind == "card_number":
                digits = re.sub(r"\D", "", value)
                if not (13 <= len(digits) <= 19 and _luhn(digits)):
                    continue
            if kind == "iban" and not _iban_ok(value):
                continue
            if kind == "phone" and len(re.sub(r"\D", "", value)) < 9:
                continue
            found.setdefault(kind, []).append(value)
    return [{"kind": k, "count": len(v), "sample": _mask(v[0])} for k, v in found.items()]


# -- grounding -------------------------------------------------------------------------------------
_CITE = re.compile(r"\[([A-Z]{1,2}\d{1,3})\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def citation_coverage(answer: str | None, valid_ids) -> float | None:
    """Share of the answer's factual sentences (>= 25 characters, not headings) with a valid citation."""
    if not answer:
        return None
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer)]
    factual = [s for s in sentences if len(s) >= 25 and not s.endswith(":") and not s.startswith("#")]
    if not factual:
        return None
    valid = set(valid_ids)
    cited = sum(1 for s in factual if any(i in valid for i in _CITE.findall(s)))
    return round(cited / len(factual), 3)


# -- rule checks -----------------------------------------------------------------------------------
def run_checks(facts: dict[str, Any]) -> dict[str, Any]:
    """Deterministic checks on one search. ``facts`` is built by the harness (see ``SearchHarness``)."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool | None, value: Any = None, detail: str | None = None) -> None:
        if passed is None:
            return
        checks.append({"name": name, "passed": bool(passed), "value": value, "detail": detail})

    check("plan_valid", facts["plan_fallback"] is None, None, facts["plan_fallback"])
    runs = facts["task_runs"]
    planned = [r for r in runs if r["status"] != "failed"]
    searched = sum(1 for r in planned if r["planned_searched"])
    check("planned_queries_searched", searched == len(planned) if planned else None,
          f"{searched}/{len(planned)}",
          ", ".join(r["id"] for r in planned if not r["planned_searched"]) or None)
    replies = facts["executor_replies"]
    invalid = facts["invalid_replies"]
    check("valid_executor_replies", invalid == 0 if replies else None,
          round(1 - invalid / replies, 3) if replies else None, f"{invalid} invalid of {replies}" if invalid else None)
    capped = [r["id"] for r in runs if r["status"] == "max_iterations"]
    check("iteration_cap_not_hit", not capped if runs else None, len(capped), ", ".join(capped) or None)
    latest = facts["latest_statuses"]
    answered = sum(1 for s in latest.values() if s == "answered")
    check("tasks_answered", answered == len(latest) if latest else None,
          round(answered / len(latest), 3) if latest else None,
          ", ".join(f"{k}:{v}" for k, v in latest.items() if v != "answered") or None)
    coverage = citation_coverage(facts["answer"], facts["evidence_ids"])
    check("citation_coverage", coverage >= CITATION_COVERAGE_MIN if coverage is not None else None, coverage,
          f"threshold {CITATION_COVERAGE_MIN}")
    check("no_invalid_citations", not facts["dropped_citations"], len(facts["dropped_citations"]),
          ", ".join(facts["dropped_citations"]) or None)
    check("evidence_found", facts["evidence_count"] > 0, facts["evidence_count"])
    top = facts["top_score"]
    check("retrieval_relevance", top >= RELEVANCE_MIN if top is not None else None, top, f"threshold {RELEVANCE_MIN}")
    pii = scan_sensitive(facts["answer"])
    check("no_sensitive_data_in_answer", not pii, sum(f["count"] for f in pii),
          ", ".join(f"{f['kind']}×{f['count']}" for f in pii) or None)
    check("no_flagged_passage_cited", not facts["flagged_cited"], len(facts["flagged_cited"]),
          ", ".join(facts["flagged_cited"]) or None)
    check("no_injection_in_tool_results", facts["injection_flags"] == 0, facts["injection_flags"])

    violations = []
    if facts["answer"] and coverage == 0:
        violations.append("uncited_answer")
    if facts["dropped_citations"]:
        violations.append("unknown_citations")
    if facts["flagged_cited"]:
        violations.append("flagged_passage_cited")
    if pii:
        violations.append("sensitive_data_in_answer")
    if planned and searched < len(planned):
        violations.append("planned_queries_ignored")
    passed = sum(1 for c in checks if c["passed"])
    compliance = round(passed / len(checks), 3) if checks else None

    metrics = get_metrics()
    for c in checks:
        metrics.eval_checks.add(1, {"check": c["name"], "result": "pass" if c["passed"] else "fail"})
    if compliance is not None:
        metrics.compliance.record(compliance)
    if coverage is not None:
        metrics.citation_coverage.record(coverage)
    for v in violations:
        metrics.policy_violations.add(1, {"policy": v})
    for f in pii:
        metrics.sensitive_findings.add(f["count"], {"kind": f["kind"], "where": "answer"})
    return {"checks": checks, "compliance": compliance, "citation_coverage": coverage, "violations": violations,
            "sensitive_findings": pii}


# -- LLM judge -------------------------------------------------------------------------------------
class SearchJudge:
    """Scores every agent search with the agent model, in the background."""

    def __init__(self, settings, llm_factory: Callable[[], Any] | None, db, phoenix=None, max_workers: int = 2):
        self.cfg = settings
        self._factory = llm_factory
        self._llm = None
        self.db = db
        self.phoenix = phoenix
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="judge")
        self._pending: set[Future] = set()
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """A judge model exists (the agent model)."""
        return self._factory is not None

    @property
    def enabled(self) -> bool:
        """Judge every search in the background (``observability.online_judge``)."""
        return bool(self.cfg.online_judge and self._factory is not None)

    def evaluate_now(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Judge one search synchronously (offline evaluation runs)."""
        return self._run(item)

    def _client(self):
        with self._lock:
            if self._llm is None:
                self._llm = self._factory()
            return self._llm

    def submit(self, item: dict[str, Any]) -> None:
        future = self._pool.submit(self._run, item)
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(lambda f: self._discard(f))

    def _discard(self, future: Future) -> None:
        with self._lock:
            self._pending.discard(future)

    def wait(self, timeout: float = 30) -> None:
        """Block until queued judgments finish (tests, evaluation runs, shutdown)."""
        with self._lock:
            pending = list(self._pending)
        if pending:
            wait(pending, timeout=timeout)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def prompt(item: dict[str, Any]) -> str:
        intents = "\n".join(f"- {i['id']}: {i['intent']}" for i in item["intents"])
        tasks = "\n".join(f"- {t['id']} ({t['status']}): {t.get('answer') or 'no result'}" for t in item["tasks"])
        return (f"User question:\n{item['query']}\n\nPlanned intents:\n{intents}\n\nTask results:\n{tasks}\n\n"
                f"<evidence>\n{item['evidence'] or '(no evidence)'}\n</evidence>\n\n"
                f"Final answer:\n{item['answer'] or '(no answer)'}\n\nReturn the JSON evaluation.")

    def _run(self, item: dict[str, Any]) -> dict[str, Any] | None:
        metrics = get_metrics()
        trace_id = item["trace_id"]
        try:
            system = Path(self.cfg.judge_prompt_path).read_text(encoding="utf-8").strip()
        except OSError:
            self._finish(trace_id, "failed", f"judge prompt not readable: {self.cfg.judge_prompt_path}")
            metrics.judge_runs.add(1, {"status": "failed"})
            return None
        with span("agent.judge", "EVALUATOR", {"agent.judged_trace_id": trace_id}) as sp:
            try:
                resp = self._client().generate(self.prompt(item), system=system, purpose="agent_judge",
                                               json_output=True)
                from ..agent.plan import parse_json_object  # local import: agent depends on observability

                raw = parse_json_object(resp.text)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {str(exc)[:200]}"
                sp.set_attribute("agent.judge.error", msg)
                self._finish(trace_id, "failed", msg)
                metrics.judge_runs.add(1, {"status": "failed"})
                return None
            scores = {}
            for name in JUDGE_METRICS:
                try:
                    scores[name] = max(0.0, min(1.0, float(raw.get(name))))
                except (TypeError, ValueError):
                    continue
            unsupported = [str(c)[:300] for c in raw.get("unsupported_claims") or [] if str(c).strip()][:20]
            explanation = str(raw.get("explanation") or "")[:2000]
            for name, score in scores.items():
                sp.set_attribute(f"agent.judge.{name}", score)
            sp.set_attribute("agent.judge.unsupported_claims", len(unsupported))
        rows = [(name, score, None, explanation if name == "faithfulness" else None) for name, score in scores.items()]
        rows.append(("hallucination", 1.0 if unsupported else 0.0, "hallucinated" if unsupported else "grounded",
                     "\n".join(unsupported) or None))
        now = utcnow()
        for name, score, label, expl in rows:
            self.db.execute("INSERT OR REPLACE INTO search_evaluations(trace_id, evaluator, name, score, label, "
                            "explanation, created_at) VALUES (?,?,?,?,?,?,?)",
                            (trace_id, "llm_judge", name, score, label, expl, now))
            if self.phoenix is not None:
                self.phoenix.annotate(item.get("span_id"), name, score=score, label=label, explanation=expl,
                                      kind="LLM", metadata={"evaluator": "llm_judge"})
            if name in JUDGE_METRICS:
                metrics.judge_score.record(score, {"metric": name})
        if unsupported:
            metrics.hallucinations.add(1)
        metrics.judge_runs.add(1, {"status": "done"})
        self._finish(trace_id, "done", None)
        return {"scores": scores, "unsupported_claims": unsupported, "explanation": explanation}

    def _finish(self, trace_id: str, status: str, error: str | None) -> None:
        self.db.execute("UPDATE searches SET judge_status = ? WHERE trace_id = ?",
                        (status if not error else f"{status}: {error}"[:300], trace_id))


def evaluations_for(db, trace_id: str) -> list[dict[str, Any]]:
    return db.query("SELECT evaluator, name, score, label, explanation, created_at FROM search_evaluations "
                    "WHERE trace_id = ? ORDER BY evaluator, name", (trace_id,))

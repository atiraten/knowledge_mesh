"""Laya decision-model reranking of retrieved document passages.

Exactly two Laya calls per query:

1. **description** (``choice``): state ``{"question": q}``; one option per top-k passage
   (``d0..dN``). Laya returns a probability per passage; the probabilities sum to 1, so at
   most one passage can clear a threshold above 0.5.
2. **quality** (``score``): for the passages whose probability is over ``min_probability``
   (best ``top_n`` of them), state ``{"question": q, "chosen_response": passage}``, evaluated
   for all shortlisted passages in one batched forward pass (``predict_batch``).

A passage rated ``min_quality`` (default "Acceptable") or better by its most likely quality
level is eligible; the best eligible one (highest level, then highest probability) is the
response. If none qualifies the response is ``no_answer_message``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

from ..common.errors import NotConfigured
from ..observability import capture_sensitive, get_metrics, span

log = logging.getLogger(__name__)


class LayaReranker:
    def __init__(self, settings, agent_factory: Callable[[], Any] | None = None):
        self.cfg = settings
        self._factory = agent_factory
        self._agent = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()  # one forward pass at a time on the shared model

    def available(self) -> tuple[bool, str | None]:
        if self._factory is not None or self._agent is not None:
            return True, None
        try:
            import laya  # noqa: F401
        except ImportError:
            return False, "laya is not installed (pip install -e '.[laya]')"
        return True, None

    def agent(self):
        if self._agent is None:
            with self._load_lock:
                if self._agent is None:
                    if self._factory is not None:
                        self._agent = self._factory()
                    else:
                        try:
                            import laya
                        except ImportError as exc:
                            raise NotConfigured("laya is not installed (pip install -e '.[laya]')") from exc
                        start = time.perf_counter()
                        self._agent = laya.load(self.cfg.model, subfolder=self.cfg.subfolder, device=self.cfg.device)
                        log.info("Laya model loaded", extra={"model": self.cfg.model,
                                                             "seconds": round(time.perf_counter() - start, 1)})
        return self._agent

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """``candidates`` are document results (rank order); returns the decision record."""
        cfg = self.cfg
        metrics = get_metrics()
        start = time.perf_counter()
        keys = [f"d{i}" for i in range(len(candidates))]
        texts = {k: c["text"][: cfg.max_candidate_chars] for k, c in zip(keys, candidates)}
        decision: dict[str, Any] = {"applied": True, "provider": "laya", "model": cfg.model,
                                    "candidates": len(candidates), "min_probability": cfg.min_probability,
                                    "top_n": cfg.top_n, "min_quality": cfg.min_quality, "calls": 0, "decisions": []}
        with span("rerank.laya", "RERANKER", {"reranker.model_name": cfg.model, "reranker.query": query[:500],
                                              "reranker.top_k": len(candidates)}) as sp:
            for i, (k, c) in enumerate(zip(keys, candidates)):
                sp.set_attribute(f"reranker.input_documents.{i}.document.id", c["chunk_id"])
                sp.set_attribute(f"reranker.input_documents.{i}.document.score", c["score"])
                if capture_sensitive():
                    sp.set_attribute(f"reranker.input_documents.{i}.document.content", texts[k])
            if not candidates:
                return self._finish(decision, None, start, sp, metrics)
            agent = self.agent()
            # Call 1: which response best answers the question?
            choice_q = {"description": {"type": "choice", "instructions": cfg.choice_instructions, "criteria": texts}}
            with span("laya.choice", "LLM", {"llm.model_name": cfg.model,
                                             "input.value": _call_input(query, cfg.choice_instructions, texts)}) as call:
                with self._predict_lock:
                    out1 = agent.predict({"question": query}, choice_q, head_max_len=cfg.choice_head_max_len)
                decision["calls"] = 1
                probs = {k: float(v) for k, v in out1["answers"]["description"]["probabilities"].items()}
                decision["choice_usage"] = _usage(out1)
                call.set_attribute("output.value", json.dumps({"probabilities": probs,
                                                               "usage": decision["choice_usage"]}))
            ranked = sorted(keys, key=lambda k: -probs.get(k, 0.0))
            shortlist = [k for k in ranked if probs.get(k, 0.0) > cfg.min_probability][: cfg.top_n]
            records = {k: {"chunk_id": candidates[int(k[1:])]["chunk_id"], "retrieval_rank": int(k[1:]) + 1,
                           "choice_probability": round(probs.get(k, 0.0), 4),
                           "passed_probability": k in shortlist} for k in ranked}
            # Call 2: is each shortlisted response correct?  (one batched forward pass)
            if shortlist:
                quality_q = {"quality": {"type": "score", "instructions": cfg.quality_instructions,
                                         "criteria": list(cfg.quality_levels)}}
                states = [{"question": query, "chosen_response": texts[k]} for k in shortlist]
                with span("laya.quality", "LLM", {
                    "llm.model_name": cfg.model,
                    "input.value": _call_input(query, cfg.quality_instructions, {k: texts[k] for k in shortlist}),
                }) as call:
                    with self._predict_lock:
                        out2 = agent.predict_batch(states, quality_q)
                    decision["calls"] = 2
                    min_level = cfg.quality_levels.index(cfg.min_quality)
                    for k, res in zip(shortlist, out2):
                        ans = res["answers"]["quality"]
                        level_probs = {int(lv): float(p) for lv, p in ans["probabilities"].items()}  # keys "0".."4"
                        level = max(level_probs, key=level_probs.get)  # most likely level
                        records[k].update(quality=cfg.quality_levels[level], quality_level=level,
                                          quality_probability=round(level_probs[level], 4),
                                          quality_score=round(float(ans.get("score", level)), 3),
                                          acceptable=level >= min_level)
                    call.set_attribute("output.value", json.dumps({k: records[k] for k in shortlist}))
            eligible = [k for k in shortlist if records[k].get("acceptable")]
            best = max(eligible, key=lambda k: (records[k]["quality_level"], probs[k])) if eligible else None
            for k in ranked:
                records[k]["selected"] = k == best
            decision["decisions"] = [records[k] for k in ranked]
            return self._finish(decision, best and int(best[1:]), start, sp, metrics)

    def _finish(self, decision: dict[str, Any], index: int | None, start: float, sp, metrics) -> dict[str, Any]:
        decision["selected_index"] = index
        decision["outcome"] = "selected" if index is not None else "no_response"
        if index is None:
            decision["message"] = self.cfg.no_answer_message
        decision["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        sp.set_attribute("reranker.calls", decision["calls"])
        sp.set_attribute("reranker.outcome", decision["outcome"])
        for i, d in enumerate(decision["decisions"]):
            sp.set_attribute(f"reranker.output_documents.{i}.document.id", d["chunk_id"])
            sp.set_attribute(f"reranker.output_documents.{i}.document.score", d["choice_probability"])
        sp.set_attribute("output.value", json.dumps(decision, default=str)[:20000])
        metrics.stage_duration.record(time.perf_counter() - start, {"stage": "rerank"})
        metrics.rerank_outcomes.add(1, {"outcome": decision["outcome"]})
        return decision


def _call_input(query: str, instructions: str, passages: dict[str, str]) -> str:
    """Span input for one Laya call; passage text only with ``capture_sensitive_content``."""
    shown = passages if capture_sensitive() else {k: f"<{len(v)} chars redacted>" for k, v in passages.items()}
    return json.dumps({"question": query[:500], "instructions": instructions, "passages": shown})[:20000]


def _usage(out: dict[str, Any]) -> dict[str, Any]:
    u = out.get("usage") or {}
    return {k: u[k] for k in ("input_tokens", "state_tokens", "truncated", "truncated_questions") if k in u}

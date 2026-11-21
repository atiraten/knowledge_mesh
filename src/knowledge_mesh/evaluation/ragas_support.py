"""Ragas evaluation with the project's own models (no OpenAI key needed).

* Evaluator LLM: the configured agent model (or the answer-generation model), wrapped for Ragas through
  LangChain: ``google`` -> ``ChatGoogleGenerativeAI``, ``openai`` -> ``ChatOpenAI`` (any OpenAI-compatible
  endpoint via ``base_url``). The API key comes from the same env var / key file as the app.
* Embeddings: the mesh's local embedding model (all-mpnet-base-v2 by default), wrapped for Ragas.
* Metrics: faithfulness (claims supported by the retrieved contexts) and response relevancy.

Requires the ``eval`` extra: ``pip install -e ".[eval]"``.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

from ..common.errors import NotConfigured
from ..llm.client import resolve_api_key

_BLOCK = re.compile(r"\n\n(?=\[[A-Z]{1,2}\d{1,3}\])")


def evidence_contexts(evidence_text: str | None) -> list[str]:
    """Split the synthesis evidence (``[D1] ...\\n\\n[T1] ...``) into one context per evidence block."""
    blocks = [b.strip() for b in _BLOCK.split(evidence_text or "") if b.strip()]
    return blocks or [""]


def ragas_models(mesh) -> tuple[Any, Any, dict[str, Any]]:
    """Ragas evaluator LLM and embeddings built from config/mesh.yaml."""
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore

    cfg = mesh.config.agent if mesh.config.agent.configured else mesh.config.answer_generation
    if not cfg.configured:
        raise NotConfigured("no evaluator model: configure agent (or answer_generation) provider and model")
    key = resolve_api_key(cfg)
    if cfg.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        chat = ChatGoogleGenerativeAI(model=cfg.model, google_api_key=key, temperature=0,
                                      timeout=cfg.timeout_seconds, max_retries=cfg.max_retries)
    elif cfg.provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore

        chat = ChatOpenAI(model=cfg.model, api_key=key, base_url=cfg.base_url, temperature=0,
                          timeout=cfg.timeout_seconds, max_retries=cfg.max_retries)
    else:
        raise NotConfigured(f"Ragas cannot use provider {cfg.provider!r}")
    with warnings.catch_warnings():  # the wrappers are deprecated aliases in Ragas 0.4 but still supported
        warnings.simplefilter("ignore", DeprecationWarning)
        # bypass_n: Gemini rejects several candidates per call ("Multiple candidates is not enabled"), which
        # response relevancy asks for; Ragas then makes one call per candidate instead.
        llm = LangchainLLMWrapper(chat, bypass_n=True)
        embeddings = LlamaIndexEmbeddingsWrapper(mesh.embed_model)
    return llm, embeddings, {"llm": f"{cfg.provider}/{cfg.model}", "embeddings": mesh.config.embedding.model}


def run_ragas(mesh, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """``rows``: ``{"user_input", "response", "retrieved_contexts": [..], "reference"}``. Never raises."""
    if not rows:
        return {"status": "skipped", "reason": "nothing to evaluate"}
    try:
        from ragas import EvaluationDataset, evaluate  # type: ignore
        from ragas.metrics import Faithfulness, ResponseRelevancy  # type: ignore
    except ImportError as exc:
        return {"status": "skipped", "reason": f"ragas not installed ({exc.name}); pip install -e '.[eval]'"}
    try:
        llm, embeddings, evaluator = ragas_models(mesh)
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {getattr(exc, 'message', str(exc))[:300]}"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = evaluate(EvaluationDataset.from_list(rows), metrics=[Faithfulness(), ResponseRelevancy()],
                              llm=llm, embeddings=embeddings, show_progress=False, raise_exceptions=False)
        frame = result.to_pandas()
        scores: dict[str, float | None] = {}
        for name in ("faithfulness", "answer_relevancy", "response_relevancy"):
            if name in frame.columns:
                value = float(frame[name].mean())
                scores[name] = None if math.isnan(value) else round(value, 4)
        per_row = [{k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in row.items() if k in scores} for row in frame.to_dict("records")]
        status = "ok" if any(v is not None for v in scores.values()) else "failed"
        return {"status": status, "scores": scores, "per_question": per_row, "evaluator": evaluator,
                **({} if status == "ok" else {"reason": "every Ragas metric failed; see per_question"})}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:300]}", "evaluator": evaluator}

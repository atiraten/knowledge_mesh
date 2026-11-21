"""Local embedding models for LlamaIndex.

Production uses ``sentence-transformers/all-mpnet-base-v2`` executed locally (no embedding API).
``HashingEmbedding`` is a deterministic, dependency-free stand-in used by the automated tests.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

log = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    device: str = "cpu"
    normalize: bool = True
    revision: str | None = None
    local_files_only: bool = False

    _model: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @classmethod
    def class_name(cls) -> str:
        return "MeshSentenceTransformerEmbedding"

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("loading local embedding model", extra={"model": self.model_name, "device": self.device})
                    self._model = SentenceTransformer(
                        self.model_name, device=self.device, revision=self.revision,
                        local_files_only=self.local_files_only,
                    )
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts, batch_size=self.embed_batch_size, normalize_embeddings=self.normalize,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._encode([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def info(self) -> dict[str, Any]:
        model = self._load()
        dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else \
            model.get_sentence_embedding_dimension()
        return {
            "provider": "sentence_transformers",
            "model": self.model_name,
            "revision": self.revision or _cached_revision(self.model_name),
            "dimension": dim,
            "normalize": self.normalize,
        }


def _cached_revision(model_name: str) -> str | None:
    """Commit hash of the locally cached model snapshot, when available."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_name, "config.json")
        if isinstance(path, str) and "/snapshots/" in path:
            return path.split("/snapshots/")[1].split("/")[0]
    except Exception:
        return None
    return None


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedding(BaseEmbedding):
    """Signed feature-hashing bag of words (with crude stemming). Test-only."""

    dim: int = 4096  # wide enough that signed-hash collisions rarely cancel real overlap
    normalize: bool = True

    @classmethod
    def class_name(cls) -> str:
        return "MeshHashingEmbedding"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            token = _stem(token)
            digest = hashlib.md5(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec] if self.normalize else vec

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def info(self) -> dict[str, Any]:
        return {"provider": "hashing", "model": self.model_name, "revision": None,
                "dimension": self.dim, "normalize": self.normalize}


def _stem(token: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def build_embedding(settings) -> BaseEmbedding:
    if settings.provider == "hashing":
        return HashingEmbedding(model_name="test-hashing-embedding", embed_batch_size=settings.batch_size,
                                normalize=settings.normalize)
    return SentenceTransformerEmbedding(
        model_name=settings.model, embed_batch_size=settings.batch_size, device=settings.device,
        normalize=settings.normalize, revision=settings.revision, local_files_only=settings.local_files_only,
    )

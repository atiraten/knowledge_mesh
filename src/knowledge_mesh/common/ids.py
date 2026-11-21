"""Stable identifiers.

Identifiers are derived from source identity (not content), so a changed source keeps its
ID and replaces its stale records, while re-ingesting unchanged content is a no-op.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import datetime, timezone

_SEP = "\x1f"


def stable_hash(*parts: object, length: int = 16) -> str:
    joined = _SEP.join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id_for(source_uri: str) -> str:
    return f"doc-{stable_hash('document', source_uri)}"


def chunk_id_for(document_id: str, content_hash: str, level: int, ordinal: int, leaf: bool) -> str:
    prefix = "chunk" if leaf else "node"
    return f"{prefix}-{stable_hash(document_id, content_hash, level, ordinal)}"


def dataset_id_for(source_id: str, schema: str | None, name: str) -> str:
    return f"ds-{stable_hash('dataset', source_id, schema or '', name)}"


def column_id_for(dataset_id: str, column_name: str) -> str:
    return f"col-{stable_hash('column', dataset_id, column_name)}"


def key_id_for(dataset_id: str, key_type: str, origin: str, columns: list[str],
               ref_dataset_id: str | None = None, ref_columns: list[str] | None = None) -> str:
    return "key-" + stable_hash(
        dataset_id, key_type, origin, ",".join(columns), ref_dataset_id or "", ",".join(ref_columns or [])
    )


def catalog_ref(source_id: str, schema: str | None, dataset: str, column: str | None = None) -> str:
    parts = [source_id] + ([schema] if schema else []) + [dataset] + ([column] if column else [])
    return "catalog:" + "/".join(parts)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def slugify(label: str, max_length: int = 96) -> str:
    text = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    if not text:
        text = stable_hash(label, length=12)
    return text[:max_length].rstrip("-")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

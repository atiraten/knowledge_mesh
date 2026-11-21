"""Column profiling.

For each column: rows evaluated, distinct values, empty/blank values, nulls, data format
patterns (digits -> ``N``, letters -> ``T``, other characters kept: ``1999-12-22`` ->
``NNNN-NN-NN``, ``PAUL MERSON`` -> ``TTTT TTTTTT``), inferred content type, and the most
common values. Profiles are stored in SQLite only; raw values never go to retrieval,
prompts or logs.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

_CONTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")),
    ("url", re.compile(r"^https?://\S+$", re.I)),
    ("uuid", re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)),
    ("ip_address", re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")),
    ("datetime", re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+\-]\d{2}:?\d{2})?$")),
    ("date", re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2})$")),
    ("time", re.compile(r"^\d{2}:\d{2}(:\d{2})?$")),
    ("boolean", re.compile(r"^(true|false|yes|no|y|n|t|f)$", re.I)),
    ("currency", re.compile(r"^[$€£¥]\s?-?\d[\d,]*(\.\d+)?$")),
    ("integer", re.compile(r"^[+\-]?\d+$")),
    ("decimal", re.compile(r"^[+\-]?\d*[.,]\d+$")),
    ("phone", re.compile(r"^\+?\d[\d\s().\-]{6,19}$")),
    ("postal_code", re.compile(r"^(\d{5}(-\d{4})?|[A-Z]\d[A-Z] ?\d[A-Z]\d|[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2})$", re.I)),
    ("code", re.compile(r"^[A-Z0-9_\-]{1,12}$")),
]


def format_pattern(value: str, max_len: int = 64) -> str:
    out = []
    for ch in value[:max_len]:
        if ch.isdigit():
            out.append("N")
        elif ch.isalpha():
            out.append("T")
        else:
            out.append(ch)
    if len(value) > max_len:
        out.append("…")
    return "".join(out)


def type_category(arrow_type: pa.DataType) -> str:
    if isinstance(arrow_type, pa.BaseExtensionType):
        # Database UUID columns arrive as the arrow.uuid extension type (storage: 16 raw bytes).
        if arrow_type.extension_name == "arrow.uuid":
            return "uuid"
        return type_category(arrow_type.storage_type)
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_integer(arrow_type):
        return "integer"
    if pa.types.is_floating(arrow_type):
        return "float"
    if pa.types.is_decimal(arrow_type):
        return "decimal"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "datetime"
    if pa.types.is_time(arrow_type):
        return "time"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "binary"
    if pa.types.is_dictionary(arrow_type):
        return type_category(arrow_type.value_type)
    return "other"


def infer_content_type(distinct_values: list[str], category: str, threshold: float) -> str:
    if category in {"integer", "decimal", "date", "datetime", "time"}:
        return category
    if category == "float":
        return "decimal"
    if category == "bool":
        return "boolean"
    if category == "binary":
        return "binary"
    values = [v.strip() for v in distinct_values if v is not None and v.strip()]
    if not values:
        return "empty"
    for name, pattern in _CONTENT_PATTERNS:
        matched = sum(1 for v in values if pattern.match(v))
        if matched / len(values) >= threshold:
            return name
    avg_len = sum(len(v) for v in values) / len(values)
    return "free_text" if avg_len > 60 else "text"


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def profile_column(arr: pa.ChunkedArray | pa.Array, settings) -> dict[str, Any]:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    category = type_category(arr.type)
    n = len(arr)
    null_count = arr.null_count
    try:
        vc = pc.value_counts(arr.drop_null())
        values = vc.field("values").to_pylist()
        counts = vc.field("counts").to_pylist()
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        c = Counter(_as_str(v) for v in arr.to_pylist() if v is not None)
        values, counts = list(c.keys()), list(c.values())
    pairs = sorted(zip(values, counts), key=lambda t: (-t[1], _as_str(t[0])))
    distinct = len(pairs)
    empty = 0
    patterns: Counter[str] = Counter()
    str_values = []
    for value, count in pairs:
        s = _as_str(value)
        if category == "string" and not s.strip():
            empty += count
        patterns[format_pattern(s, settings.max_pattern_length)] += count
        if len(str_values) < 2000:
            str_values.append(s)
    mcv = []
    if settings.store_most_common_values and settings.most_common_values:
        mcv = [{"value": _as_str(v)[:200], "count": c} for v, c in pairs[: settings.most_common_values]]
    return {
        "row_count": n,
        "distinct_count": distinct,
        "empty_count": empty,
        "null_count": null_count,
        "format_patterns": [{"pattern": p, "count": c} for p, c in patterns.most_common(settings.format_patterns)],
        "content_type": infer_content_type(str_values, category, settings.content_type_threshold),
        "most_common_values": mcv,
        "category": category,
    }

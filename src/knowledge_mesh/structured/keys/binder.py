"""BINDER-style inclusion dependency (IND) discovery for foreign-key candidates.

BINDER (Papenbrock et al., PVLDB 2015) *divides* every attribute's distinct values into
hash buckets and *conquers* bucket by bucket: a candidate ``A ⊆ B`` is refuted as soon as
one bucket of A is not contained in the same bucket of B, so most invalid candidates die
after looking at a fraction of the data. N-ary candidates are generated Apriori-style from
valid lower-arity INDs (every (n-1)-ary projection must hold) and validated the same way on
value tuples.

Here buckets are kept in memory (raw values never leave the profiling process; nothing is
spilled to disk). Only INDs whose referenced side is a declared or observed unique key are
reported as foreign-key candidates, preserving the ordered column mapping. Rows with a null
in any dependent column are ignored (SQL foreign-key semantics). An optional ``eligible``
predicate restricts both sides to admitted columns (key discovery admits identifiers only).
"""

from __future__ import annotations

import datetime as _dt
import decimal
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

Attr = tuple[str, str]  # (dataset_id, column)


def normalize_value(v: Any) -> str | None:
    """Type-normalised string form so equal values compare equal across sources."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return None
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, decimal.Decimal):
        return str(int(v)) if v == v.to_integral_value() else str(v.normalize())
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return v.isoformat()
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, uuid.UUID):
        return str(v)  # canonical lower-case form, as UUIDs stored as text usually are
    return str(v)


@dataclass
class DatasetValues:
    dataset_id: str
    columns: dict[str, list[str | None]]  # normalised values per row
    categories: dict[str, str]
    distinct: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, values in self.columns.items():
            self.distinct[name] = {v for v in values if v is not None}

    def tuples(self, cols: tuple[str, ...]) -> set[tuple[str, ...]]:
        series = [self.columns[c] for c in cols]
        return {t for t in zip(*series) if all(v is not None for v in t)}


@dataclass(frozen=True)
class IND:
    dep_dataset: str
    dep_columns: tuple[str, ...]
    ref_dataset: str
    ref_columns: tuple[str, ...]


_COMPATIBLE = {
    "integer": "numeric", "decimal": "numeric", "float": "numeric",
    "string": "string", "uuid": "string",  # a UUID column can reference UUIDs stored as text
    "date": "temporal", "datetime": "temporal", "time": "temporal",
}


class Binder:
    def __init__(self, n_buckets: int = 16, allow_cross_type: bool = False, min_distinct_dependent: int = 2,
                 max_nary_candidates: int = 10_000):
        self.n = max(1, n_buckets)
        self.allow_cross_type = allow_cross_type
        self.min_distinct = min_distinct_dependent
        self.max_nary = max_nary_candidates
        self.stats: dict[str, int] = {"unary_candidates": 0, "unary_valid": 0, "nary_candidates": 0, "nary_valid": 0,
                                      "excluded_columns": 0}

    def _bucket(self, values: Iterable) -> list[set]:
        buckets: list[set] = [set() for _ in range(self.n)]
        for v in values:
            buckets[hash(v) % self.n].add(v)
        return buckets

    def _validate(self, candidates: list[tuple[Any, Any]], value_sets: dict[Any, set]) -> set[tuple[Any, Any]]:
        """Divide & conquer containment check of dep ⊆ ref for every candidate pair."""
        needed = {d for d, _ in candidates} | {r for _, r in candidates}
        buckets = {key: self._bucket(value_sets[key]) for key in needed}
        alive = set(candidates)
        for i in range(self.n):
            for cand in list(alive):
                dep, ref = cand
                if not buckets[dep][i] <= buckets[ref][i]:
                    alive.discard(cand)
            if not alive:
                break
        return alive

    def _compatible(self, a: str, b: str) -> bool:
        ca, cb = _COMPATIBLE.get(a), _COMPATIBLE.get(b)
        if ca is None or cb is None:
            return False
        return ca == cb or self.allow_cross_type

    def discover(self, datasets: dict[str, DatasetValues],
                 unique_keys: dict[str, list[tuple[str, ...]]], max_arity: int,
                 eligible: Callable[[Attr], bool] | None = None) -> list[IND]:
        """Return IND-based FK candidates referencing the given unique keys (per dataset).

        ``eligible`` (optional) admits a column on either side of a candidate; key discovery passes
        the identifier rule (see ``fk_rules``). A composite candidate needs every column admitted.
        """
        ok = eligible or (lambda _attr: True)
        all_attrs = {(ds.dataset_id, c) for ds in datasets.values() for c in ds.columns}
        self.stats["excluded_columns"] = sum(1 for a in all_attrs if not ok(a))
        ref_columns: set[Attr] = {(ds, c) for ds, keys in unique_keys.items() if ds in datasets
                                  for key in keys for c in key if ok((ds, c))}
        dep_columns: list[Attr] = [
            (ds.dataset_id, c) for ds in datasets.values() for c in ds.columns
            if _COMPATIBLE.get(ds.categories.get(c, "")) and len(ds.distinct[c]) >= self.min_distinct
            and ok((ds.dataset_id, c))
        ]
        candidates: list[tuple[Attr, Attr]] = []
        for dep in dep_columns:
            dvals = datasets[dep[0]]
            for ref in ref_columns:
                rvals = datasets[ref[0]]
                if dep == ref:
                    continue
                if not self._compatible(dvals.categories[dep[1]], rvals.categories.get(ref[1], "")):
                    continue
                if len(dvals.distinct[dep[1]]) > len(rvals.distinct[ref[1]]):
                    continue  # cardinality pruning: |A| > |B| cannot be contained
                candidates.append((dep, ref))
        self.stats["unary_candidates"] = len(candidates)
        value_sets = {a: datasets[a[0]].distinct[a[1]] for pair in candidates for a in pair}
        valid_unary = self._validate(candidates, value_sets) if candidates else set()
        self.stats["unary_valid"] = len(valid_unary)

        results: list[IND] = []
        for ds, keys in unique_keys.items():
            for key in keys:
                if len(key) == 1:
                    for dep, ref in sorted(valid_unary):
                        if ref == (ds, key[0]):
                            results.append(IND(dep[0], (dep[1],), ds, key))

        # n-ary: apriori-style extension over the referenced composite key's column order
        for ref_ds, keys in unique_keys.items():
            for key in keys:
                if not 2 <= len(key) <= max_arity or ref_ds not in datasets:
                    continue
                for dep_ds in datasets:
                    per_pos = [[dep[1] for dep, ref in valid_unary if dep[0] == dep_ds and ref == (ref_ds, k)]
                               for k in key]
                    if any(not p for p in per_pos):
                        continue
                    partials: list[tuple[str, ...]] = [(c,) for c in sorted(per_pos[0])]
                    for pos in range(1, len(key)):
                        extended = [p + (c,) for p in partials for c in sorted(per_pos[pos]) if c not in p]
                        if dep_ds == ref_ds:
                            extended = [e for e in extended if e != key[: pos + 1]]
                        if len(extended) > self.max_nary:
                            extended = extended[: self.max_nary]
                        self.stats["nary_candidates"] += len(extended)
                        prefix = key[: pos + 1]
                        dep_sets = {e: datasets[dep_ds].tuples(e) for e in extended}
                        ref_set = datasets[ref_ds].tuples(prefix)
                        sets: dict[Any, set] = {("dep", e): s for e, s in dep_sets.items()}
                        sets[("ref", prefix)] = ref_set
                        cands = [(("dep", e), ("ref", prefix)) for e in extended if dep_sets[e]]
                        alive = self._validate(cands, sets) if cands else set()
                        partials = [c[0][1] for c in cands if c in alive]
                        if not partials:
                            break
                    for p in partials:
                        if len(p) == len(key):
                            self.stats["nary_valid"] += 1
                            results.append(IND(dep_ds, p, ref_ds, key))
        return results

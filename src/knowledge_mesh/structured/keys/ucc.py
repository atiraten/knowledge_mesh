"""Minimal unique column combination (UCC) discovery: HyUCC and Apriori-style lattice search.

Data model: each column is dictionary-encoded into integer cluster IDs, with ``-1`` for a
value that occurs exactly once in that column (a "stripped partition" in PLI terms). A row
holding ``-1`` in any column of a combination is necessarily unique for that combination.

**HyUCC** (Papenbrock & Naumann, 2017) alternates two phases until the result is exact:

1. *Sampling*: compare record pairs that share a PLI cluster (sorted-neighbourhood
   windows). Each pair's *agree set* is a non-UCC; its complement (difference set) must be
   hit by every UCC.
2. *Induction*: the candidate positive cover is the set of minimal hitting sets
   (transversals) of all difference sets seen so far, bounded by ``max_size``.
3. *Validation*: candidates are validated in Apriori order (smallest first) with vectorised
   group-by checks. Invalid candidates yield violating record pairs, whose difference sets
   refine the cover. When every candidate validates, the cover is exactly the set of minimal
   UCCs up to ``max_size``.

**Apriori** (Agrawal & Srikant, 1994 style candidate generation) walks the lattice level by
level: unique sets are reported; non-unique ``k``-sets are joined on a shared ``k-1``
prefix and a candidate survives only if all its ``k``-subsets are non-unique (downward
closure / minimality pruning). It is used for small inputs and as a cross-check.

Null semantics: with ``null_equals_null`` (default) nulls form one shared value; otherwise
every null is distinct (SQL UNIQUE semantics). Primary-key eligibility additionally requires
no nulls in the evaluated scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def encode_table(table: pa.Table, null_equals_null: bool = True) -> np.ndarray:
    """Return an (n_rows, n_cols) int64 matrix of cluster IDs (-1 = singleton/unique value)."""
    n = table.num_rows
    out = np.empty((n, table.num_columns), dtype=np.int64)
    for j, col in enumerate(table.columns):
        out[:, j] = encode_column(col, null_equals_null)
    return out


def encode_column(col: pa.ChunkedArray | pa.Array, null_equals_null: bool = True) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    try:
        encoded = pc.dictionary_encode(col)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid, pa.ArrowTypeError):
        encoded = pc.dictionary_encode(pa.array([None if v is None else str(v) for v in col.to_pylist()],
                                                type=pa.string()))
    codes = encoded.indices.fill_null(-2).to_numpy(zero_copy_only=False).astype(np.int64)
    nulls = codes == -2
    if null_equals_null and nulls.any():
        codes[nulls] = codes.max() + 1 if (~nulls).any() else 0
    valid = codes >= 0
    if valid.any():
        counts = np.bincount(codes[valid])
        singleton = np.zeros(n, dtype=bool)
        singleton[valid] = counts[codes[valid]] == 1
        codes[singleton] = -1
    codes[codes < 0] = -1
    return codes


def _bits(mask: int) -> list[int]:
    out, i = [], 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return out


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


@dataclass
class UCCResult:
    uccs: list[tuple[int, ...]]
    complete: bool = True
    stats: dict = field(default_factory=dict)


class _Checker:
    def __init__(self, matrix: np.ndarray, columns: list[int]):
        self.m = matrix[:, columns] if len(columns) else np.empty((matrix.shape[0], 0), dtype=np.int64)
        self.n = matrix.shape[0]
        self.k = len(columns)
        self.full = (1 << self.k) - 1
        self.validations = 0
        self._radix = [int(self.m[:, j].max()) + 2 if self.n else 1 for j in range(self.k)]

    def check(self, mask: int, max_pairs: int = 256) -> tuple[bool, list[tuple[int, int]]]:
        """Is the combination unique? If not, return some violating row pairs."""
        self.validations += 1
        cols = _bits(mask)
        if not cols:
            return (self.n <= 1), ([(0, 1)] if self.n > 1 else [])
        sub = self.m[:, cols]
        keep = np.nonzero((sub != -1).all(axis=1))[0]
        if keep.size < 2:
            return True, []
        sub = sub[keep]
        product = 1
        for c in cols:
            product *= self._radix[c]
        if product < 2 ** 62:
            key = np.zeros(sub.shape[0], dtype=np.int64)
            for j, c in enumerate(cols):
                key = key * self._radix[c] + sub[:, j]
            order = np.argsort(key, kind="stable")
            sk = key[order]
            dup = np.nonzero(sk[1:] == sk[:-1])[0]
        else:
            order = np.lexsort(sub.T[::-1])
            ss = sub[order]
            dup = np.nonzero((ss[1:] == ss[:-1]).all(axis=1))[0]
        if dup.size == 0:
            return True, []
        pairs = [(int(keep[order[i]]), int(keep[order[i + 1]])) for i in dup[:max_pairs]]
        return False, pairs

    def difference_sets(self, pairs: list[tuple[int, int]]) -> set[int]:
        if not pairs or self.k == 0:
            return {0} if pairs else set()
        a = self.m[[p[0] for p in pairs]]
        b = self.m[[p[1] for p in pairs]]
        agree = (a == b) & (a != -1)
        weights = 1 << np.arange(self.k, dtype=object)
        out = set()
        for row in agree:
            agree_mask = int(sum(weights[row])) if row.any() else 0
            out.add(self.full & ~agree_mask)
        return out


def _update_transversals(cover: set[int], diff: int, max_size: int) -> set[int]:
    keep = {t for t in cover if t & diff}
    extend: set[int] = set()
    bits = [1 << b for b in _bits(diff)]
    for t in cover:
        if t & diff:
            continue
        for b in bits:
            nt = t | b
            if _popcount(nt) <= max_size:
                extend.add(nt)
    result = set(keep)
    for nt in sorted(extend, key=lambda x: (_popcount(x), x)):
        if not any((s & nt) == s for s in result):
            result.add(nt)
    return result


def _sample_pairs(m: np.ndarray, windows: int = 3, max_pairs: int = 20000) -> list[tuple[int, int]]:
    n, k = m.shape
    pairs: list[tuple[int, int]] = []
    for j in range(k):
        col = m[:, j]
        rows = np.nonzero(col != -1)[0]
        if rows.size < 2:
            continue
        nxt = m[rows, (j + 1) % k] if k > 1 else np.zeros(rows.size, dtype=np.int64)
        order = np.lexsort((nxt, col[rows]))
        srt = rows[order]
        vals = col[srt]
        for w in range(1, windows + 1):
            if srt.size <= w:
                break
            same = np.nonzero(vals[w:] == vals[:-w])[0]
            pairs.extend((int(srt[i]), int(srt[i + w])) for i in same[: max(0, max_pairs - len(pairs))])
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def discover_uccs(matrix: np.ndarray, columns: list[int], max_size: int, algorithm: str = "hyucc",
                  max_candidates: int = 100_000) -> UCCResult:
    """Minimal UCCs over ``columns`` (indices into matrix), each returned as sorted column indices."""
    if matrix.shape[0] == 0 or not columns:
        return UCCResult([], True, {"rows": int(matrix.shape[0])})
    checker = _Checker(matrix, columns)
    if algorithm == "apriori":
        masks, complete = _apriori(checker, max_size, max_candidates)
    else:
        masks, complete = _hyucc(checker, max_size, max_candidates)
    uccs = sorted(
        (tuple(columns[b] for b in _bits(mask)) for mask in masks), key=lambda t: (len(t), t)
    )
    return UCCResult(uccs, complete, {"rows": int(matrix.shape[0]), "columns": len(columns),
                                      "validations": checker.validations, "algorithm": algorithm})


def _hyucc(checker: _Checker, max_size: int, max_candidates: int) -> tuple[list[int], bool]:
    seen_diffs: set[int] = set()
    cover: set[int] = {0}
    diffs = checker.difference_sets(_sample_pairs(checker.m))
    for d in sorted(diffs, key=_popcount):
        seen_diffs.add(d)
        cover = _update_transversals(cover, d, max_size)
    valid: set[int] = set()
    while True:
        new_diffs: set[int] = set()
        for cand in sorted(cover - valid, key=lambda x: (_popcount(x), x)):
            ok, pairs = checker.check(cand)
            if ok:
                valid.add(cand)
            else:
                new_diffs |= checker.difference_sets(pairs)
        new_diffs -= seen_diffs
        if not new_diffs:
            return sorted(cover & valid), True
        for d in sorted(new_diffs, key=_popcount):
            seen_diffs.add(d)
            cover = _update_transversals(cover, d, max_size)
            if len(cover) > max_candidates:
                return sorted(cover & valid), False
        valid &= cover


def _apriori(checker: _Checker, max_size: int, max_candidates: int) -> tuple[list[int], bool]:
    uccs: list[int] = []
    level = [(b,) for b in range(checker.k)]
    size = 1
    while level and size <= max_size:
        non_unique: list[tuple[int, ...]] = []
        for cand in level:
            mask = sum(1 << b for b in cand)
            if any((u & mask) == u for u in uccs):
                continue
            ok, _ = checker.check(mask)
            if ok:
                uccs.append(mask)
            else:
                non_unique.append(cand)
        nonu = set(non_unique)
        nxt: list[tuple[int, ...]] = []
        non_unique.sort()
        for i, a in enumerate(non_unique):
            for b in non_unique[i + 1:]:
                if a[:-1] != b[:-1]:
                    break
                cand = a + (b[-1],)
                if all(sub in nonu for sub in combinations(cand, len(cand) - 1)):
                    nxt.append(cand)
                    if len(nxt) > max_candidates:
                        return uccs, False
        level = nxt
        size += 1
    return uccs, True

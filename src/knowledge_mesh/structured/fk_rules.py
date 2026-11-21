"""Foreign-key rules applied to BINDER's inclusion dependencies.

An inclusion ``A ⊆ B`` only says that A's values occur in B. Two observations turn it into a
foreign key:

* **Keys are identifiers.** The referencing column and the key it references are identifiers,
  codes, numbers, keys, indicators or flags, not dates, amounts or free text. A column counts as
  an identifier when it is part of a declared key, is a UUID, its business term ends with an
  identifier class word ("Loan Identifier", "Product Code"), or, while it has no business term,
  its name ends with an identifier token (``loan_id``, ``prd_cd``, ``ln_no``, ``active_flg``).
  Key discovery applies this before BINDER validates candidates (``identifier_basis``).
* **An FK references the primary key and shares its business term** ("Loan Identifier" ->
  "Loan Identifier"), possibly with a role qualifier ("Parent Counterparty Identifier" ->
  "Counterparty Identifier"). Referenced keys are a table's declared primary/unique keys when it
  has a declared primary key, otherwise its primary-key candidates; a column that is merely unique
  in the sample (``insurance.loan_id``) is not a target (``referenceable_keys``).

``assess_foreign_key`` grades an inferred FK whenever it is read (API, agent tool, knowledge
graph), from the business terms current at that moment, so terms generated after key discovery
raise or lower the confidence without re-running discovery:

* ``high``   every column pair shares the business term (exactly or role-qualified);
* ``medium`` no shared term (or no terms yet), but the column names match (same name,
  role-prefixed name, or ``<table>_id`` -> ``id``);
* ``low``    values only, or a business term says one side is not an identifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..storage.db import Database, loads

DEFAULT_CLASS_WORDS = ("Identifier", "Code", "Number", "Key", "Indicator", "Flag")
DEFAULT_NAME_TOKENS = {
    "id": "Identifier", "ident": "Identifier", "identifier": "Identifier", "uuid": "Identifier", "guid": "Identifier",
    "key": "Key", "cd": "Code", "code": "Code", "num": "Number", "nbr": "Number", "no": "Number", "number": "Number",
    "ind": "Indicator", "indicator": "Indicator", "flg": "Flag", "flag": "Flag",
}
# Name tokens too generic to identify the referenced entity on their own ("id", "code"): a column
# named ``customer_id`` matches ``customers.id`` only because "customer" names the table.
_GENERIC = {"id", "ident", "identifier", "key", "cd", "code", "no", "num", "nbr", "number", "uuid", "guid"}
_LEVELS = {"low": 0, "medium": 1, "high": 2}


def name_tokens(name: str | None) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name or ""))
    return [t.lower() for t in re.split(r"[_\W]+", spaced) if t]


def term_words(label: str | None) -> list[str]:
    return [w.lower() for w in re.split(r"[^\w]+", str(label or "")) if w]


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _term(col: dict[str, Any]) -> str | None:
    return col.get("term_label") if col.get("term_status") == "generated" and col.get("term_label") else None


@dataclass(frozen=True)
class FKRules:
    class_words: frozenset[str] = frozenset(w.lower() for w in DEFAULT_CLASS_WORDS)
    name_tokens: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_NAME_TOKENS))
    identifiers_only: bool = True
    reference_primary_keys: bool = True

    @classmethod
    def from_settings(cls, settings=None) -> "FKRules":
        if settings is None:
            return cls()
        return cls(frozenset(w.lower() for w in settings.fk_identifier_class_words),
                   {k.lower(): v for k, v in settings.fk_identifier_name_tokens.items()},
                   settings.fk_identifiers_only, settings.fk_reference_primary_keys)

    def referenceable_keys(self, declared: list[tuple[str, tuple[str, ...]]], inferred: list[tuple[str, ...]],
                           null_free: dict[str, bool]) -> set[tuple[str, ...]]:
        """Keys an FK may reference. ``declared``: (key_type, columns) of the table's declared keys."""
        declared_unique = {cols for kind, cols in declared if kind in {"primary", "unique"} and cols}
        if not self.reference_primary_keys:
            return declared_unique | set(inferred)
        if any(kind == "primary" for kind, _ in declared):
            return declared_unique
        return declared_unique | {k for k in inferred if all(null_free.get(c, False) for c in k)}

    # -- identifiers ---------------------------------------------------------------------------
    def term_is_identifier(self, label: str | None) -> bool:
        words = term_words(label)
        return bool(words) and words[-1] in self.class_words

    def name_is_identifier(self, name: str | None) -> bool:
        tokens = name_tokens(name)
        cls = self.name_tokens.get(tokens[-1]) if tokens else None
        return bool(cls) and cls.lower() in self.class_words

    def identifier_basis(self, col: dict[str, Any], declared: bool = False, category: str | None = None,
                         content_type: str | None = None) -> str | None:
        """Why ``col`` (a ``columns`` row) is an identifier, or None when it is not one.

        Order: declared key, UUID, business term (decides when present), column name.
        """
        if declared:
            return "declared_key"
        if "uuid" in str(col.get("data_type") or "").lower() or "uuid" in (category, content_type):
            return "uuid"
        term = _term(col)
        if term is not None:
            return "business_term" if self.term_is_identifier(term) else None
        return "column_name" if self.name_is_identifier(col.get("name")) else None

    # -- matching ------------------------------------------------------------------------------
    @staticmethod
    def term_match(a: str | None, b: str | None) -> bool:
        """Same business term, or one is the other with a role qualifier in front."""
        wa, wb = term_words(a), term_words(b)
        if not wa or not wb:
            return False
        if wa == wb:
            return True
        short, long_ = sorted((wa, wb), key=len)
        return len(short) >= 2 and long_[-len(short):] == short

    @staticmethod
    def name_match(dep: str, ref: str, ref_table: str | None = None) -> bool:
        """Same column name, role-prefixed name (parent_cpty_id -> cpty_id) or <table>_id -> id."""
        d, r = name_tokens(dep), name_tokens(ref)
        if not d or not r:
            return False
        if d == r:
            return True
        if len(d) <= len(r) or d[-len(r):] != r:
            return False
        if len(r) >= 2 or r[0] not in _GENERIC:
            return True
        prefix = [_singular(t) for t in d[:-len(r)]]
        table = [_singular(t) for t in name_tokens(ref_table)]
        return bool(table) and table[-len(prefix):] == prefix

    def assess_pair(self, dep: dict[str, Any], ref: dict[str, Any], ref_table: str | None,
                    dep_declared: bool, ref_declared: bool) -> tuple[str, str]:
        dep_term, ref_term = _term(dep), _term(ref)
        for col, term, declared in ((dep, dep_term, dep_declared), (ref, ref_term, ref_declared)):
            if term and not declared and "uuid" not in str(col.get("data_type") or "").lower() \
                    and not self.term_is_identifier(term):
                return "low", f"{col['name']}: business term '{term}' is not an identifier"
        if dep_term and ref_term and self.term_match(dep_term, ref_term):
            same = term_words(dep_term) == term_words(ref_term)
            return "high", (f"same business term '{ref_term}'" if same
                            else f"business term '{dep_term}' qualifies '{ref_term}'")
        if self.name_match(dep["name"], ref["name"], ref_table):
            why = "same column name" if name_tokens(dep["name"]) == name_tokens(ref["name"]) else "matching column name"
            if dep_term and ref_term:
                why += f" (terms differ: '{dep_term}' / '{ref_term}')"
            else:
                why += " (business terms not generated yet)"
            return "medium", why
        if dep_term and ref_term:
            return "low", f"values only; terms differ: '{dep_term}' / '{ref_term}'"
        return "low", "values only"

    def assess(self, pairs: list[tuple[dict[str, Any], dict[str, Any]]], ref_table: str | None,
               dep_declared: set[str], ref_declared: set[str]) -> dict[str, Any]:
        graded = [(d["name"], r["name"], *self.assess_pair(d, r, ref_table, d["name"] in dep_declared,
                                                           r["name"] in ref_declared)) for d, r in pairs]
        if not graded:
            return {"level": "low", "basis": "values_only", "reasons": ["referenced columns not cataloged"]}
        level = min((g[2] for g in graded), key=_LEVELS.__getitem__)
        basis = {"high": "business_term", "medium": "column_name", "low": "values_only"}[level]
        return {"level": level, "basis": basis, "reasons": [f"{d} → {r}: {why}" for d, r, _, why in graded]}


def declared_columns(db: Database, dataset_id: str) -> set[str]:
    """Columns that belong to any declared key (primary, unique or foreign) of the dataset."""
    out: set[str] = set()
    for k in db.query("SELECT columns_json FROM keys WHERE dataset_id = ? AND origin = 'declared' AND active = 1",
                      (dataset_id,)):
        out.update(loads(k["columns_json"], []))
    return out


def assess_foreign_key(db: Database, key: dict[str, Any], rules: FKRules | None = None) -> dict[str, Any] | None:
    """Confidence of an inferred FK (a ``keys`` row) from the current business terms; None otherwise."""
    if key.get("key_type") != "foreign" or key.get("origin") != "inferred":
        return None
    rules = rules or FKRules()
    dep_cols = {c["name"]: c for c in db.query("SELECT * FROM columns WHERE dataset_id = ? AND active = 1",
                                               (key["dataset_id"],))}
    ref_cols = {c["name"]: c for c in db.query("SELECT * FROM columns WHERE dataset_id = ? AND active = 1",
                                               (key["referenced_dataset_id"],))}
    ref_ds = db.query_one("SELECT name FROM datasets WHERE dataset_id = ?", (key["referenced_dataset_id"],))
    cols, ref = loads(key["columns_json"], []), loads(key["referenced_columns_json"], [])
    pairs = [(dep_cols[a], ref_cols[b]) for a, b in zip(cols, ref) if a in dep_cols and b in ref_cols]
    if len(pairs) != len(cols):
        pairs = []
    return rules.assess(pairs, ref_ds["name"] if ref_ds else None, declared_columns(db, key["dataset_id"]),
                        declared_columns(db, key["referenced_dataset_id"]))

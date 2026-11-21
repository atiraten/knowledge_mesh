"""Verify / reject decisions on inferred knowledge (spec 2.11, 6.8).

Reviewable objects and their identifiers:

* ``foreign_key``     an inferred FK (``keys.key_id``); declared FKs are always ``verified``;
* ``relationship``    a proposed named relationship (``semantic_relationships.relationship_id``);
* ``term_relation``   a proposed term relation (``term_relations.relation_id``);
* ``entity_mapping``  a table -> business entity link (``datasets.dataset_id``);
* ``term_mapping``    a column -> business term link (``columns.column_id``).

Everything inferred or LLM-proposed starts ``inferred``. A decision is keyed by the object's
stable ID, so it survives catalog, key-discovery and relationship re-runs. Mapping decisions also
record the entity/term they were made for (``subject_slug``): when term generation later assigns a
different term, the old decision no longer applies and the new link is ``inferred`` again.
"""

from __future__ import annotations

from typing import Any

from ..common.errors import InvalidArgument, NotFound
from ..common.ids import utcnow
from ..storage.db import Database

OBJECT_TYPES = ("foreign_key", "relationship", "term_relation", "entity_mapping", "term_mapping")
DECISIONS = ("verified", "rejected", "inferred")  # inferred = clear the decision


def lookup(decisions: dict[tuple[str, str], dict[str, Any]], object_type: str, object_id: str,
           subject_slug: str | None = None) -> str:
    """Status of an inferred object from a snapshot of the decisions (``ReviewStore.all()``)."""
    row = decisions.get((object_type, object_id))
    if row is None or (subject_slug is not None and row.get("subject_slug") not in (None, subject_slug)):
        return "inferred"
    return row["status"]


def combine(*statuses: str) -> str:
    """Status of something that depends on several statuses (e.g. a join mapping = relationship + FK)."""
    if "rejected" in statuses:
        return "rejected"
    return "verified" if all(s == "verified" for s in statuses) else "inferred"


class ReviewStore:
    def __init__(self, db: Database):
        self.db = db

    def all(self) -> dict[tuple[str, str], dict[str, Any]]:
        return {(r["object_type"], r["object_id"]): r for r in self.db.query("SELECT * FROM review_decisions")}

    def status(self, object_type: str, object_id: str, subject_slug: str | None = None) -> str:
        return lookup(self.all(), object_type, object_id, subject_slug)

    def decision(self, object_type: str, object_id: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM review_decisions WHERE object_type = ? AND object_id = ?",
                                 (object_type, object_id))

    def target(self, object_type: str, object_id: str) -> dict[str, Any]:
        """The reviewed object's source, dataset and (for mappings) current entity/term slug."""
        if object_type == "foreign_key":
            row = self.db.query_one(
                "SELECT k.key_id, k.key_type, k.origin, d.source_id, d.dataset_id FROM keys k "
                "JOIN datasets d ON d.dataset_id = k.dataset_id WHERE k.key_id = ? AND k.active = 1", (object_id,))
            if row is None or row["key_type"] != "foreign":
                raise NotFound(f"foreign key not found: {object_id}")
            if row["origin"] != "inferred":
                raise InvalidArgument("declared foreign keys are verified by their source and cannot be reviewed")
            return {"source_id": row["source_id"], "dataset_id": row["dataset_id"], "subject_slug": None}
        if object_type == "relationship":
            row = self.db.query_one("SELECT source_id, dataset_id FROM semantic_relationships "
                                    "WHERE relationship_id = ? AND active = 1", (object_id,))
            if row is None:
                raise NotFound(f"relationship not found: {object_id}")
            return {**row, "subject_slug": None}
        if object_type == "term_relation":
            row = self.db.query_one("SELECT source_id, dataset_id FROM term_relations "
                                    "WHERE relation_id = ? AND active = 1", (object_id,))
            if row is None:
                raise NotFound(f"term relation not found: {object_id}")
            return {**row, "subject_slug": None}
        if object_type == "entity_mapping":
            row = self.db.query_one("SELECT source_id, dataset_id, entity_slug FROM datasets "
                                    "WHERE dataset_id = ? AND active = 1", (object_id,))
            if row is None or not row["entity_slug"]:
                raise NotFound(f"no business entity link for dataset {object_id}")
            return {"source_id": row["source_id"], "dataset_id": row["dataset_id"], "subject_slug": row["entity_slug"]}
        if object_type == "term_mapping":
            row = self.db.query_one(
                "SELECT d.source_id, d.dataset_id, c.term_slug, c.term_status FROM columns c "
                "JOIN datasets d ON d.dataset_id = c.dataset_id WHERE c.column_id = ? AND c.active = 1", (object_id,))
            if row is None or row["term_status"] != "generated" or not row["term_slug"]:
                raise NotFound(f"no business term link for column {object_id}")
            return {"source_id": row["source_id"], "dataset_id": row["dataset_id"], "subject_slug": row["term_slug"]}
        raise InvalidArgument(f"unknown object_type {object_type!r}", {"allowed": list(OBJECT_TYPES)})

    def set(self, object_type: str, object_id: str, status: str, actor: str | None,
            note: str | None = None) -> dict[str, Any]:
        if object_type not in OBJECT_TYPES:
            raise InvalidArgument(f"unknown object_type {object_type!r}", {"allowed": list(OBJECT_TYPES)})
        if status not in DECISIONS:
            raise InvalidArgument(f"status must be one of {list(DECISIONS)}")
        target = self.target(object_type, object_id)
        if status == "inferred":
            self.db.execute("DELETE FROM review_decisions WHERE object_type = ? AND object_id = ?",
                            (object_type, object_id))
        else:
            self.db.execute(
                "INSERT INTO review_decisions(object_type, object_id, status, subject_slug, actor, note, decided_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(object_type, object_id) DO UPDATE SET status=excluded.status, "
                "subject_slug=excluded.subject_slug, actor=excluded.actor, note=excluded.note, "
                "decided_at=excluded.decided_at",
                (object_type, object_id, status, target["subject_slug"], actor, (note or "")[:1000] or None,
                 utcnow()))
        return {"object_type": object_type, "object_id": object_id, "status": status, **target}

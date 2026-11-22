"""Business glossary store: business terms and business entities, one row each.

``business_terms`` and ``business_entities`` hold every term and entity once (label, description, origin).
Catalog columns reference a term by ``columns.term_slug``; datasets reference an entity by
``datasets.entity_slug``. ``columns_select`` / ``datasets_select`` read columns and datasets with the term /
entity label and description joined from these tables. The older per-row copies
(``columns.term_label`` / ``term_description`` / ``term_sensitive``, ``datasets.entity_label``) are no longer
read or written.

* Slugs are identities: editing a label keeps the slug, so graph URIs and every reference stay valid.
* Origins: ``generated`` (term generation), ``document`` (found in documents), ``manual`` (added in the
  glossary), ``edited`` (generated, then edited in the glossary).
* Term generation never changes a ``manual`` / ``edited`` row and never re-creates a deleted one
  (``deleted = 1`` is a tombstone); document extraction never re-creates a deleted entity.
* A term referenced by a business definition cannot be deleted. Deleting an entity removes it everywhere:
  datasets lose it, its document mentions and document relationships are removed, and relationships to or
  from it are deactivated.
* Generated terms and generated / document entities that nothing uses any more (e.g. their source was
  removed) are dropped by ``prune_unused``; manual and edited rows stay.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Iterable

from ..common.errors import Conflict, InvalidArgument, NotFound
from ..common.ids import slugify, utcnow
from ..storage.db import Database

MANUAL_ORIGINS = ("manual", "edited")
_LEGACY = {"columns": ("term_label", "term_description", "term_sensitive"), "datasets": ("entity_label",)}
MAX_LABEL = 200
MAX_DESCRIPTION = 2000


def _fields(db: Database, table: str, alias: str) -> str:
    cache = db.__dict__.setdefault("_glossary_fields", {})
    if table not in cache:
        names = [r["name"] for r in db.query(f"PRAGMA table_info({table})") if r["name"] not in _LEGACY[table]]
        cache[table] = ", ".join(f"{alias}.{n}" for n in names)
    return cache[table]


def columns_select(db: Database) -> str:
    """``SELECT ... FROM columns c`` with the business term (``term_label``, ``term_description``,
    ``term_sensitive``, ``term_origin``) read from ``business_terms``. Append ``WHERE`` with ``c.`` names."""
    return (f"SELECT {_fields(db, 'columns', 'c')}, t.label AS term_label, t.description AS term_description, "
            "t.sensitive AS term_sensitive, t.origin AS term_origin FROM columns c "
            "LEFT JOIN business_terms t ON t.slug = c.term_slug AND t.deleted = 0")


def datasets_select(db: Database) -> str:
    """``SELECT ... FROM datasets d`` with the business entity (``entity_label``, ``entity_description``,
    ``entity_origin``) read from ``business_entities``. Append ``WHERE`` with ``d.`` names."""
    return (f"SELECT {_fields(db, 'datasets', 'd')}, e.label AS entity_label, e.description AS entity_description, "
            "e.origin AS entity_origin FROM datasets d "
            "LEFT JOIN business_entities e ON e.slug = d.entity_slug AND e.deleted = 0")


def _label(value: str | None, what: str) -> str:
    label = " ".join((value or "").split())
    if not label or len(label) > MAX_LABEL:
        raise InvalidArgument(f"{what} label is required (max {MAX_LABEL} characters)")
    if not slugify(label):
        raise InvalidArgument(f"{what} label must contain letters or digits")
    return label


def _description(value: str | None) -> str | None:
    text = (value or "").strip()
    if len(text) > MAX_DESCRIPTION:
        raise InvalidArgument(f"description must be at most {MAX_DESCRIPTION} characters")
    return text or None


def _sensitive(value: Any) -> int | None:
    return None if value is None else int(bool(value))


def _sensitive_of(flags: list[Any]) -> int | None:
    """One flag for a term shared by several columns: 1 if any column says so, 0 if all say no, else unknown."""
    if 1 in flags:
        return 1
    return 0 if flags and None not in flags else None


class GlossaryStore:
    def __init__(self, db: Database):
        self.db = db

    # -- one-time fill from the per-row copies ---------------------------------------------------------
    def fill_if_empty(self) -> dict[str, int]:
        """Fill ``business_terms`` / ``business_entities`` from the catalog when a table is empty.
        Terms: one per ``columns.term_slug`` (the most common label and description when columns differ)
        plus the terms of term relations. Entities: dataset entities, relationship ends, then entities
        found only in documents (origin ``document``)."""
        now = utcnow()
        out = {"terms": 0, "entities": 0, "label_conflicts": 0}
        with self.db.transaction() as conn:
            if conn.execute("SELECT COUNT(*) FROM business_terms").fetchone()[0] == 0:
                groups: dict[str, list[dict[str, Any]]] = {}
                for r in conn.execute("SELECT term_slug, term_label, term_description, term_sensitive FROM columns "
                                      "WHERE active = 1 AND term_status = 'generated' AND term_slug IS NOT NULL "
                                      "AND term_label IS NOT NULL"):
                    groups.setdefault(r["term_slug"], []).append(dict(r))
                for slug, rows in groups.items():
                    labels = Counter(r["term_label"] for r in rows)
                    descriptions = Counter(r["term_description"] for r in rows if r["term_description"])
                    if len(labels) > 1 or len(descriptions) > 1:
                        out["label_conflicts"] += 1
                    conn.execute(
                        "INSERT INTO business_terms(slug, label, description, sensitive, origin, created_at, updated_at)"
                        " VALUES (?,?,?,?,'generated',?,?)",
                        (slug, labels.most_common(1)[0][0],
                         descriptions.most_common(1)[0][0] if descriptions else None,
                         _sensitive_of([r["term_sensitive"] for r in rows]), now, now))
                    out["terms"] += 1
                for r in conn.execute("SELECT subject_slug AS slug, subject_label AS label FROM term_relations "
                                      "UNION SELECT object_slug, object_label FROM term_relations"):
                    if r["slug"] and r["label"]:
                        out["terms"] += conn.execute(
                            "INSERT OR IGNORE INTO business_terms(slug, label, origin, created_at, updated_at) "
                            "VALUES (?,?,'generated',?,?)", (r["slug"], r["label"], now, now)).rowcount
            if conn.execute("SELECT COUNT(*) FROM business_entities").fetchone()[0] == 0:
                sources = [
                    ("generated", "SELECT entity_slug AS slug, entity_label AS label FROM datasets "
                                  "WHERE active = 1 AND entity_slug IS NOT NULL AND entity_label IS NOT NULL"),
                    ("generated", "SELECT source_entity_slug AS slug, source_entity_label AS label FROM semantic_relationships "
                                  "UNION ALL SELECT target_entity_slug, target_entity_label FROM semantic_relationships"),
                    ("document", "SELECT entity_slug AS slug, entity_label AS label FROM doc_entity_mentions"),
                    ("document", "SELECT source_slug AS slug, source_label AS label FROM doc_relationships "
                                 "UNION ALL SELECT target_slug, target_label FROM doc_relationships"),
                ]
                for origin, sql in sources:
                    labels: dict[str, Counter] = {}
                    for r in conn.execute(sql):
                        if r["slug"] and r["label"]:
                            labels.setdefault(r["slug"], Counter())[r["label"]] += 1
                    for slug, counts in labels.items():
                        out["entities"] += conn.execute(
                            "INSERT OR IGNORE INTO business_entities(slug, label, origin, created_at, updated_at) "
                            "VALUES (?,?,?,?,?)", (slug, counts.most_common(1)[0][0], origin, now, now)).rowcount
        return out

    # -- terms -----------------------------------------------------------------------------------------
    def list_terms(self, query: str | None = None) -> list[dict[str, Any]]:
        where, params = ["t.deleted = 0"], []
        if query and query.strip():
            like = f"%{query.strip().lower()}%"
            where.append("(LOWER(t.label) LIKE ? OR LOWER(COALESCE(t.description, '')) LIKE ?)")
            params += [like, like]
        return self.db.query(
            "SELECT t.slug, t.label, t.description, t.sensitive, t.origin, t.created_at, t.updated_at, t.updated_by,"
            " (SELECT COUNT(*) FROM columns c WHERE c.term_slug = t.slug AND c.active = 1"
            "  AND c.term_status = 'generated') AS column_count,"
            " (SELECT COUNT(*) FROM definition_links l JOIN business_definitions d ON d.definition_id = l.definition_id"
            "  WHERE l.target_type = 'term' AND l.target_slug = t.slug AND d.active = 1) AS definition_count"
            f" FROM business_terms t WHERE {' AND '.join(where)} ORDER BY LOWER(t.label)", params)

    def term(self, slug: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM business_terms WHERE slug = ? AND deleted = 0", (slug,))
        if row is None:
            raise NotFound(f"business term not found: {slug}")
        return row

    def add_term(self, label: str, description: str | None = None, sensitive: bool | None = None,
                 actor: str | None = None) -> dict[str, Any]:
        label, description = _label(label, "term"), _description(description)
        slug, now = slugify(label), utcnow()
        with self.db.transaction() as conn:
            self._check_term_label(conn, label, None)
            row = conn.execute("SELECT deleted FROM business_terms WHERE slug = ?", (slug,)).fetchone()
            if row is not None and not row["deleted"]:
                raise Conflict(f"a business term with this name already exists: {slug}", {"slug": slug})
            conn.execute(
                "INSERT INTO business_terms(slug, label, description, sensitive, origin, deleted, created_at,"
                " updated_at, updated_by) VALUES (?,?,?,?,'manual',0,?,?,?) ON CONFLICT(slug) DO UPDATE SET"
                " label=excluded.label, description=excluded.description, sensitive=excluded.sensitive,"
                " origin='manual', deleted=0, created_at=excluded.created_at, updated_at=excluded.updated_at,"
                " updated_by=excluded.updated_by", (slug, label, description, _sensitive(sensitive), now, now, actor))
        return self.term(slug)

    def update_term(self, slug: str, label: str, description: str | None = None, sensitive: bool | None = None,
                    actor: str | None = None) -> dict[str, Any]:
        current = self.term(slug)
        label, description = _label(label, "term"), _description(description)
        origin = "edited" if current["origin"] == "generated" else current["origin"]
        with self.db.transaction() as conn:
            self._check_term_label(conn, label, slug)
            conn.execute("UPDATE business_terms SET label=?, description=?, sensitive=?, origin=?, updated_at=?,"
                         " updated_by=? WHERE slug=?",
                         (label, description, _sensitive(sensitive), origin, utcnow(), actor, slug))
            conn.execute("UPDATE definition_links SET target_label=? WHERE target_type='term' AND target_slug=?",
                         (label, slug))
        return self.term(slug)

    def delete_term(self, slug: str, actor: str | None = None) -> dict[str, Any]:
        """Tombstone the term, unbind its columns (``term_status = 'removed'``, skipped by term generation)
        and deactivate its term relations and synonyms. Refused while a definition references it."""
        self.term(slug)
        used = self.db.query(
            "SELECT d.definition_id, d.name FROM definition_links l JOIN business_definitions d"
            " ON d.definition_id = l.definition_id WHERE l.target_type = 'term' AND l.target_slug = ? AND d.active = 1"
            " ORDER BY d.name", (slug,))
        if used:
            raise Conflict("the business term is referenced by business definitions; edit or delete them first",
                           {"definitions": used})
        affected = self.term_scope(slug)
        now = utcnow()
        with self.db.transaction() as conn:
            conn.execute("UPDATE business_terms SET deleted=1, updated_at=?, updated_by=? WHERE slug=?",
                         (now, actor, slug))
            conn.execute("UPDATE columns SET term_slug=NULL, term_explanation=NULL, term_status='removed',"
                         " term_error=? WHERE term_slug=?", (f"term {slug} deleted from the business glossary", slug))
            conn.execute("UPDATE term_relations SET active=0 WHERE subject_slug=? OR object_slug=?", (slug, slug))
            conn.execute("UPDATE term_synonyms SET active=0 WHERE term_slug=?", (slug,))
        return {"slug": slug, "deleted": True, **affected}

    def term_scope(self, slug: str) -> dict[str, list[str]]:
        """Datasets and sources whose published graph or glossary records show the term."""
        datasets = self.db.query("SELECT DISTINCT c.dataset_id, d.source_id FROM columns c JOIN datasets d"
                                 " ON d.dataset_id = c.dataset_id WHERE c.term_slug = ? AND c.active = 1", (slug,))
        sources = {r["source_id"] for r in datasets} | {r["source_id"] for r in self.db.query(
            "SELECT DISTINCT source_id FROM term_relations WHERE (subject_slug=? OR object_slug=?) AND active=1",
            (slug, slug))}
        return {"dataset_ids": sorted({r["dataset_id"] for r in datasets}), "source_ids": sorted(sources)}

    @staticmethod
    def _check_term_label(conn, label: str, slug: str | None) -> None:
        clash = conn.execute("SELECT slug FROM business_terms WHERE deleted = 0 AND LOWER(label) = LOWER(?)"
                             " AND slug != COALESCE(?, '')", (label, slug)).fetchone()
        if clash is not None:
            raise Conflict(f"another business term already has this label: {clash['slug']}", {"slug": clash["slug"]})

    # -- entities --------------------------------------------------------------------------------------
    def list_entities(self, query: str | None = None) -> list[dict[str, Any]]:
        where, params = ["e.deleted = 0"], []
        if query and query.strip():
            like = f"%{query.strip().lower()}%"
            where.append("(LOWER(e.label) LIKE ? OR LOWER(COALESCE(e.description, '')) LIKE ?)")
            params += [like, like]
        return self.db.query(
            "SELECT e.slug, e.label, e.description, e.origin, e.created_at, e.updated_at, e.updated_by,"
            " (SELECT COUNT(*) FROM datasets d WHERE d.entity_slug = e.slug AND d.active = 1) AS dataset_count,"
            " (SELECT COUNT(*) FROM doc_entity_mentions m WHERE m.entity_slug = e.slug) AS mention_count"
            f" FROM business_entities e WHERE {' AND '.join(where)} ORDER BY LOWER(e.label)", params)

    def entity(self, slug: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM business_entities WHERE slug = ? AND deleted = 0", (slug,))
        if row is None:
            raise NotFound(f"business entity not found: {slug}")
        return row

    def update_entity(self, slug: str, label: str, description: str | None = None,
                      actor: str | None = None) -> dict[str, Any]:
        current = self.entity(slug)
        label, description = _label(label, "entity"), _description(description)
        origin = current["origin"] if current["origin"] == "manual" else "edited"
        with self.db.transaction() as conn:
            clash = conn.execute("SELECT slug FROM business_entities WHERE deleted = 0 AND LOWER(label) = LOWER(?)"
                                 " AND slug != ?", (label, slug)).fetchone()
            if clash is not None:
                raise Conflict(f"another business entity already has this label: {clash['slug']}",
                               {"slug": clash["slug"]})
            conn.execute("UPDATE business_entities SET label=?, description=?, origin=?, updated_at=?, updated_by=?"
                         " WHERE slug=?", (label, description, origin, utcnow(), actor, slug))
            # keep the stored copies of the label consistent (relationship labels, document mentions)
            conn.execute("UPDATE semantic_relationships SET source_entity_label=? WHERE source_entity_slug=?", (label, slug))
            conn.execute("UPDATE semantic_relationships SET target_entity_label=? WHERE target_entity_slug=?", (label, slug))
            conn.execute("UPDATE doc_entity_mentions SET entity_label=? WHERE entity_slug=?", (label, slug))
            conn.execute("UPDATE doc_relationships SET source_label=? WHERE source_slug=?", (label, slug))
            conn.execute("UPDATE doc_relationships SET target_label=? WHERE target_slug=?", (label, slug))
        return self.entity(slug)

    def delete_entity(self, slug: str, actor: str | None = None) -> dict[str, Any]:
        """Tombstone the entity and remove it everywhere (datasets, document mentions and relationships,
        semantic relationships). Returns what was affected, for republishing."""
        self.entity(slug)
        affected = self.entity_scope(slug)
        now = utcnow()
        with self.db.transaction() as conn:
            conn.execute("UPDATE business_entities SET deleted=1, updated_at=?, updated_by=? WHERE slug=?",
                         (now, actor, slug))
            datasets = conn.execute("UPDATE datasets SET entity_slug=NULL, entity_method='removed' WHERE entity_slug=?",
                                    (slug,)).rowcount
            mentions = conn.execute("DELETE FROM doc_entity_mentions WHERE entity_slug=?", (slug,)).rowcount
            doc_rels = conn.execute("DELETE FROM doc_relationships WHERE source_slug=? OR target_slug=?",
                                    (slug, slug)).rowcount
            relationships = conn.execute("UPDATE semantic_relationships SET active=0 WHERE active=1 AND"
                                         " (source_entity_slug=? OR target_entity_slug=?)", (slug, slug)).rowcount
        return {"slug": slug, "deleted": True, **affected,
                "removed": {"datasets": datasets, "document_mentions": mentions, "document_relationships": doc_rels,
                            "relationships": relationships}}

    def entity_scope(self, slug: str) -> dict[str, list[str]]:
        """Datasets, sources and documents whose published graph or glossary records show the entity."""
        datasets = self.db.query("SELECT dataset_id, source_id FROM datasets WHERE entity_slug = ? AND active = 1",
                                 (slug,))
        sources = {r["source_id"] for r in datasets} | {r["source_id"] for r in self.db.query(
            "SELECT DISTINCT source_id FROM semantic_relationships WHERE active = 1 AND"
            " (source_entity_slug = ? OR target_entity_slug = ?)", (slug, slug))}
        documents = {r["document_id"] for r in self.db.query(
            "SELECT DISTINCT document_id FROM doc_entity_mentions WHERE entity_slug = ? UNION"
            " SELECT DISTINCT document_id FROM doc_relationships WHERE source_slug = ? OR target_slug = ?",
            (slug, slug, slug))}
        return {"dataset_ids": sorted({r["dataset_id"] for r in datasets}), "source_ids": sorted(sources),
                "document_ids": sorted(documents)}

    # -- writers used by term generation and document indexing -------------------------------------------
    @staticmethod
    def generated_term(conn, label: str, description: str | None, sensitive: bool | None, now: str,
                       dataset_id: str) -> str | None:
        """Slug of a generated term after recording it for a column of ``dataset_id``.

        * new term: recorded with this label, description and flag;
        * generated term used only by this dataset: refreshed with the new label, description and flag;
        * generated term also used by other datasets (shared): label and description kept, so regenerating
          one table does not change what the others show; flagged sensitive if this column says so;
        * manual / edited term: left as it is; deleted term: not re-created (None)."""
        slug = slugify(label)
        row = conn.execute("SELECT origin, deleted, sensitive FROM business_terms WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO business_terms(slug, label, description, sensitive, origin, created_at,"
                         " updated_at) VALUES (?,?,?,?,'generated',?,?)",
                         (slug, label, description, _sensitive(sensitive), now, now))
        elif row["deleted"]:
            return None
        elif row["origin"] == "generated":
            shared = conn.execute("SELECT 1 FROM columns WHERE term_slug = ? AND active = 1 AND dataset_id != ?"
                                  " AND term_status = 'generated' LIMIT 1", (slug, dataset_id)).fetchone()
            if shared:
                if sensitive:
                    conn.execute("UPDATE business_terms SET sensitive=1, updated_at=? WHERE slug=?", (now, slug))
            else:
                conn.execute("UPDATE business_terms SET label=?, description=?, sensitive=?, updated_at=? WHERE slug=?",
                             (label, description, _sensitive(sensitive), now, slug))
        return slug

    @staticmethod
    def generated_entity(conn, label: str, now: str) -> str | None:
        """Slug of a generated table entity after recording it (None: the entity was deleted by hand)."""
        slug = slugify(label)
        row = conn.execute("SELECT origin, deleted FROM business_entities WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO business_entities(slug, label, origin, created_at, updated_at)"
                         " VALUES (?,?,'generated',?,?)", (slug, label, now, now))
        elif row["deleted"]:
            return None
        elif row["origin"] in ("generated", "document"):
            conn.execute("UPDATE business_entities SET label=?, origin='generated', updated_at=? WHERE slug=?",
                         (label, now, slug))
        return slug

    @staticmethod
    def resolve_document_entities(conn, mentions: Iterable, relationships: Iterable) -> tuple[list, list]:
        """Map entities found in a document to glossary entities: by slug, then by label (an entity renamed
        in the glossary keeps its slug); unknown ones are recorded with origin ``document``; deleted ones are
        dropped. Returns the mentions and relationships with glossary slugs and labels."""
        now = utcnow()
        cache: dict[tuple[str, str], tuple[str, str] | None] = {}

        def resolve(slug: str, label: str) -> tuple[str, str] | None:
            key = (slug, label.lower())
            if key not in cache:
                row = conn.execute("SELECT slug, label, deleted FROM business_entities WHERE slug = ?", (slug,)).fetchone()
                if row is None:
                    row = conn.execute("SELECT slug, label, deleted FROM business_entities WHERE LOWER(label) = ?"
                                       " ORDER BY deleted LIMIT 1", (label.lower(),)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO business_entities(slug, label, origin, created_at, updated_at)"
                                 " VALUES (?,?,'document',?,?)", (slug, label, now, now))
                    cache[key] = (slug, label)
                else:
                    cache[key] = None if row["deleted"] else (row["slug"], row["label"])
            return cache[key]

        out_mentions, out_rels = [], []
        for m in mentions:
            hit = resolve(m.slug, m.label)
            if hit:
                out_mentions.append(replace(m, slug=hit[0], label=hit[1]))
        for r in relationships:
            src, dst = resolve(r.source_slug, r.source_label), resolve(r.target_slug, r.target_label)
            if src and dst and src[0] != dst[0]:
                out_rels.append(replace(r, source_slug=src[0], source_label=src[1],
                                        target_slug=dst[0], target_label=dst[1]))
        return out_mentions, out_rels

    def prune_unused(self) -> dict[str, int]:
        """Delete generated terms and generated / document entities that nothing uses any more (their source
        was removed, or term generation replaced them). Manual and edited rows and tombstones are kept."""
        with self.db.transaction() as conn:
            terms = conn.execute(
                "DELETE FROM business_terms WHERE origin = 'generated' AND deleted = 0"
                " AND NOT EXISTS (SELECT 1 FROM columns c WHERE c.term_slug = business_terms.slug AND c.active = 1)"
                " AND NOT EXISTS (SELECT 1 FROM definition_links l JOIN business_definitions d"
                "  ON d.definition_id = l.definition_id WHERE l.target_type = 'term'"
                "  AND l.target_slug = business_terms.slug AND d.active = 1)"
                " AND NOT EXISTS (SELECT 1 FROM term_relations r WHERE r.active = 1"
                "  AND business_terms.slug IN (r.subject_slug, r.object_slug))").rowcount
            entities = conn.execute(
                "DELETE FROM business_entities WHERE origin IN ('generated', 'document') AND deleted = 0"
                " AND NOT EXISTS (SELECT 1 FROM datasets d WHERE d.entity_slug = business_entities.slug AND d.active = 1)"
                " AND NOT EXISTS (SELECT 1 FROM doc_entity_mentions m WHERE m.entity_slug = business_entities.slug)"
                " AND NOT EXISTS (SELECT 1 FROM doc_relationships r"
                "  WHERE business_entities.slug IN (r.source_slug, r.target_slug))"
                " AND NOT EXISTS (SELECT 1 FROM semantic_relationships s WHERE s.active = 1"
                "  AND business_entities.slug IN (s.source_entity_slug, s.target_entity_slug))").rowcount
        return {"terms": terms, "entities": entities}

    def labels(self) -> tuple[dict[str, str], dict[str, str]]:
        """(term slug -> label, entity slug -> label) of active rows."""
        terms = {r["slug"]: r["label"] for r in self.db.query("SELECT slug, label FROM business_terms WHERE deleted = 0")}
        entities = {r["slug"]: r["label"] for r in self.db.query(
            "SELECT slug, label FROM business_entities WHERE deleted = 0")}
        return terms, entities

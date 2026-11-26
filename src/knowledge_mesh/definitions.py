"""Business definitions (FR-13): captured text that defines business terms.

A definition references one or more business terms of the business glossary (``business_terms``; never
business entities) and, optionally, the policy document where it is defined (an ingested document, with an
optional page or section ``location``).

The graph is the gate: the definition context is SHACL-validated and published first, and
the SQLite record is written only if publication succeeds. A rejected definition therefore
leaves both stores exactly as they were.
"""

from __future__ import annotations

from typing import Any

from .common.errors import Conflict, InvalidArgument, NotFound
from .common.ids import slugify, utcnow
from .graph.builders import build_definition_context
from .graph.ontology import context_uri, definition_uri
from .graph.store import KnowledgeGraph
from .storage.db import Database

MAX_TERMS = 50
MAX_LOCATION = 200


class DefinitionService:
    def __init__(self, db: Database, graph: KnowledgeGraph):
        self.db = db
        self.graph = graph

    def create(self, name: str, definition: str, terms: list[str] | None, document_id: str | None,
               location: str | None = None, author: str | None = None) -> dict[str, Any]:
        """A new definition; its ID is the slug of its name (409 when an active one has it)."""
        name = self._name(name)
        definition_id = slugify(name)
        if self.db.query_one("SELECT 1 FROM business_definitions WHERE definition_id = ? AND active = 1",
                             (definition_id,)):
            raise Conflict(f"a business definition with this name already exists: {definition_id}",
                           {"definition_id": definition_id})
        return self._save(definition_id, name, definition, terms, document_id, location, author, utcnow())

    def update(self, definition_id: str, name: str, definition: str, terms: list[str] | None,
               document_id: str | None, location: str | None = None, author: str | None = None) -> dict[str, Any]:
        """Edit a definition; its ID stays the same when the name changes."""
        row = self.db.query_one("SELECT created_at FROM business_definitions WHERE definition_id = ? AND active = 1",
                                (definition_id,))
        if row is None:
            raise NotFound(f"business definition not found: {definition_id}")
        return self._save(definition_id, self._name(name), definition, terms, document_id, location, author,
                          row["created_at"])

    def _save(self, definition_id: str, name: str, definition: str, terms: list[str] | None,
              document_id: str | None, location: str | None, author: str | None, created: str) -> dict[str, Any]:
        if definition is None or not definition.strip() or len(definition) > 5000:
            raise InvalidArgument("definition is required (max 5000 characters)")
        definition = definition.strip()
        resolved = self._terms(terms)
        document_id = self._document(document_id)
        location = " ".join((location or "").split())
        if len(location) > MAX_LOCATION:
            raise InvalidArgument(f"location must be at most {MAX_LOCATION} characters")
        if location and document_id is None:
            raise InvalidArgument("a location (page or section) needs the policy document it refers to")
        location = location or None
        author = (author or "").strip()[:200] or None
        graph = build_definition_context(definition_id, name, definition, author, document_id, location, resolved,
                                         created)
        # Raises GraphValidationError (HTTP 422) with violated shapes; nothing is written in that case.
        self.graph.publish(context_uri("definition", definition_id), graph, f"definition {name!r}")
        now = utcnow()
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO business_definitions(definition_id, name, definition_text, author, source_ref,
                       document_id, location, created_at, updated_at, active) VALUES (?,?,?,?,NULL,?,?,?,?,1)
                   ON CONFLICT(definition_id) DO UPDATE SET name=excluded.name,
                       definition_text=excluded.definition_text, author=excluded.author, source_ref=NULL,
                       document_id=excluded.document_id, location=excluded.location,
                       created_at=excluded.created_at, updated_at=excluded.updated_at, active=1""",
                (definition_id, name, definition, author, document_id, location, created, now))
            conn.execute("DELETE FROM definition_links WHERE definition_id = ?", (definition_id,))
            conn.executemany(
                "INSERT INTO definition_links(definition_id, target_type, target_label, target_slug) VALUES (?,?,?,?)",
                [(definition_id, "term", label, slug) for slug, label in resolved])
        return self.get(definition_id)

    @staticmethod
    def _name(name: str | None) -> str:
        name = " ".join((name or "").split())
        if not name or len(name) > 200 or not slugify(name):
            raise InvalidArgument("name is required (max 200 characters)")
        return name

    def _terms(self, terms: list[str] | None) -> list[tuple[str, str]]:
        """(slug, label) of the referenced business terms, given by slug or label; all must exist."""
        out: list[tuple[str, str]] = []
        unknown: list[str] = []
        for value in terms or []:
            value = " ".join(str(value).split())
            if not value:
                continue
            row = self.db.query_one("SELECT slug, label FROM business_terms WHERE deleted = 0 AND slug = ?", (value,)) \
                or self.db.query_one("SELECT slug, label FROM business_terms WHERE deleted = 0 AND LOWER(label) = LOWER(?)",
                                     (value,))
            if row is None:
                unknown.append(value)
            elif row["slug"] not in {s for s, _ in out}:
                out.append((row["slug"], row["label"]))
        if unknown:
            raise InvalidArgument("a business definition can only reference existing business terms",
                                  {"unknown_terms": unknown[:20]})
        if not out:
            raise InvalidArgument("a business definition must reference at least one business term")
        if len(out) > MAX_TERMS:
            raise InvalidArgument(f"a business definition can reference at most {MAX_TERMS} business terms")
        return out

    def _document(self, document_id: str | None) -> str | None:
        """The policy document (optional); when given, it must be an ingested, active document."""
        document_id = (document_id or "").strip()
        if not document_id:
            return None
        if self.db.query_one("SELECT 1 FROM documents WHERE document_id = ? AND active = 1", (document_id,)) is None:
            raise InvalidArgument(f"policy document not found: {document_id}", {"document_id": document_id})
        return document_id

    def get(self, definition_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM business_definitions WHERE definition_id = ? AND active = 1",
                                (definition_id,))
        if row is None:
            raise NotFound(f"business definition not found: {definition_id}")
        terms = self.db.query(
            "SELECT l.target_slug AS slug, COALESCE(t.label, l.target_label) AS label FROM definition_links l"
            " LEFT JOIN business_terms t ON t.slug = l.target_slug AND t.deleted = 0"
            " WHERE l.definition_id = ? AND l.target_type = 'term' ORDER BY LOWER(label)", (definition_id,))
        document = None
        if row.get("document_id"):
            doc = self.db.query_one("SELECT document_id, title, filename, source_uri FROM documents WHERE document_id = ?",
                                    (row["document_id"],))
            document = {"document_id": row["document_id"],
                        "title": (doc["title"] or doc["filename"] or doc["source_uri"]) if doc else None,
                        "ref": f"/api/documents/{row['document_id']}"}
        return {
            "definition_id": definition_id, "name": row["name"], "definition": row["definition_text"],
            "author": row["author"], "document": document, "location": row.get("location"),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "graph_ref": str(definition_uri(definition_id)), "ref": f"/api/business-definitions/{definition_id}",
            "terms": [{"slug": t["slug"], "label": t["label"], "graph_ref": f"urn:mesh:term:{t['slug']}",
                       "ref": f"/api/business-terms/{t['slug']}"} for t in terms],
        }

    def graph_for(self, definition_id: str):
        """The definition's graph context rebuilt from SQLite with the current term labels (None: not active).
        Used after a referenced business term is renamed."""
        row = self.db.query_one("SELECT * FROM business_definitions WHERE definition_id = ? AND active = 1",
                                (definition_id,))
        if row is None:
            return None
        terms = [(t["slug"], t["label"]) for t in self.get(definition_id)["terms"]]
        return build_definition_context(definition_id, row["name"], row["definition_text"], row["author"],
                                        row["document_id"], row.get("location"), terms, row["created_at"])

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT definition_id FROM business_definitions WHERE active = 1 ORDER BY name LIMIT ?",
                             (max(1, min(limit, 500)),))
        return [self.get(r["definition_id"]) for r in rows]

    def for_term(self, slug: str) -> list[dict[str, Any]]:
        return self.for_terms([slug])

    def for_terms(self, slugs: list[str]) -> list[dict[str, Any]]:
        """Active definitions that reference any of these business terms, once each, by name."""
        slugs = sorted(set(slugs))
        if not slugs:
            return []
        return [{"definition_id": r["definition_id"], "name": r["name"],
                 "ref": f"/api/business-definitions/{r['definition_id']}"}
                for r in self.db.query(
                    "SELECT DISTINCT d.definition_id, d.name FROM business_definitions d JOIN definition_links l"
                    " ON l.definition_id = d.definition_id WHERE d.active = 1 AND l.target_type = 'term'"
                    f" AND l.target_slug IN ({','.join('?' * len(slugs))}) ORDER BY d.name", tuple(slugs))]

    def delete(self, definition_id: str) -> dict[str, Any]:
        self.get(definition_id)
        self.graph.remove(context_uri("definition", definition_id))
        self.db.execute("UPDATE business_definitions SET active = 0, updated_at = ? WHERE definition_id = ?",
                        (utcnow(), definition_id))
        return {"definition_id": definition_id, "status": "deleted"}

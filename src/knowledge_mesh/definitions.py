"""Business definitions (FR-13): captured text linked to business entities and terms.

The graph is the gate: the definition context is SHACL-validated and published first, and
the SQLite record is written only if publication succeeds. A rejected definition therefore
leaves both stores exactly as they were.
"""

from __future__ import annotations

from typing import Any

from .common.errors import InvalidArgument, NotFound
from .common.ids import slugify, utcnow
from .graph.builders import build_definition_context
from .graph.ontology import context_uri, definition_uri
from .graph.store import KnowledgeGraph
from .storage.db import Database


def _clean_labels(labels: list[str] | None) -> list[str]:
    out: list[str] = []
    for label in labels or []:
        label = " ".join(str(label).split())[:120]
        if label and slugify(label) not in {slugify(x) for x in out}:
            out.append(label)
    return out[:50]


class DefinitionService:
    def __init__(self, db: Database, graph: KnowledgeGraph):
        self.db = db
        self.graph = graph

    def upsert(self, name: str, definition: str, entities: list[str] | None = None, terms: list[str] | None = None,
               author: str | None = None, source_ref: str | None = None) -> dict[str, Any]:
        name = " ".join((name or "").split())
        if not name or len(name) > 200:
            raise InvalidArgument("name is required (max 200 characters)")
        if definition is None or len(definition) > 5000:
            raise InvalidArgument("definition must be at most 5000 characters")
        definition = definition.strip()
        entities, terms = _clean_labels(entities), _clean_labels(terms)
        author = (author or "").strip()[:200] or None
        source_ref = (source_ref or "").strip()[:500] or None
        definition_id = slugify(name)
        existing = self.db.query_one("SELECT created_at FROM business_definitions WHERE definition_id = ?",
                                     (definition_id,))
        created = existing["created_at"] if existing else utcnow()
        graph = build_definition_context(definition_id, name, definition, author, source_ref, entities, terms, created)
        # Raises GraphValidationError (HTTP 422) with violated shapes; nothing is written in that case.
        self.graph.publish(context_uri("definition", definition_id), graph, f"definition {name!r}")
        now = utcnow()
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO business_definitions(definition_id, name, definition_text, author, source_ref,
                       created_at, updated_at, active) VALUES (?,?,?,?,?,?,?,1)
                   ON CONFLICT(definition_id) DO UPDATE SET name=excluded.name,
                       definition_text=excluded.definition_text, author=excluded.author,
                       source_ref=excluded.source_ref, updated_at=excluded.updated_at, active=1""",
                (definition_id, name, definition, author, source_ref, created, now))
            conn.execute("DELETE FROM definition_links WHERE definition_id = ?", (definition_id,))
            conn.executemany(
                "INSERT INTO definition_links(definition_id, target_type, target_label, target_slug) VALUES (?,?,?,?)",
                [(definition_id, "entity", e, slugify(e)) for e in entities]
                + [(definition_id, "term", t, slugify(t)) for t in terms])
        return self.get(definition_id)

    def get(self, definition_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM business_definitions WHERE definition_id = ? AND active = 1",
                                (definition_id,))
        if row is None:
            raise NotFound(f"business definition not found: {definition_id}")
        links = self.db.query("SELECT target_type, target_label, target_slug FROM definition_links "
                              "WHERE definition_id = ? ORDER BY target_type, target_label", (definition_id,))
        return {
            "definition_id": definition_id, "name": row["name"], "definition": row["definition_text"],
            "author": row["author"], "source_ref": row["source_ref"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], "graph_ref": str(definition_uri(definition_id)),
            "ref": f"/api/business-definitions/{definition_id}",
            "entities": [{"label": l["target_label"], "graph_ref": f"urn:mesh:entity:{l['target_slug']}",
                          "ref": f"/api/business-entities/{l['target_slug']}"}
                         for l in links if l["target_type"] == "entity"],
            "terms": [{"label": l["target_label"], "graph_ref": f"urn:mesh:term:{l['target_slug']}",
                       "ref": f"/api/business-terms/{l['target_slug']}"}
                      for l in links if l["target_type"] == "term"],
        }

    def list(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT definition_id FROM business_definitions WHERE active = 1 ORDER BY name LIMIT ?",
                             (max(1, min(limit, 500)),))
        return [self.get(r["definition_id"]) for r in rows]

    def delete(self, definition_id: str) -> dict[str, Any]:
        self.get(definition_id)
        self.graph.remove(context_uri("definition", definition_id))
        self.db.execute("UPDATE business_definitions SET active = 0, updated_at = ? WHERE definition_id = ?",
                        (utcnow(), definition_id))
        return {"definition_id": definition_id, "status": "deleted"}

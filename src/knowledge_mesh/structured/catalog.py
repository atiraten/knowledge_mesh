"""Structured metadata catalog in SQLite (FR-09).

Extraction is idempotent: dataset and column IDs derive from (source, schema, name), so a
re-run updates the same rows. Columns or datasets no longer present are retired (active=0)
together with their keys; their glossary vectors and graph assertions are retired by the
orchestrator when the source context is republished.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..common.errors import NotFound, SourceUnavailable
from ..common.ids import catalog_ref, column_id_for, dataset_id_for, key_id_for, sha256_text, utcnow
from ..jobs import JobContext
from ..observability import get_metrics, span
from ..storage.db import Database, dumps, loads
from .connectors import Connector, DatasetRef, DatasetSchema
from .connectors.base import coverage_note
from .fk_rules import FKRules, assess_foreign_key
from .reviews import ReviewStore, lookup

if TYPE_CHECKING:
    from .sources import SourceRegistry

log = logging.getLogger(__name__)


class CatalogService:
    def __init__(self, config, db: Database, registry: "SourceRegistry"):
        self.config = config
        self.db = db
        self.registry = registry
        for src in registry.all():
            self.db.execute(
                "INSERT INTO structured_sources(source_id, kind, status) VALUES (?,?,?) "
                "ON CONFLICT(source_id) DO UPDATE SET kind=excluded.kind",
                (src.id, src.kind, "configured"),
            )

    # -- sources ---------------------------------------------------------------------------
    def source_settings(self, source_id: str):
        return self.registry.require(source_id)

    def connector(self, source_id: str) -> Connector:
        return self.registry.connector(source_id)

    def list_sources(self) -> list[dict[str, Any]]:
        out = []
        for src in self.registry.all():
            row = self.db.query_one("SELECT * FROM structured_sources WHERE source_id = ?", (src.id,)) or {}
            counts = self.db.query_one(
                "SELECT COUNT(*) AS datasets, COUNT(DISTINCT schema_name) AS schemas FROM datasets "
                "WHERE source_id = ? AND active = 1", (src.id,))
            out.append({
                **self.registry.describe(src), "status": row.get("status"), "status_reason": row.get("status_reason"),
                "last_extracted_at": row.get("last_extracted_at"), "last_job_id": row.get("last_job_id"),
                "dataset_count": counts["datasets"], "schema_count": counts["schemas"],
                "ref": f"/api/structured-sources/{src.id}",
            })
        return out

    def retire_source(self, source_id: str) -> int:
        """Retire every dataset, column and key of a removed source (rows are kept for history)."""
        now = utcnow()
        ids = [r["dataset_id"] for r in self.db.query(
            "SELECT dataset_id FROM datasets WHERE source_id = ? AND active = 1", (source_id,))]
        with self.db.transaction() as conn:
            conn.execute("UPDATE datasets SET active = 0, retired_at = ? WHERE source_id = ? AND active = 1",
                         (now, source_id))
            for ds in ids:
                conn.execute("UPDATE columns SET active = 0, retired_at = ? WHERE dataset_id = ?", (now, ds))
                conn.execute("UPDATE keys SET active = 0, updated_at = ? WHERE dataset_id = ? OR referenced_dataset_id = ?",
                             (now, ds, ds))
            conn.execute("DELETE FROM structured_sources WHERE source_id = ?", (source_id,))
        return len(ids)

    def source_tree(self, source_id: str) -> dict[str, Any]:
        src = self.source_settings(source_id)
        datasets = self.db.query(
            "SELECT dataset_id, schema_name, name, kind, row_count, column_count, entity_label, entity_slug, catalog_ref "
            "FROM datasets WHERE source_id = ? AND active = 1 ORDER BY schema_name, name", (source_id,))
        schemas: dict[str, list] = {}
        for d in datasets:
            schemas.setdefault(d["schema_name"] or "", []).append({**d, "ref": f"/api/datasets/{d['dataset_id']}"})
        return {"source_id": src.id, "kind": src.kind,
                "schemas": [{"schema": s, "datasets": ds} for s, ds in schemas.items()]}

    def find_datasets(self, name_contains: str | None = None, column_contains: str | None = None,
                      source_id: str | None = None, limit: int = 25) -> dict[str, Any]:
        """Active datasets whose name/schema/entity label (or a column name/business term) contains the text."""
        where, params = ["d.active = 1"], []
        if source_id:
            where.append("d.source_id = ?")
            params.append(source_id)
        if name_contains:
            like = _like(name_contains)
            where.append("(LOWER(d.name) LIKE ? ESCAPE '\\' OR LOWER(COALESCE(d.schema_name, '')) LIKE ? ESCAPE '\\' "
                         "OR LOWER(COALESCE(d.entity_label, '')) LIKE ? ESCAPE '\\')")
            params += [like] * 3
        if column_contains:
            like = _like(column_contains)
            where.append("d.dataset_id IN (SELECT c.dataset_id FROM columns c WHERE c.active = 1 AND "
                         "(LOWER(c.name) LIKE ? ESCAPE '\\' OR LOWER(COALESCE(c.term_label, '')) LIKE ? ESCAPE '\\'))")
            params += [like] * 2
        cond = " AND ".join(where)
        total = self.db.scalar(f"SELECT COUNT(*) FROM datasets d WHERE {cond}", params)
        rows = self.db.query(
            "SELECT d.dataset_id, d.source_id, d.schema_name, d.name, d.kind, d.row_count, d.column_count, "
            f"d.entity_label, d.catalog_ref FROM datasets d WHERE {cond} "
            "ORDER BY d.source_id, d.schema_name, d.name LIMIT ?", [*params, max(1, min(int(limit), 100))])
        return {"total": total, "datasets": [{**r, "ref": f"/api/datasets/{r['dataset_id']}"} for r in rows]}

    # -- extraction --------------------------------------------------------------------------
    def extract(self, source_id: str, datasets: list[str] | None = None,
                job: JobContext | None = None) -> dict[str, Any]:
        metrics = get_metrics()
        with span("catalog.extract", "TOOL", {"catalog.source_id": source_id}) as sp:
            try:
                connector = self.connector(source_id)
            except SourceUnavailable as exc:
                self._set_source_status(source_id, "unavailable", exc.message, job)
                raise
            try:
                refs = connector.list_datasets()
                if datasets:
                    wanted = set(datasets)
                    missing = wanted - {r.name for r in refs} - {r.qualified for r in refs}
                    refs = [r for r in refs if r.name in wanted or r.qualified in wanted]
                    for name in sorted(missing):
                        if job:
                            job.add_item(name, "failed", "dataset not found or not allowed in source")
                seen: list[str] = []
                changed: list[str] = []
                for ref in refs:
                    try:
                        schema = connector.describe(ref)
                        dataset_id, was_changed = self._upsert_dataset(source_id, schema, job.job_id if job else None)
                        seen.append(dataset_id)
                        if was_changed:
                            changed.append(dataset_id)
                        metrics.datasets.add(1, {"source_id": source_id})
                        if job:
                            job.add_item(dataset_id, "succeeded", "schema changed" if was_changed else "unchanged",
                                         dataset=ref.qualified, columns=len(schema.columns),
                                         declared_keys=len(schema.keys))
                    except Exception as exc:  # one dataset failure does not stop the job
                        log.warning("dataset extraction failed",
                                    extra={"source_id": source_id, "dataset": ref.qualified,
                                           "error": type(exc).__name__})
                        if job:
                            job.add_item(ref.qualified, "failed", f"{type(exc).__name__}: {str(exc)[:200]}")
                retired = 0
                if not datasets:
                    retired = self._retire_missing_datasets(source_id, seen)
                note = None if datasets else coverage_note(self.source_settings(source_id), connector, len(refs))
                if not refs and not datasets:
                    # Zero datasets is not a success: say why, on the job and on the source.
                    if job:
                        job.add_item(source_id, "failed", note or "No tables or views found")
                    self._set_source_status(source_id, "empty", note, job)
                else:
                    if note and job:
                        job.add_item(f"schemas:{source_id}", "skipped", note)
                    self._set_source_status(source_id, "cataloged", note, job)
            finally:
                connector.close()
            sp.set_attribute("catalog.datasets", len(seen))
            sp.set_attribute("catalog.retired", retired)
        return {"source_id": source_id, "dataset_ids": seen, "changed_dataset_ids": changed,
                "retired_datasets": retired,
                "catalog_refs": [f"/api/datasets/{d}" for d in seen[:50]]}

    def _set_source_status(self, source_id: str, status: str, reason: str | None, job: JobContext | None) -> None:
        self.db.execute(
            "UPDATE structured_sources SET status = ?, status_reason = ?, last_extracted_at = ?, last_job_id = ? "
            "WHERE source_id = ?", (status, reason, utcnow(), job.job_id if job else None, source_id))

    def _upsert_dataset(self, source_id: str, schema: DatasetSchema, job_id: str | None) -> tuple[str, bool]:
        ref = schema.ref
        dataset_id = dataset_id_for(source_id, ref.schema, ref.name)
        schema_hash = sha256_text(dumps([(c.name, c.data_type, c.nullable) for c in schema.columns]
                                        + [(k.key_type, k.columns, k.ref_table, k.ref_columns) for k in schema.keys]))
        now = utcnow()
        with self.db.transaction() as conn:
            prev = conn.execute("SELECT schema_hash, active FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
            changed = prev is None or prev["schema_hash"] != schema_hash or not prev["active"]
            conn.execute(
                """INSERT INTO datasets(dataset_id, source_id, schema_name, name, kind, catalog_ref, physical_ref,
                       row_count, column_count, schema_hash, active, first_seen_at, last_seen_at, last_job_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?,?)
                   ON CONFLICT(dataset_id) DO UPDATE SET kind=excluded.kind, physical_ref=excluded.physical_ref,
                       row_count=excluded.row_count, column_count=excluded.column_count,
                       schema_hash=excluded.schema_hash, active=1, retired_at=NULL,
                       last_seen_at=excluded.last_seen_at, last_job_id=excluded.last_job_id""",
                (dataset_id, source_id, ref.schema, ref.name, ref.kind, catalog_ref(source_id, ref.schema, ref.name),
                 schema.physical_ref, schema.row_count, len(schema.columns), schema_hash, now, now, job_id),
            )
            present = []
            for col in schema.columns:
                cid = column_id_for(dataset_id, col.name)
                present.append(cid)
                conn.execute(
                    """INSERT INTO columns(column_id, dataset_id, name, ordinal, data_type, nullable, catalog_ref,
                           active, first_seen_at, last_seen_at) VALUES (?,?,?,?,?,?,?,1,?,?)
                       ON CONFLICT(column_id) DO UPDATE SET ordinal=excluded.ordinal, data_type=excluded.data_type,
                           nullable=excluded.nullable, active=1, retired_at=NULL, last_seen_at=excluded.last_seen_at""",
                    (cid, dataset_id, col.name, col.ordinal, col.data_type,
                     None if col.nullable is None else int(col.nullable),
                     catalog_ref(source_id, ref.schema, ref.name, col.name), now, now),
                )
            marks = ",".join("?" * len(present)) or "''"
            conn.execute(
                f"UPDATE columns SET active = 0, retired_at = ? WHERE dataset_id = ? AND active = 1 "
                f"AND column_id NOT IN ({marks})", (now, dataset_id, *present))
            # declared keys: replace the declared set; inferred keys on retired columns are retired.
            declared_ids = []
            for key in schema.keys:
                ref_ds = dataset_id_for(source_id, key.ref_schema, key.ref_table) if key.key_type == "foreign" else None
                kid = key_id_for(dataset_id, key.key_type, "declared", key.columns, ref_ds, key.ref_columns)
                declared_ids.append(kid)
                conn.execute(
                    """INSERT INTO keys(key_id, dataset_id, key_type, origin, constraint_name, columns_json,
                           referenced_dataset_id, referenced_columns_json, is_pk_candidate, active, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,1,?,?)
                       ON CONFLICT(key_id) DO UPDATE SET constraint_name=excluded.constraint_name, active=1,
                           updated_at=excluded.updated_at""",
                    (kid, dataset_id, key.key_type, "declared", key.name, dumps(key.columns), ref_ds,
                     dumps(key.ref_columns) if key.ref_columns else None, int(key.key_type == "primary"), now, now),
                )
            kmarks = ",".join("?" * len(declared_ids)) or "''"
            conn.execute(
                f"UPDATE keys SET active = 0, updated_at = ? WHERE dataset_id = ? AND origin = 'declared' "
                f"AND key_id NOT IN ({kmarks})", (now, dataset_id, *declared_ids))
            names = {c.name for c in schema.columns}
            for k in conn.execute("SELECT key_id, columns_json FROM keys WHERE dataset_id = ? AND origin = 'inferred' "
                                  "AND active = 1", (dataset_id,)).fetchall():
                if not set(loads(k["columns_json"], [])) <= names:
                    conn.execute("UPDATE keys SET active = 0, updated_at = ? WHERE key_id = ?", (now, k["key_id"]))
        return dataset_id, changed

    def _retire_missing_datasets(self, source_id: str, seen: list[str]) -> int:
        now = utcnow()
        marks = ",".join("?" * len(seen)) or "''"
        stale = [r["dataset_id"] for r in self.db.query(
            f"SELECT dataset_id FROM datasets WHERE source_id = ? AND active = 1 AND dataset_id NOT IN ({marks})",
            (source_id, *seen))]
        for ds in stale:
            with self.db.transaction() as conn:
                conn.execute("UPDATE datasets SET active = 0, retired_at = ? WHERE dataset_id = ?", (now, ds))
                conn.execute("UPDATE columns SET active = 0, retired_at = ? WHERE dataset_id = ?", (now, ds))
                conn.execute("UPDATE keys SET active = 0, updated_at = ? WHERE dataset_id = ? OR referenced_dataset_id = ?",
                             (now, ds, ds))
        return len(stale)

    # -- reads ---------------------------------------------------------------------------------
    def dataset_row(self, dataset_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        if row is None:
            raise NotFound(f"dataset not found: {dataset_id}")
        return row

    def dataset_ref(self, row: dict[str, Any]) -> DatasetRef:
        return DatasetRef(row["schema_name"], row["name"], row["kind"] or "table")

    def active_columns(self, dataset_id: str) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM columns WHERE dataset_id = ? AND active = 1 ORDER BY ordinal", (dataset_id,))

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        d = self.dataset_row(dataset_id)
        columns = self.active_columns(dataset_id)
        profiles = {p["column_id"]: p for p in self.db.query(
            "SELECT p.* FROM column_profiles p JOIN profiling_runs r ON r.run_id = p.run_id "
            "WHERE p.dataset_id = ? ORDER BY r.started_at", (dataset_id,))}
        keys = self.db.query("SELECT * FROM keys WHERE dataset_id = ? AND active = 1 ORDER BY origin, key_type",
                             (dataset_id,))
        incoming = self.db.query("SELECT * FROM keys WHERE referenced_dataset_id = ? AND active = 1", (dataset_id,))
        names = {r["dataset_id"]: r for r in self.db.query("SELECT dataset_id, name, catalog_ref FROM datasets")}
        rules = FKRules.from_settings(self.config.profiling)
        decisions = ReviewStore(self.db).all()
        definitions = []
        if d.get("entity_slug"):
            definitions = self._definitions_for("entity", d["entity_slug"])

        def key_view(k: dict[str, Any]) -> dict[str, Any]:
            view = self._key_with_confidence(k, names, rules)
            view["status"] = ("verified" if k["origin"] == "declared" else
                              lookup(decisions, "foreign_key", k["key_id"]) if k["key_type"] == "foreign" else "inferred")
            return view

        def column_view(c: dict[str, Any]) -> dict[str, Any]:
            view = self._column_view(c, profiles.get(c["column_id"]))
            if view["business_term"]:
                view["business_term"]["status"] = lookup(decisions, "term_mapping", c["column_id"], c["term_slug"])
            return view

        entity = None
        if d.get("entity_label"):
            entity = {"label": d["entity_label"], "slug": d["entity_slug"], "method": d["entity_method"],
                      "graph_ref": f"urn:mesh:entity:{d['entity_slug']}", "ref": f"/api/business-entities/{d['entity_slug']}",
                      "status": lookup(decisions, "entity_mapping", dataset_id, d["entity_slug"])}
        return {
            "dataset_id": dataset_id, "source_id": d["source_id"], "schema": d["schema_name"], "name": d["name"],
            "kind": d["kind"], "catalog_ref": d["catalog_ref"], "physical_ref": d["physical_ref"],
            "row_count": d["row_count"], "active": bool(d["active"]), "last_seen_at": d["last_seen_at"],
            "business_entity": entity,
            "definitions": definitions,
            "columns": [column_view(c) for c in columns],
            "keys": [key_view(k) for k in keys],
            "referenced_by": [key_view(k) for k in incoming],
        }

    def _definitions_for(self, target_type: str, slug: str) -> list[dict[str, Any]]:
        return [{"definition_id": r["definition_id"], "name": r["name"],
                 "ref": f"/api/business-definitions/{r['definition_id']}"}
                for r in self.db.query(
                    "SELECT d.definition_id, d.name FROM business_definitions d JOIN definition_links l "
                    "ON l.definition_id = d.definition_id WHERE d.active = 1 AND l.target_type = ? AND l.target_slug = ?",
                    (target_type, slug))]

    @staticmethod
    def _column_view(c: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, Any]:
        view = {
            "column_id": c["column_id"], "name": c["name"], "ordinal": c["ordinal"], "data_type": c["data_type"],
            "nullable": None if c["nullable"] is None else bool(c["nullable"]), "catalog_ref": c["catalog_ref"],
            "business_term": None,
        }
        if c.get("term_label") and c.get("term_status") == "generated":
            view["business_term"] = {
                "label": c["term_label"], "slug": c["term_slug"], "description": c["term_description"],
                "explanation": c["term_explanation"], "graph_ref": f"urn:mesh:term:{c['term_slug']}",
                # can identify a person (PII / PHI / GDPR); None when not assessed (terms generated before the flag)
                "sensitive": None if c.get("term_sensitive") is None else bool(c["term_sensitive"]),
                "ref": f"/api/business-terms/{c['term_slug']}",
            }
        elif c.get("term_status") == "failed":
            view["business_term_error"] = c.get("term_error")
        if profile:
            view["profile"] = {
                "run_id": profile["run_id"], "row_count": profile["row_count"],
                "distinct_count": profile["distinct_count"], "empty_count": profile["empty_count"],
                "null_count": profile["null_count"], "content_type": profile["content_type"],
                "type_category": profile.get("type_category"),  # generic type; null for profiles saved before it existed
                "format_patterns": loads(profile["format_patterns_json"], []),
                "most_common_values": loads(profile["most_common_values_json"], []),
            }
        return view

    def _key_with_confidence(self, k: dict[str, Any], names: dict[str, dict], rules: FKRules) -> dict[str, Any]:
        view = self._key_view(k, names)
        confidence = assess_foreign_key(self.db, k, rules)  # inferred FKs only; from the current terms
        if confidence:
            view["confidence"] = confidence
        return view

    @staticmethod
    def _key_view(k: dict[str, Any], names: dict[str, dict]) -> dict[str, Any]:
        out = {
            "key_id": k["key_id"], "dataset_id": k["dataset_id"], "key_type": k["key_type"], "origin": k["origin"],
            "columns": loads(k["columns_json"], []), "constraint_name": k["constraint_name"],
            "primary_key_candidate": bool(k["is_pk_candidate"]), "graph_ref": f"urn:mesh:key:{k['key_id']}",
        }
        if k["origin"] == "inferred":
            out["profiling_run_id"] = k["profiling_run_id"]
            out["scope"] = loads(k["scope_json"], {})
            out["metrics"] = loads(k["metrics_json"], {})
        if k["key_type"] == "foreign":
            ref = names.get(k["referenced_dataset_id"]) or {}
            out["referenced_dataset_id"] = k["referenced_dataset_id"]
            out["referenced_dataset"] = ref.get("name")
            out["referenced_columns"] = loads(k["referenced_columns_json"], [])
            out["mapping"] = [{"from": a, "to": b} for a, b in zip(out["columns"], out["referenced_columns"])]
        return out

    def get_column(self, column_id: str) -> dict[str, Any]:
        c = self.db.query_one("SELECT * FROM columns WHERE column_id = ?", (column_id,))
        if c is None:
            raise NotFound(f"column not found: {column_id}")
        return {**self._column_view(c, None), "dataset_id": c["dataset_id"], "active": bool(c["active"]),
                "dataset_ref": f"/api/datasets/{c['dataset_id']}"}

    def entity_labels(self) -> list[str]:
        return [r["entity_label"] for r in self.db.query(
            "SELECT DISTINCT entity_label FROM datasets WHERE active = 1 AND entity_label IS NOT NULL")]


def _like(text: str) -> str:
    """Case-insensitive LIKE pattern for a substring, with %, _ and the escape character escaped."""
    escaped = text.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"

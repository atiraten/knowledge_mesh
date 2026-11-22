"""Named relationships, term relations and term synonyms proposed at catalog time.

For each dataset, the term-generation model receives (``prompts/relationship_generation.txt``):

* the table, its business entity and its business terms (with descriptions);
* the foreign keys where the table is the referencing side, limited to declared FKs and inferred FKs
  of at least ``relationship_generation.min_fk_confidence``, between tables of different entities.

It returns JSON with three lists, validated before anything is stored:

* ``relationships``: one named, directional relationship per FK (``Customer placesOrder Order``) with
  a definition, synonyms, an inverse name and a cardinality. The FK becomes the relationship's
  explicit join mapping;
* ``term_relations``: relations between the table's own terms, restricted to ``expectedAfter``,
  ``derivedFrom``, ``equivalentTo``, ``dependsOn`` and ``isPartOf``. Descriptive only: they are never
  checked against records;
* ``term_synonyms``: alternate labels for the table's terms (``skos:altLabel``).

Only catalog metadata is sent, never row values. Everything proposed starts with status ``inferred``
(see ``structured/reviews.py``). A successful run replaces the dataset's earlier proposals; a failed
run keeps them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from ..agent.plan import parse_json_object
from ..common.errors import NotConfigured, NotFound
from ..common.ids import new_id, sha256_text, slugify, stable_hash, utcnow
from ..graph.ontology import TERM_PREDICATES
from ..jobs import JobContext
from ..observability import span
from ..storage.db import Database, dumps, loads
from .fk_rules import FKRules, assess_foreign_key
from .glossary_store import columns_select, datasets_select
from .reviews import ReviewStore, lookup

log = logging.getLogger(__name__)
_LEVELS = {"low": 0, "medium": 1, "high": 2}
CARDINALITIES = ("one-to-one", "one-to-many", "many-to-one", "many-to-many")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,59}$")


class _RelationshipOut(BaseModel):
    fk: str
    source_entity: str
    name: str
    target_entity: str
    definition: str = ""
    synonyms: list[str] = Field(default_factory=list)
    inverse_name: str | None = None
    cardinality: str | None = None


class _TermRelationOut(BaseModel):
    subject: str
    predicate: str
    object: str
    definition: str = ""


class _SynonymsOut(BaseModel):
    term: str
    synonyms: list[str] = Field(default_factory=list)


class _Output(BaseModel):
    relationships: list[_RelationshipOut] = Field(default_factory=list)
    term_relations: list[_TermRelationOut] = Field(default_factory=list)
    term_synonyms: list[_SynonymsOut] = Field(default_factory=list)


def relationship_id_for(source_slug: str, name: str, target_slug: str) -> str:
    return "rel-" + stable_hash(source_slug, slugify(name), target_slug)


def relation_id_for(subject_slug: str, predicate: str, object_slug: str) -> str:
    return "trel-" + stable_hash(subject_slug, predicate, object_slug)


def _clean(text: Any, limit: int) -> str:
    return " ".join(str(text or "").replace("\x00", "").split())[:limit]


class RelationshipGenerationService:
    def __init__(self, settings, term_settings, db: Database, llm_factory: Callable[[], Any] | None,
                 fk_rules: FKRules | None = None):
        self.cfg = settings
        self.term_cfg = term_settings
        self.db = db
        self.llm_factory = llm_factory
        self.fk_rules = fk_rules or FKRules()
        self._llm = None

    @property
    def template(self) -> str:
        path = self.cfg.prompt_path
        if not path or not Path(path).exists():
            raise NotConfigured(f"relationship generation prompt not found: {path}")
        return Path(path).read_text(encoding="utf-8")

    def available(self) -> tuple[bool, str | None]:
        if not self.cfg.enabled:
            return False, "relationship_generation.enabled is false"
        if self.llm_factory is None:
            return False, "term_generation provider/model not configured"
        try:
            self.template
            self.llm()
        except NotConfigured as exc:
            return False, exc.message
        return True, None

    def llm(self):
        if self._llm is None:
            if self.llm_factory is None:
                raise NotConfigured("term generation model is not configured")
            self._llm = self.llm_factory()
        return self._llm

    # -- input -------------------------------------------------------------------------------------
    def _context(self, dataset_id: str) -> dict[str, Any]:
        active_dataset = datasets_select(self.db) + " WHERE d.dataset_id = ? AND d.active = 1"
        d = self.db.query_one(active_dataset, (dataset_id,))
        if d is None:
            raise NotFound(f"dataset {dataset_id} not found or retired")
        decisions = ReviewStore(self.db).all()
        columns = self.db.query(columns_select(self.db) + " WHERE c.dataset_id = ? AND c.active = 1 ORDER BY c.ordinal",
                                (dataset_id,))
        terms = {}
        for c in columns:
            if c.get("term_label") and c.get("term_status") == "generated" \
                    and lookup(decisions, "term_mapping", c["column_id"], c["term_slug"]) != "rejected":
                terms.setdefault(c["term_slug"], {"label": c["term_label"], "description": c["term_description"],
                                                  "columns": []})["columns"].append(c["name"])
        entity = d["entity_label"] if d.get("entity_label") and lookup(
            decisions, "entity_mapping", dataset_id, d["entity_slug"]) != "rejected" else None
        fks = []
        min_level = _LEVELS[self.cfg.min_fk_confidence]
        rows = self.db.query("SELECT * FROM keys WHERE dataset_id = ? AND key_type = 'foreign' AND active = 1 "
                             "ORDER BY origin, key_id", (dataset_id,))
        declared = {(k["referenced_dataset_id"], k["columns_json"], k["referenced_columns_json"])
                    for k in rows if k["origin"] == "declared"}
        for k in rows:
            if k["origin"] == "inferred" and (k["referenced_dataset_id"], k["columns_json"],
                                              k["referenced_columns_json"]) in declared:
                continue  # profiling confirmed a declared key: the same join, proposed once
            ref = self.db.query_one(active_dataset, (k["referenced_dataset_id"],))
            if ref is None or not entity:
                continue
            ref_entity = ref["entity_label"] if ref.get("entity_label") and lookup(
                decisions, "entity_mapping", ref["dataset_id"], ref["entity_slug"]) != "rejected" else None
            if not ref_entity or ref["entity_slug"] == d["entity_slug"]:
                continue
            if k["origin"] == "inferred":
                if lookup(decisions, "foreign_key", k["key_id"]) == "rejected":
                    continue
                grade = assess_foreign_key(self.db, k, self.fk_rules)
                if grade is None or _LEVELS[grade["level"]] < min_level:
                    continue
                status = f"inferred, {grade['level']} confidence"
            else:
                status = "declared"
            cols, ref_cols = loads(k["columns_json"], []), loads(k["referenced_columns_json"], [])
            fks.append({"ref": f"FK{len(fks) + 1}", "key_id": k["key_id"], "entity": entity,
                        "ref_entity": ref_entity, "entity_slug": d["entity_slug"],
                        "ref_entity_slug": ref["entity_slug"], "status": status,
                        "text": f"{d['name']}({', '.join(cols)}) -> {ref['name']}({', '.join(ref_cols)})"})
        return {"dataset": d, "entity": entity, "terms": terms, "fks": fks}

    def render(self, ctx: dict[str, Any]) -> str:
        d = ctx["dataset"]
        term_lines = [f"- {t['label']} (columns: {', '.join(t['columns'])})"
                      + (f": {_clean(t['description'], 300)}" if t.get("description") else "")
                      for t in ctx["terms"].values()] or ["(none)"]
        fk_lines = [f"- {f['ref']}: {f['text']}; this table's entity: {f['entity']}; referenced table's entity: "
                    f"{f['ref_entity']}; {f['status']}" for f in ctx["fks"]] or ["(none)"]
        values = {
            "table": d["name"], "entity": ctx["entity"] or "(unknown)", "terms": "\n".join(term_lines),
            "foreign_keys": "\n".join(fk_lines), "predicates": ", ".join(TERM_PREDICATES),
            "max_synonyms": str(self.cfg.max_synonyms), "language_name": self.term_cfg.language_name,
            "industry_name": self.term_cfg.industry_name,
        }
        text = self.template
        for key, value in values.items():
            text = text.replace("{" + key + "}", value)
        return text

    # -- output ------------------------------------------------------------------------------------
    def parse(self, text: str, ctx: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
        """Validated proposals and the issues found (dropped items)."""
        issues: list[str] = []
        try:
            out = _Output.model_validate(parse_json_object(text))
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"unusable model output: {str(exc)[:200]}") from exc
        fks = {f["ref"]: f for f in ctx["fks"]}
        terms = {slug: t["label"] for slug, t in ctx["terms"].items()}
        # the model answers with labels; a label edited in the business glossary keeps its slug
        term_slug = {**{s: s for s in terms}, **{slugify(label): s for s, label in terms.items()}}
        rels, term_rels, synonyms = [], [], []
        seen: set[tuple[str, str]] = set()
        for r in out.relationships:
            fk = fks.get(r.fk.strip())
            if fk is None:
                issues.append(f"relationship for unknown foreign key {r.fk!r} dropped")
                continue
            ends = {slugify(fk["entity"]): fk["entity_slug"], slugify(fk["ref_entity"]): fk["ref_entity_slug"]}
            pair = {fk["entity_slug"], fk["ref_entity_slug"]}
            src, tgt = ends.get(slugify(r.source_entity)), ends.get(slugify(r.target_entity))
            if {src, tgt} != pair or src == tgt:
                issues.append(f"{r.fk}: entities must be {fk['entity']} and {fk['ref_entity']}")
                continue
            name = _clean(r.name, 60)
            if not _NAME.match(name):
                issues.append(f"{r.fk}: invalid relationship name {r.name!r}")
                continue
            labels = {fk["entity_slug"]: fk["entity"], fk["ref_entity_slug"]: fk["ref_entity"]}
            rid = relationship_id_for(src, name, tgt)
            if (rid, fk["key_id"]) in seen:
                continue
            seen.add((rid, fk["key_id"]))
            inverse = _clean(r.inverse_name, 60) or None
            if inverse and not _NAME.match(inverse):
                inverse = None
            card = (r.cardinality or "").strip().lower() or None
            rels.append({
                "relationship_id": rid, "key_id": fk["key_id"], "name": name,
                "key_direction": "forward" if src == fk["entity_slug"] else "reverse",
                "source_entity_slug": src, "source_entity_label": labels[src],
                "target_entity_slug": tgt, "target_entity_label": labels[tgt],
                "definition": _clean(r.definition, 1000) or None,
                "synonyms": self._synonyms(r.synonyms, {slugify(name)}),
                "inverse_name": inverse, "cardinality": card if card in CARDINALITIES else None,
            })
        for tr in out.term_relations:
            subj, obj, pred = term_slug.get(slugify(tr.subject)), term_slug.get(slugify(tr.object)), tr.predicate.strip()
            if pred not in TERM_PREDICATES:
                issues.append(f"term relation with unknown predicate {tr.predicate!r} dropped")
                continue
            if subj not in terms or obj not in terms or subj == obj:
                issues.append(f"term relation {tr.subject!r} {pred} {tr.object!r}: both must be different terms "
                              "of this table")
                continue
            term_rels.append({"relation_id": relation_id_for(subj, pred, obj), "subject_slug": subj,
                              "subject_label": terms[subj], "predicate": pred, "object_slug": obj,
                              "object_label": terms[obj], "definition": _clean(tr.definition, 1000) or None})
        for s in out.term_synonyms:
            slug = term_slug.get(slugify(s.term))
            if slug not in terms:
                issues.append(f"synonyms for unknown term {s.term!r} dropped")
                continue
            for syn in self._synonyms(s.synonyms, {slug}):
                synonyms.append({"term_slug": slug, "synonym": syn})
        return {"relationships": rels, "term_relations": term_rels, "term_synonyms": synonyms}, issues

    def _synonyms(self, values: list[str], exclude: set[str]) -> list[str]:
        out: list[str] = []
        for v in values or []:
            v = _clean(v, 60)
            if v and slugify(v) not in exclude and slugify(v) not in {slugify(x) for x in out}:
                out.append(v)
        return out[: self.cfg.max_synonyms]

    # -- run ---------------------------------------------------------------------------------------
    def generate_for_dataset(self, dataset_id: str, job: JobContext | None = None) -> dict[str, Any]:
        ctx = self._context(dataset_id)
        d = ctx["dataset"]
        if not ctx["fks"] and len(ctx["terms"]) == 0:
            if job:
                job.add_item(f"relationships:{dataset_id}", "skipped", "no foreign keys or business terms",
                             dataset=d["name"])
            return {"dataset_id": dataset_id, "status": "skipped", "reason": "no foreign keys or business terms"}
        template = self.template
        version = sha256_text(template)[:12]
        llm = self.llm()
        run_id = new_id("rels")
        with span("relationships.generate", "CHAIN", {"dataset.id": dataset_id, "relationships.fks": len(ctx["fks"]),
                                                       "relationships.terms": len(ctx["terms"]),
                                                       "llm.prompt_template.version": version}) as sp:
            prompt = self.render(ctx)
            response = llm.generate(prompt, purpose="relationship_generation", json_output=True,
                                    prompt_version=version)
            try:
                proposals, issues = self.parse(response.text, ctx)
            except ValueError as exc:
                self._record_run(run_id, job, dataset_id, llm, version, "failed", str(exc), 0, 0, 0)
                sp.set_attribute("relationships.status", "failed")
                if job:
                    job.add_item(f"relationships:{dataset_id}", "failed", str(exc), dataset=d["name"])
                return {"dataset_id": dataset_id, "run_id": run_id, "status": "failed", "error": str(exc)}
            self._store(d, run_id, proposals)
            counts = {k: len(v) for k, v in proposals.items()}
            self._record_run(run_id, job, dataset_id, llm, version, "partial" if issues else "succeeded",
                             "; ".join(issues)[:2000] or None, counts["relationships"], counts["term_relations"],
                             counts["term_synonyms"])
            sp.set_attribute("relationships.proposed", counts["relationships"])
            sp.set_attribute("relationships.term_relations", counts["term_relations"])
            sp.set_attribute("relationships.synonyms", counts["term_synonyms"])
        if job:
            job.add_item(f"relationships:{dataset_id}", "succeeded", "; ".join(issues)[:500] or None,
                         dataset=d["name"], **counts)
        return {"dataset_id": dataset_id, "run_id": run_id, "status": "partial" if issues else "succeeded",
                **counts, "issues": issues[:20]}

    def _store(self, d: dict[str, Any], run_id: str, p: dict[str, list[dict[str, Any]]]) -> None:
        now, ds, src = utcnow(), d["dataset_id"], d["source_id"]
        with self.db.transaction() as conn:
            for table in ("semantic_relationships", "term_relations", "term_synonyms"):
                conn.execute(f"UPDATE {table} SET active = 0 WHERE dataset_id = ?", (ds,))
            for r in p["relationships"]:
                conn.execute(
                    "INSERT OR REPLACE INTO semantic_relationships(relationship_id, dataset_id, source_id, key_id,"
                    " key_direction, name, source_entity_slug, source_entity_label, target_entity_slug,"
                    " target_entity_label, definition, synonyms_json, inverse_name, cardinality, run_id, active,"
                    " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?)",
                    (r["relationship_id"], ds, src, r["key_id"], r["key_direction"], r["name"],
                     r["source_entity_slug"], r["source_entity_label"], r["target_entity_slug"],
                     r["target_entity_label"], r["definition"], dumps(r["synonyms"]), r["inverse_name"],
                     r["cardinality"], run_id, now))
            for r in p["term_relations"]:
                conn.execute(
                    "INSERT OR REPLACE INTO term_relations(relation_id, dataset_id, source_id, subject_slug,"
                    " subject_label, predicate, object_slug, object_label, definition, run_id, active, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,1,?)",
                    (r["relation_id"], ds, src, r["subject_slug"], r["subject_label"], r["predicate"],
                     r["object_slug"], r["object_label"], r["definition"], run_id, now))
            for r in p["term_synonyms"]:
                conn.execute("INSERT OR REPLACE INTO term_synonyms(term_slug, synonym, dataset_id, source_id, run_id,"
                             " active) VALUES (?,?,?,?,?,1)", (r["term_slug"], r["synonym"], ds, src, run_id))

    def _record_run(self, run_id, job, dataset_id, llm, version, status, error, rels, term_rels, syns) -> None:
        self.db.execute(
            "INSERT INTO relationship_runs(run_id, job_id, dataset_id, provider, model, prompt_version, status, error,"
            " relationships, term_relations, synonyms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, job.job_id if job else None, dataset_id, getattr(llm, "provider", None),
             getattr(llm, "model", None), version, status, error, rels, term_rels, syns, utcnow()))

    # -- read ----------------------------------------------------------------------------------------
    def for_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Proposals touching a dataset (its own and those of FKs referencing it), with review status."""
        d = self.db.query_one("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
        if d is None:
            raise NotFound(f"dataset not found: {dataset_id}")
        decisions = ReviewStore(self.db).all()
        rels = self.db.query(
            "SELECT r.* FROM semantic_relationships r WHERE r.active = 1 AND (r.dataset_id = ? OR r.key_id IN "
            "(SELECT key_id FROM keys WHERE referenced_dataset_id = ? AND active = 1)) ORDER BY r.name",
            (dataset_id, dataset_id))
        grouped: dict[str, dict[str, Any]] = {}
        for r in rels:
            g = grouped.setdefault(r["relationship_id"], {
                "relationship_id": r["relationship_id"], "name": r["name"],
                "source_entity": r["source_entity_label"], "target_entity": r["target_entity_label"],
                "definition": r["definition"], "synonyms": loads(r["synonyms_json"], []),
                "inverse_name": r["inverse_name"], "cardinality": r["cardinality"], "run_id": r["run_id"],
                "graph_ref": f"urn:mesh:relationship:sem-{r['relationship_id']}",
                "status": lookup(decisions, "relationship", r["relationship_id"]), "foreign_keys": []})
            g["foreign_keys"].append({"key_id": r["key_id"], "direction": r["key_direction"]})
        term_rels = [{**{k: r[k] for k in ("relation_id", "subject_label", "predicate", "object_label",
                                           "definition", "run_id")},
                      "graph_ref": f"urn:mesh:term-relation:{r['relation_id']}",
                      "status": lookup(decisions, "term_relation", r["relation_id"])}
                     for r in self.db.query("SELECT * FROM term_relations WHERE dataset_id = ? AND active = 1 "
                                            "ORDER BY subject_label", (dataset_id,))]
        synonyms: dict[str, list[str]] = {}
        for r in self.db.query("SELECT term_slug, synonym FROM term_synonyms WHERE dataset_id = ? AND active = 1 "
                               "ORDER BY term_slug, synonym", (dataset_id,)):
            synonyms.setdefault(r["term_slug"], []).append(r["synonym"])
        last = self.db.query_one("SELECT * FROM relationship_runs WHERE dataset_id = ? ORDER BY created_at DESC "
                                 "LIMIT 1", (dataset_id,))
        return {"dataset_id": dataset_id, "relationships": list(grouped.values()), "term_relations": term_rels,
                "term_synonyms": synonyms, "last_run": last}

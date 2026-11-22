"""Profiling run: column profiles, unique keys (HyUCC/Apriori) and FK candidates (BINDER).

FK candidates are inclusions between identifier columns only (``fk_rules``): dates, amounts and
free text are not offered to BINDER. Each inferred FK records why each side is an identifier
(``identifier_basis``); its confidence is computed when it is read, from the current business terms.

Profiling scope (recorded on every inferred key): rows evaluated, total rows when known,
whether the result came from a sample, method (``head`` = first ``max_rows`` rows, ``full``),
and the profiling run ID. Raw values exist only in memory during the run.

Primary keys: a table with a declared primary key gets no PK candidate. When FKs target primary keys
(``fk_reference_primary_keys``, the default) it gets no unique-key search either: its declared keys are its
keys, so its earlier inferred unique keys are retired. (With ``fk_reference_primary_keys: false`` the search
still runs, since FKs may then reference any unique key.) For any other table, after the FK search, a null-free
unique key referenced by other tables (declared FKs, or inferred FKs of high or medium confidence) is the PK
candidate: the most referenced one (ties: fewer columns, then name order); its other unique keys are only
unique. When none is referenced, every null-free unique key stays a PK candidate.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..common.errors import InvalidArgument, NotFound
from ..common.ids import key_id_for, new_id, utcnow
from ..jobs import JobContext
from ..observability import get_metrics, span
from ..storage.db import Database, dumps, loads
from .catalog import CatalogService
from .fk_rules import FKRules, assess_foreign_key
from .keys.binder import Binder, DatasetValues, normalize_value
from .keys.ucc import discover_uccs, encode_table
from .profiling import profile_column

log = logging.getLogger(__name__)
_IND_CATEGORIES = {"integer", "decimal", "string", "uuid", "date", "datetime", "time"}
REFERENCE_LEVELS = {"high", "medium"}  # inferred FKs that count as references when choosing the PK candidate


def choose_pk_candidates(keys: dict[tuple[str, ...], bool], references: dict[frozenset, int]) -> set[tuple[str, ...]]:
    """PK candidates among a table's unique keys (``columns -> null-free``), given how many FKs of other tables
    reference each column set: the most referenced null-free key (ties: fewer columns, then name order), or,
    when none is referenced, every null-free key."""
    eligible = [k for k, null_free in keys.items() if null_free]
    referenced = [k for k in eligible if references.get(frozenset(k), 0) > 0]
    if not referenced:
        return set(eligible)
    return {min(referenced, key=lambda k: (-references[frozenset(k)], len(k), k))}


class KeyDiscoveryService:
    def __init__(self, settings, db: Database, catalog: CatalogService):
        self.cfg = settings
        self.db = db
        self.catalog = catalog
        self.rules = FKRules.from_settings(settings)

    def discover(self, dataset_ids: list[str], max_rows: int | None = None,
                 job: JobContext | None = None) -> dict[str, Any]:
        if not dataset_ids:
            raise InvalidArgument("at least one dataset_id is required")
        rows = []
        for ds_id in dict.fromkeys(dataset_ids):
            row = self.catalog.dataset_row(ds_id)
            if not row["active"]:
                raise NotFound(f"dataset {ds_id} is retired")
            rows.append(row)
        limit = self.cfg.max_rows if max_rows is None else max_rows
        if limit < 0:
            raise InvalidArgument("max_rows must be >= 0")
        run_id = new_id("prof")
        algorithm = {"ucc": self.cfg.ucc_algorithm, "ind": "binder", "max_key_size": self.cfg.max_key_size,
                     "max_fk_arity": self.cfg.max_fk_arity, "null_equals_null": self.cfg.null_equals_null,
                     "fk_identifiers_only": self.rules.identifiers_only,
                     "fk_reference_primary_keys": self.rules.reference_primary_keys,
                     "fk_identifier_class_words": sorted(self.rules.class_words)}
        base_scope = {"max_rows": limit or None, "method": "head" if limit else "full"}
        self.db.execute(
            "INSERT INTO profiling_runs(run_id, job_id, dataset_ids_json, scope_json, algorithm_json, status, started_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (run_id, job.job_id if job else None, dumps([r["dataset_id"] for r in rows]), dumps(base_scope),
             dumps(algorithm), "running", utcnow()))
        metrics = get_metrics()
        values: dict[str, DatasetValues] = {}
        scopes: dict[str, dict[str, Any]] = {}
        unique_keys: dict[str, list[tuple[str, ...]]] = {}
        identifiers: dict[str, dict[str, str]] = {}  # dataset -> column -> why it is an identifier
        inferred_ids: dict[str, set[str]] = {}
        counts = {"datasets_profiled": 0, "inferred_unique_keys": 0, "primary_key_candidates": 0,
                  "inferred_foreign_keys": 0, "declared_keys": 0, "failed": 0}
        with span("keys.discover", "TOOL", {"profiling.run_id": run_id, "profiling.datasets": len(rows)}) as sp:
            connectors: dict[str, Any] = {}
            try:
                for row in rows:
                    t0 = time.perf_counter()
                    try:
                        conn = connectors.get(row["source_id"]) or connectors.setdefault(
                            row["source_id"], self.catalog.connector(row["source_id"]))
                        result = self._profile_dataset(run_id, row, conn, limit, base_scope)
                    except Exception as exc:
                        counts["failed"] += 1
                        log.warning("profiling failed", extra={"dataset_id": row["dataset_id"],
                                                               "error": type(exc).__name__})
                        if job:
                            job.add_item(row["dataset_id"], "failed", f"{type(exc).__name__}: {str(exc)[:200]}")
                        continue
                    values[row["dataset_id"]] = result["values"]
                    scopes[row["dataset_id"]] = result["scope"]
                    unique_keys[row["dataset_id"]] = result["unique_keys"]
                    identifiers[row["dataset_id"]] = result["identifiers"]
                    inferred_ids[row["dataset_id"]] = set(result["key_ids"])
                    counts["datasets_profiled"] += 1
                    counts["inferred_unique_keys"] += result["ucc_count"]
                    counts["primary_key_candidates"] += result["pk_candidates"]
                    counts["declared_keys"] += result["declared"]
                    metrics.stage_duration.record(time.perf_counter() - t0, {"stage": "profiling_ucc"})
                    if job:
                        job.add_item(row["dataset_id"], "succeeded", None, rows_evaluated=result["scope"]["rows_evaluated"],
                                     sampled=result["scope"]["sampled"], inferred_unique_keys=result["ucc_count"],
                                     complete=result["complete"])
            finally:
                for c in connectors.values():
                    c.close()

            t1 = time.perf_counter()
            binder = Binder(self.cfg.binder_buckets, self.cfg.binder_allow_cross_type,
                            self.cfg.binder_min_distinct_dependent, self.cfg.max_nary_candidates)
            eligible = (lambda attr: attr[1] in identifiers.get(attr[0], {})) if self.rules.identifiers_only else None
            inds = binder.discover(values, unique_keys, self.cfg.max_fk_arity, eligible) if values else []
            metrics.stage_duration.record(time.perf_counter() - t1, {"stage": "binder"})
            now = utcnow()
            for ind in inds:
                kid = key_id_for(ind.dep_dataset, "foreign", "inferred", list(ind.dep_columns),
                                 ind.ref_dataset, list(ind.ref_columns))
                dep_vals, ref_vals = values[ind.dep_dataset], values[ind.ref_dataset]
                dep_distinct = len(dep_vals.tuples(ind.dep_columns)) if len(ind.dep_columns) > 1 else \
                    len(dep_vals.distinct[ind.dep_columns[0]])
                ref_distinct = len(ref_vals.tuples(ind.ref_columns)) if len(ind.ref_columns) > 1 else \
                    len(ref_vals.distinct[ind.ref_columns[0]])
                scope = {"dependent": scopes[ind.dep_dataset], "referenced": scopes[ind.ref_dataset],
                         "sampled": scopes[ind.dep_dataset]["sampled"] or scopes[ind.ref_dataset]["sampled"],
                         "profiling_run_id": run_id}
                metrics_json = {"dependent_distinct": dep_distinct, "referenced_distinct": ref_distinct,
                                "coverage": round(dep_distinct / ref_distinct, 4) if ref_distinct else None,
                                "name_similarity": _name_similarity(ind.dep_columns, ind.ref_columns),
                                "identifier_basis": {
                                    "dependent": [identifiers.get(ind.dep_dataset, {}).get(c) for c in ind.dep_columns],
                                    "referenced": [identifiers.get(ind.ref_dataset, {}).get(c) for c in ind.ref_columns]},
                                "matches_declared": self._matches_declared(ind)}
                self._upsert_inferred(kid, ind.dep_dataset, "foreign", list(ind.dep_columns), run_id, scope,
                                      metrics_json, False, ind.ref_dataset, list(ind.ref_columns), now)
                inferred_ids.setdefault(ind.dep_dataset, set()).add(kid)
                counts["inferred_foreign_keys"] += 1
            self._retire_stale(inferred_ids, set(values), now)
            # PK candidates of the profiled tables and of the tables the FKs found here reference
            targets = list(values) + sorted({ind.ref_dataset for ind in inds} - set(values))
            candidates = self._assign_pk_candidates(targets, now)
            counts["primary_key_candidates"] = sum(candidates.get(ds, 0) for ds in values)
            for key_type, n in (("unique", counts["inferred_unique_keys"]), ("foreign", counts["inferred_foreign_keys"])):
                metrics.keys.add(n, {"key_type": key_type, "origin": "inferred"})
            metrics.keys.add(counts["declared_keys"], {"key_type": "any", "origin": "declared"})
            status = "failed" if counts["failed"] and not counts["datasets_profiled"] else "succeeded"
            self.db.execute("UPDATE profiling_runs SET status = ?, finished_at = ?, scope_json = ? WHERE run_id = ?",
                            (status, utcnow(), dumps({**base_scope, "datasets": scopes}), run_id))
            sp.set_attribute("profiling.inferred_unique_keys", counts["inferred_unique_keys"])
            sp.set_attribute("profiling.inferred_foreign_keys", counts["inferred_foreign_keys"])
            sp.set_attribute("binder.stats", dumps(binder.stats))
        return {"profiling_run_id": run_id, "profiling_ref": f"/api/profiling-runs/{run_id}", **counts,
                "binder": binder.stats,
                "catalog_refs": [f"/api/datasets/{d}" for d in list(values)[:50]]}

    def _profile_dataset(self, run_id: str, row: dict[str, Any], connector, limit: int,
                         base_scope: dict[str, Any]) -> dict[str, Any]:
        dataset_id = row["dataset_id"]
        columns = self.catalog.active_columns(dataset_id)
        names = [c["name"] for c in columns]
        table = connector.read_sample(self.catalog.dataset_ref(row), names, limit or None)
        n = table.num_rows
        total = row["row_count"]
        sampled = bool(limit) and (total is None or n < total)
        scope = {**base_scope, "rows_evaluated": n, "total_rows": total, "sampled": sampled,
                 "profiling_run_id": run_id}
        now = utcnow()
        profile_rows = []
        categories: dict[str, str] = {}
        content_types: dict[str, str] = {}
        null_free: dict[str, bool] = {}
        for col in columns:
            arr = table.column(col["name"])
            prof = profile_column(arr, self.cfg)
            categories[col["name"]] = prof["category"]
            content_types[col["name"]] = prof["content_type"]
            null_free[col["name"]] = prof["null_count"] == 0
            profile_rows.append((run_id, col["column_id"], dataset_id, prof["row_count"], prof["distinct_count"],
                                 prof["empty_count"], prof["null_count"], dumps(prof["format_patterns"]),
                                 prof["content_type"], dumps(prof["most_common_values"]), now, prof["category"]))
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM column_profiles WHERE dataset_id = ?", (dataset_id,))
            conn.executemany(
                "INSERT INTO column_profiles(run_id, column_id, dataset_id, row_count, distinct_count, empty_count,"
                " null_count, format_patterns_json, content_type, most_common_values_json, created_at, type_category)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", profile_rows)

        declared = self.db.query("SELECT key_type, columns_json FROM keys WHERE dataset_id = ? AND origin = 'declared' "
                                 "AND active = 1", (dataset_id,))
        has_declared_pk = any(k["key_type"] == "primary" for k in declared)
        # unique column combinations: not searched when the table declares its primary key and FKs may only
        # reference primary keys (they are then useless as PK candidates and as FK targets)
        excluded = [k.lower() for k in self.cfg.key_excluded_type_keywords]
        eligible = [i for i, c in enumerate(columns)
                    if not any(x in (c["data_type"] or "").lower() for x in excluded)
                    and categories[c["name"]] not in {"float", "binary", "other"}][: self.cfg.max_columns]
        uccs: list[tuple[str, ...]] = []
        complete = True
        if has_declared_pk and self.rules.reference_primary_keys:
            scope["ucc_search_skipped"] = "declared primary key"
        elif n >= self.cfg.min_rows_for_inference and eligible:
            matrix = encode_table(table.select(names), self.cfg.null_equals_null)
            result = discover_uccs(matrix, eligible, self.cfg.max_key_size, self.cfg.ucc_algorithm,
                                   self.cfg.max_ucc_candidates)
            complete = result.complete
            uccs = [tuple(names[i] for i in combo) for combo in result.uccs]
        scope["ucc_search_complete"] = complete
        key_ids = []
        pk_candidates = 0
        for combo in uccs:
            combo_null_free = all(null_free[c] for c in combo)
            pk = combo_null_free and not has_declared_pk  # refined after the FK search (_assign_pk_candidates)
            pk_candidates += int(pk)
            kid = key_id_for(dataset_id, "unique", "inferred", list(combo))
            self._upsert_inferred(kid, dataset_id, "unique", list(combo), run_id, scope,
                                  {"matches_declared": self._declared_unique(dataset_id, combo),
                                   "null_free": combo_null_free}, pk, None, None, now)
            key_ids.append(kid)

        unique_keys = self.rules.referenceable_keys(
            [(k["key_type"], tuple(loads(k["columns_json"], []))) for k in declared], uccs, null_free)
        ind_columns = {c["name"]: [normalize_value(v) for v in table.column(c["name"]).to_pylist()]
                       for c in columns if categories[c["name"]] in _IND_CATEGORIES}
        values = DatasetValues(dataset_id, ind_columns, {k: categories[k] for k in ind_columns})
        unique_keys = {k for k in unique_keys if k and all(c in ind_columns for c in k)}
        declared_cols = {c for k in declared for c in loads(k["columns_json"], [])}
        identifiers = {}
        for col in columns:
            if col["name"] in ind_columns:
                basis = self.rules.identifier_basis(col, col["name"] in declared_cols, categories[col["name"]],
                                                    content_types[col["name"]])
                if basis:
                    identifiers[col["name"]] = basis
        return {"values": values, "scope": scope, "unique_keys": sorted(unique_keys, key=lambda k: (len(k), k)),
                "identifiers": identifiers,
                "key_ids": key_ids, "ucc_count": len(uccs), "pk_candidates": pk_candidates,
                "declared": len(declared), "complete": complete}

    def _upsert_inferred(self, kid, dataset_id, key_type, columns, run_id, scope, metrics_json, pk,
                         ref_dataset, ref_columns, now) -> None:
        self.db.execute(
            """INSERT INTO keys(key_id, dataset_id, key_type, origin, columns_json, referenced_dataset_id,
                   referenced_columns_json, is_pk_candidate, profiling_run_id, scope_json, metrics_json, active,
                   created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?,?)
               ON CONFLICT(key_id) DO UPDATE SET profiling_run_id=excluded.profiling_run_id,
                   scope_json=excluded.scope_json, metrics_json=excluded.metrics_json,
                   is_pk_candidate=excluded.is_pk_candidate, active=1, updated_at=excluded.updated_at""",
            (kid, dataset_id, key_type, "inferred", dumps(columns), ref_dataset,
             dumps(ref_columns) if ref_columns else None, int(pk), run_id, dumps(scope), dumps(metrics_json), now, now))

    def _assign_pk_candidates(self, dataset_ids: list[str], now: str) -> dict[str, int]:
        """Set ``is_pk_candidate`` on the active inferred unique keys of each table (see the module docstring);
        returns the number of PK candidates per table."""
        out: dict[str, int] = {}
        for ds in dict.fromkeys(dataset_ids):
            uniques = self.db.query("SELECT key_id, columns_json, metrics_json, is_pk_candidate FROM keys "
                                    "WHERE dataset_id = ? AND origin = 'inferred' AND key_type = 'unique' "
                                    "AND active = 1", (ds,))
            declared_pk = self.db.query_one("SELECT 1 FROM keys WHERE dataset_id = ? AND origin = 'declared' "
                                            "AND key_type = 'primary' AND active = 1", (ds,)) is not None
            chosen: set[tuple[str, ...]] = set()
            if not declared_pk:  # a declared primary key is the PK: no candidate is computed
                keys = {tuple(loads(k["columns_json"], [])): bool(loads(k["metrics_json"], {}).get(
                    "null_free", k["is_pk_candidate"])) for k in uniques}
                references: dict[frozenset, int] = {}
                for fk in self.db.query("SELECT * FROM keys WHERE referenced_dataset_id = ? AND key_type = 'foreign' "
                                        "AND active = 1 AND dataset_id <> ?", (ds, ds)):  # other tables only
                    if fk["origin"] == "inferred":
                        confidence = assess_foreign_key(self.db, fk, self.rules)
                        if not confidence or confidence.get("level") not in REFERENCE_LEVELS:
                            continue
                    cols = frozenset(loads(fk["referenced_columns_json"], []))
                    references[cols] = references.get(cols, 0) + 1
                chosen = choose_pk_candidates(keys, references)
            for k in uniques:
                pk = int(tuple(loads(k["columns_json"], [])) in chosen)
                if pk != k["is_pk_candidate"]:
                    self.db.execute("UPDATE keys SET is_pk_candidate = ?, updated_at = ? WHERE key_id = ?",
                                    (pk, now, k["key_id"]))
            out[ds] = len(chosen)
        return out

    def _retire_stale(self, found: dict[str, set[str]], profiled: set[str], now: str) -> None:
        for ds in profiled:
            for k in self.db.query("SELECT key_id, key_type, referenced_dataset_id FROM keys WHERE dataset_id = ? "
                                   "AND origin = 'inferred' AND active = 1", (ds,)):
                if k["key_id"] in found.get(ds, set()):
                    continue
                if k["key_type"] == "foreign" and k["referenced_dataset_id"] not in profiled:
                    continue  # referenced side not re-evaluated in this run; keep the earlier finding
                self.db.execute("UPDATE keys SET active = 0, updated_at = ? WHERE key_id = ?", (now, k["key_id"]))

    def _declared_unique(self, dataset_id: str, combo: tuple[str, ...]) -> bool:
        for k in self.db.query("SELECT columns_json FROM keys WHERE dataset_id = ? AND origin = 'declared' "
                               "AND key_type IN ('primary','unique') AND active = 1", (dataset_id,)):
            if tuple(loads(k["columns_json"], [])) == combo:
                return True
        return False

    def _matches_declared(self, ind) -> bool:
        return self.db.query_one(
            "SELECT 1 FROM keys WHERE dataset_id = ? AND origin = 'declared' AND key_type = 'foreign' AND active = 1 "
            "AND columns_json = ? AND referenced_dataset_id = ? AND referenced_columns_json = ?",
            (ind.dep_dataset, dumps(list(ind.dep_columns)), ind.ref_dataset, dumps(list(ind.ref_columns)))) is not None

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM profiling_runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise NotFound(f"profiling run not found: {run_id}")
        keys = self.db.query("SELECT key_id, dataset_id, key_type, columns_json, referenced_dataset_id, "
                             "referenced_columns_json, is_pk_candidate, active FROM keys WHERE profiling_run_id = ?",
                             (run_id,))
        return {"run_id": run_id, "job_id": row["job_id"], "status": row["status"],
                "dataset_ids": loads(row["dataset_ids_json"], []), "scope": loads(row["scope_json"], {}),
                "algorithm": loads(row["algorithm_json"], {}), "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "keys": [{"key_id": k["key_id"], "dataset_id": k["dataset_id"], "key_type": k["key_type"],
                          "columns": loads(k["columns_json"], []),
                          "referenced_dataset_id": k["referenced_dataset_id"],
                          "referenced_columns": loads(k["referenced_columns_json"], []),
                          "primary_key_candidate": bool(k["is_pk_candidate"]), "active": bool(k["active"])}
                         for k in keys[:200]]}


def _name_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    import re

    ta = {t for c in a for t in re.split(r"[_\W]+", c.lower()) if t}
    tb = {t for c in b for t in re.split(r"[_\W]+", c.lower()) if t}
    return round(len(ta & tb) / len(ta | tb), 3) if ta | tb else 0.0

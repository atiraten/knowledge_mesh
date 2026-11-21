"""``execute_sql``: one read-only query on a cataloged database or Databricks source (spec 3.17).

Checks before anything reaches the database (sqlglot, dialect of the source):

* exactly one statement, a SELECT (optionally WITH ... / UNION ...), with no SELECT INTO and no
  locking clause (FOR UPDATE) anywhere;
* every table it reads (CTE names excluded) is an active cataloged dataset of that source, and the
  source and datasets are inside the caller's scope (the search request's filters);
* the SQL is at most ``sql_execution.max_sql_chars`` long.

The query then runs through the source's connector on its read-only session (SQLite ``mode=ro`` +
``query_only``, PostgreSQL/MySQL read-only transactions; Databricks relies on a SELECT-only token)
with a statement timeout where the dialect supports one. At most ``max_rows`` rows are returned
(``truncated`` says when more exist). Results are structured: ``status`` ``ok`` (an empty ``rows``
list when nothing matches) or ``error`` with a safe message. Row values are returned to the caller
only; logs and metrics never contain them.

Each result column carries ``sensitive``: true when its value is computed from a cataloged column whose
business term can identify a person (``columns.term_sensitive``, set by the term model), with the
terms in ``sensitive_terms``. Lineage is read with sqlglot (aliases, expressions, CTEs, derived tables,
scalar subqueries and UNION branches); a column used only inside ``COUNT(...)`` does not count, since a
count reveals no value. ``false`` when every contributing column was assessed as not sensitive (or
nothing contributes, e.g. ``COUNT(*)``); ``null`` when a contributing term was never assessed or the
lineage could not be read (then matched by output column name).
"""

from __future__ import annotations

import datetime as dt
import decimal
import logging
import time
import uuid
from typing import Any

from ..common.errors import MeshError, NotFound
from ..common.secrets import mask_secrets
from ..observability import capture_sensitive, get_metrics, span
from ..storage.db import Database

log = logging.getLogger(__name__)
DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql", "mariadb": "mysql", "mssql": "tsql",
            "oracle": "oracle", "databricks": "databricks"}


class SqlRejected(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class SqlExecutor:
    def __init__(self, settings, db: Database, catalog):
        self.cfg = settings
        self.db = db
        self.catalog = catalog

    # -- validation ------------------------------------------------------------------------------
    def dialect(self, source_id: str) -> str:
        src = self.catalog.source_settings(source_id)
        if src.kind not in ("database", "databricks"):
            raise SqlRejected("unsupported_source", f"execute_sql supports database and Databricks sources; "
                                                    f"{source_id} is a {src.kind} source")
        if src.kind == "databricks":
            return "databricks"
        row = self.db.query_one("SELECT physical_ref FROM datasets WHERE source_id = ? AND active = 1 "
                                "AND physical_ref IS NOT NULL LIMIT 1", (source_id,))
        backend = (row["physical_ref"].split(":", 1)[0] if row else None) or src.db_type
        if backend not in DIALECTS:
            raise SqlRejected("not_cataloged", f"source {source_id} has no cataloged tables; catalog it first")
        return DIALECTS[backend]

    def validate(self, source_id: str, sql: str, sources: list[str] | None = None,
                 dataset_ids: list[str] | None = None) -> dict[str, Any]:
        """Parse and check the query; returns ``{dialect, tables}`` or raises ``SqlRejected``."""
        import sqlglot
        from sqlglot import exp

        if sources is not None and source_id not in sources:
            raise SqlRejected("out_of_scope", f"source {source_id} is outside this search's scope")
        try:
            self.catalog.source_settings(source_id)
        except NotFound:
            raise SqlRejected("unknown_source", f"structured source not configured: {source_id}")
        sql = (sql or "").strip()
        if not sql:
            raise SqlRejected("invalid_sql", "sql is required")
        if len(sql) > self.cfg.max_sql_chars:
            raise SqlRejected("invalid_sql", f"sql exceeds {self.cfg.max_sql_chars} characters")
        dialect = self.dialect(source_id)
        try:
            statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
        except Exception as exc:
            raise SqlRejected("invalid_sql", f"the SQL does not parse as {dialect}: {str(exc).splitlines()[0][:200]}")
        if len(statements) != 1:
            raise SqlRejected("not_read_only", "exactly one SQL statement is allowed")
        stmt = statements[0]
        root = stmt.this if isinstance(stmt, exp.Subquery) else stmt
        if not isinstance(root, (exp.Select, exp.SetOperation)):
            raise SqlRejected("not_read_only", f"only SELECT queries are allowed (got {type(root).__name__.upper()})")
        forbidden = (exp.Into, exp.Lock, exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop,
                     exp.Command, exp.Alter if hasattr(exp, "Alter") else exp.Command)
        for node in stmt.walk():
            if isinstance(node, forbidden):
                raise SqlRejected("not_read_only", f"{type(node).__name__.upper()} is not allowed in execute_sql")
        ctes = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
        datasets = self.db.query("SELECT dataset_id, schema_name, name, catalog_ref FROM datasets "
                                 "WHERE source_id = ? AND active = 1", (source_id,))
        used: dict[str, dict[str, Any]] = {}
        for t in stmt.find_all(exp.Table):
            name = (t.name or "").lower()
            if not name:
                raise SqlRejected("uncataloged_table", "table functions are not allowed; query cataloged tables")
            schema = (t.db or "").lower()
            if not schema and name in ctes:
                continue
            matches = [d for d in datasets if d["name"].lower() == name
                       and (not schema or (d["schema_name"] or "").lower() == schema)]
            if not matches:
                where = f"{schema}.{name}" if schema else name
                raise SqlRejected("uncataloged_table", f"table {where} is not a cataloged dataset of source {source_id}")
            if len(matches) > 1:
                raise SqlRejected("ambiguous_table", f"table {name} exists in several schemas of {source_id}: "
                                  + ", ".join(sorted(d["schema_name"] or "" for d in matches)) + "; qualify it")
            d = matches[0]
            if dataset_ids is not None and d["dataset_id"] not in dataset_ids:
                raise SqlRejected("out_of_scope", f"table {d['name']} is outside this search's scope")
            used[d["dataset_id"]] = d
        if not used:
            raise SqlRejected("uncataloged_table", "the query reads no cataloged table")
        return {"dialect": dialect, "tables": [{"dataset_id": d["dataset_id"], "schema": d["schema_name"],
                                                "name": d["name"], "catalog_ref": d["catalog_ref"]}
                                               for d in used.values()]}

    # -- execution -------------------------------------------------------------------------------
    def execute(self, source_id: str, sql: str, max_rows: int | None = None, sources: list[str] | None = None,
                dataset_ids: list[str] | None = None) -> dict[str, Any]:
        metrics = get_metrics()
        if not self.cfg.enabled:
            return _error(source_id, "disabled", "execute_sql is disabled (sql_execution.enabled is false)")
        limit = min(max(1, int(max_rows or self.cfg.default_max_rows)), self.cfg.max_rows_limit)
        start = time.perf_counter()
        attrs = {"sql.source_id": source_id, "sql.max_rows": limit}
        if capture_sensitive():
            attrs["input.value"] = (sql or "")[:4000]
        with span("sql.execute", "TOOL", attrs) as sp:
            try:
                checked = self.validate(source_id, sql, sources, dataset_ids)
            except SqlRejected as exc:
                sp.set_attribute("sql.status", "rejected")
                sp.set_attribute("sql.error_code", exc.code)
                metrics.sql_queries.add(1, {"status": "rejected"})
                return _error(source_id, exc.code, exc.message)
            connector = None
            try:
                connector = self.catalog.connector(source_id)
                result = connector.execute_read_only(sql.strip(), limit, self.cfg.timeout_seconds)
            except MeshError as exc:
                sp.set_attribute("sql.status", "error")
                metrics.sql_queries.add(1, {"status": "error"})
                return _error(source_id, exc.code, exc.message, checked)
            except Exception as exc:
                detail = str(getattr(exc, "orig", None) or exc).splitlines()[0] if str(exc) else ""
                timed_out = "interrupt" in detail.lower() or "timeout" in detail.lower() or "cancel" in detail.lower()
                code = "timeout" if timed_out else "execution_failed"
                message = mask_secrets(f"query failed ({type(getattr(exc, 'orig', None) or exc).__name__}: "
                                       f"{detail})"[:400], [])
                if timed_out:
                    message = f"query exceeded the {self.cfg.timeout_seconds}s timeout"
                sp.set_attribute("sql.status", code)
                metrics.sql_queries.add(1, {"status": code})
                log.warning("execute_sql failed", extra={"source_id": source_id, "code": code})
                return _error(source_id, code, message, checked)
            finally:
                if connector is not None:
                    try:
                        connector.close()
                    except Exception:
                        pass
            elapsed = int((time.perf_counter() - start) * 1000)
            rows = [[_json_value(v) for v in r] for r in result.rows]
            marks = self.column_sensitivity(sql.strip(), checked["dialect"], checked["tables"],
                                            [name for name, _ in result.columns])
            columns = [{"name": name, "type": _column_type(dtype, [r[i] for r in result.rows]), **marks[i]}
                       for i, (name, dtype) in enumerate(result.columns)]
            sp.set_attribute("sql.sensitive_columns", sum(1 for c in columns if c["sensitive"]))
            sp.set_attribute("sql.status", "ok")
            sp.set_attribute("sql.row_count", len(rows))
            sp.set_attribute("sql.truncated", result.truncated)
            metrics.sql_queries.add(1, {"status": "ok"})
            metrics.sql_rows.record(len(rows))
        return {"status": "ok", "source_id": source_id, "dialect": checked["dialect"], "sql": sql.strip(),
                "tables": checked["tables"], "columns": columns, "rows": rows, "row_count": len(rows),
                "truncated": result.truncated, "max_rows": limit, "elapsed_ms": elapsed}


    # -- sensitive data --------------------------------------------------------------------------
    def column_sensitivity(self, sql: str, dialect: str, tables: list[dict[str, Any]],
                           names: list[str]) -> list[dict[str, Any]]:
        """``{sensitive, sensitive_terms}`` for each result column (see the module docstring)."""
        ids = [t["dataset_id"] for t in tables]
        marks = ",".join("?" * len(ids)) or "''"
        info = {(r["dataset_id"], r["name"].lower()): r for r in self.db.query(
            f"SELECT dataset_id, name, term_label, term_sensitive FROM columns WHERE active = 1 "
            f"AND dataset_id IN ({marks}) ORDER BY dataset_id, ordinal", ids)}
        try:
            lineage = _lineage(sql, dialect, tables, info)
        except Exception:  # unreadable lineage never fails the query; fall back to the output names
            lineage = None
        if lineage is None or len(lineage) != len(names):
            lineage = [{k for k in info if k[1] == n.lower()} or None for n in names]
        out = []
        for sources in lineage:
            if sources is None:
                out.append({"sensitive": None, "sensitive_terms": []})
                continue
            rows = [info.get(k) for k in sources]
            terms = sorted({r["term_label"] or r["name"] for r in rows if r and r["term_sensitive"] == 1})
            if terms:
                out.append({"sensitive": True, "sensitive_terms": terms})
            elif any(r is None or r["term_sensitive"] is None for r in rows):
                out.append({"sensitive": None, "sensitive_terms": []})
            else:
                out.append({"sensitive": False, "sensitive_terms": []})
        return out


def _lineage(sql: str, dialect: str, tables: list[dict[str, Any]],
             info: dict[tuple[str, str], dict[str, Any]]) -> list[set[tuple[str, str]] | None] | None:
    """For each output column, the cataloged ``(dataset_id, column)`` pairs whose values flow into it
    (``None`` where that cannot be told). ``None`` when the query cannot be qualified."""
    import sqlglot
    from sqlglot import exp
    from sqlglot.optimizer.qualify import qualify
    from sqlglot.optimizer.scope import Scope, build_scope

    schema: dict[str, dict[str, dict[str, str]]] = {}
    for t in tables:
        cols = {name: "TEXT" for (ds, name) in info if ds == t["dataset_id"]}
        schema.setdefault(t["schema"] or "", {})[t["name"]] = cols
    tree = qualify(sqlglot.parse_one(sql, read=dialect), schema=schema, dialect=dialect,
                   validate_qualify_columns=False)
    root = build_scope(tree)
    if root is None:
        return None
    scope_of = {id(s.expression): s for s in root.traverse()}

    def dataset_of(table: exp.Table) -> str | None:
        name, db = (table.name or "").lower(), (table.db or "").lower()
        hits = [t for t in tables if t["name"].lower() == name and (not db or (t["schema"] or "").lower() == db)]
        return hits[0]["dataset_id"] if len(hits) == 1 else None

    def column(scope: Scope, col: exp.Column) -> set[tuple[str, str]] | None:
        s, src = scope, None
        while s is not None and src is None:  # correlated references resolve in an outer scope
            src = s.sources.get(col.table)
            s = s.parent
        if isinstance(src, exp.Table):
            ds = dataset_of(src)
            return {(ds, col.name.lower())} if ds else None
        if isinstance(src, Scope):
            return named(src, col.name)
        return None

    def expression(scope: Scope, node: exp.Expression) -> set[tuple[str, str]] | None:
        found: set[tuple[str, str]] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, exp.Count):
                continue  # a count reveals no value of its argument
            if isinstance(n, exp.Star):
                return None  # not expanded: the catalog does not list the columns
            if isinstance(n, exp.Subquery) and id(n.this) in scope_of:
                sub = scope_of[id(n.this)]
                width = len(_selects(sub))
                parts = [position(sub, i) for i in range(width)]
                if any(p is None for p in parts):
                    return None
                found.update(*parts)
                continue
            if isinstance(n, exp.Column):
                part = column(scope, n)
                if part is None:
                    return None
                found |= part
                continue
            stack.extend(n.iter_expressions())
        return found

    def _selects(scope: Scope) -> list[exp.Expression]:
        while isinstance(scope.expression, exp.SetOperation):
            scope = scope.set_operation_scopes[0]
        return list(scope.expression.selects)

    def position(scope: Scope, i: int) -> set[tuple[str, str]] | None:
        if isinstance(scope.expression, exp.SetOperation):
            parts = [position(s, i) for s in scope.set_operation_scopes]
            return None if any(p is None for p in parts) else set().union(*parts)
        selects = scope.expression.selects
        return expression(scope, selects[i]) if i < len(selects) else None

    def named(scope: Scope, name: str) -> set[tuple[str, str]] | None:
        names = [s.alias_or_name.lower() for s in _selects(scope)]
        return position(scope, names.index(name.lower())) if name.lower() in names else None

    return [position(root, i) for i in range(len(_selects(root)))]


def _error(source_id: str, code: str, message: str, checked: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {"status": "error", "source_id": source_id, "error": {"code": code, "message": message},
           "columns": [], "rows": [], "row_count": 0}
    if checked:
        out["tables"] = checked["tables"]
    return out


def _json_value(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(v))} bytes>"
    return str(v)


def _column_type(driver_type: str | None, values: list[Any]) -> str | None:
    sample = next((v for v in values if v is not None), None)
    inferred = None
    if isinstance(sample, bool):
        inferred = "boolean"
    elif isinstance(sample, int):
        inferred = "integer"
    elif isinstance(sample, (float, decimal.Decimal)):
        inferred = "number"
    elif isinstance(sample, dt.datetime):
        inferred = "datetime"
    elif isinstance(sample, dt.date):
        inferred = "date"
    elif isinstance(sample, str):
        inferred = "text"
    if driver_type and not driver_type.isdigit():
        return driver_type
    return inferred

"""Databricks SQL warehouses via ``databricks-sql-connector`` (install the ``databricks`` extra).

Metadata comes from Unity Catalog ``information_schema`` (columns, informational PRIMARY KEY /
FOREIGN KEY / UNIQUE constraints). Use a token whose principal only has SELECT/USE privileges.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import pyarrow as pa

from ...common.errors import SourceUnavailable
from ...common.secrets import mask_secrets
from .base import (ColumnInfo, Connector, DatasetRef, DatasetSchema, DeclaredKey, QueryResult, resolve_credential,
                   rows_to_table)

_IDENT = re.compile(r"^[A-Za-z0-9_]+$")


def _default_connect(**kwargs):
    try:
        from databricks import sql  # type: ignore
    except ImportError as exc:
        raise SourceUnavailable("databricks-sql-connector is not installed (pip install 'knowledge-mesh[databricks]')") from exc
    return sql.connect(**kwargs)


def _q(ident: str) -> str:
    return "`" + ident.replace("`", "``") + "`"


def _lit(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


class DatabricksConnector(Connector):
    def __init__(self, settings, secrets=None, connect: Callable[..., Any] | None = None):
        super().__init__(settings, secrets)
        for name in (settings.catalog, settings.schema_ or "default"):
            if not _IDENT.match(name):
                raise SourceUnavailable(f"source {settings.id}: invalid catalog/schema identifier {name!r}")
        self.catalog = settings.catalog
        self.schema = settings.schema_ or "default"
        kwargs = {
            "server_hostname": resolve_credential(settings, "server_hostname", "server hostname", secrets),
            "http_path": resolve_credential(settings, "http_path", "HTTP path", secrets),
            "access_token": resolve_credential(settings, "token", "access token", secrets),
        }
        try:
            self.conn = (connect or _default_connect)(**kwargs)
        except SourceUnavailable:
            raise
        except Exception as exc:
            detail = mask_secrets(f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:300],
                                  [kwargs["access_token"]])
            raise SourceUnavailable(f"source {settings.id}: connection failed ({detail})") from exc

    def _query(self, sql: str) -> list[tuple]:
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
                return [tuple(r) for r in cur.fetchall()]
        except Exception as exc:
            raise SourceUnavailable(f"source {self.source_id}: query failed ({type(exc).__name__})") from exc

    def _info(self, view: str) -> str:
        return f"{_q(self.catalog)}.information_schema.{view}"

    def list_datasets(self) -> list[DatasetRef]:
        rows = self._query(
            f"SELECT table_name, table_type FROM {self._info('tables')} WHERE table_schema = {_lit(self.schema)} "
            "ORDER BY table_name"
        )
        refs = [DatasetRef(self.schema, r[0], "view" if "VIEW" in str(r[1]).upper() else "table") for r in rows]
        return [r for r in refs if self.allowed(r)]

    def describe(self, ref: DatasetRef) -> DatasetSchema:
        where = f"table_schema = {_lit(self.schema)} AND table_name = {_lit(ref.name)}"
        cols = [
            ColumnInfo(r[0], str(r[1]), str(r[2]).upper() == "YES", int(r[3]) - 1)
            for r in self._query(
                f"SELECT column_name, data_type, is_nullable, ordinal_position FROM {self._info('columns')} "
                f"WHERE {where} ORDER BY ordinal_position"
            )
        ]
        keys: list[DeclaredKey] = []
        constraints = self._query(
            f"SELECT constraint_name, constraint_type FROM {self._info('table_constraints')} WHERE {where}"
        )
        for name, ctype in constraints:
            usage = self._query(
                f"SELECT column_name, ordinal_position FROM {self._info('key_column_usage')} "
                f"WHERE {where} AND constraint_name = {_lit(name)} ORDER BY ordinal_position"
            )
            columns = [u[0] for u in usage]
            ctype = str(ctype).upper()
            if ctype == "PRIMARY KEY":
                keys.append(DeclaredKey("primary", columns, name))
            elif ctype == "UNIQUE":
                keys.append(DeclaredKey("unique", columns, name))
            elif ctype == "FOREIGN KEY":
                ref_rows = self._query(
                    f"SELECT unique_constraint_schema, unique_constraint_name FROM {self._info('referential_constraints')} "
                    f"WHERE constraint_schema = {_lit(self.schema)} AND constraint_name = {_lit(name)}"
                )
                if not ref_rows:
                    continue
                ref_schema, ref_constraint = ref_rows[0]
                ref_usage = self._query(
                    f"SELECT table_name, column_name FROM {self._info('key_column_usage')} "
                    f"WHERE constraint_schema = {_lit(ref_schema)} AND constraint_name = {_lit(ref_constraint)} "
                    "ORDER BY ordinal_position"
                )
                if ref_usage:
                    keys.append(DeclaredKey("foreign", columns, name, ref_schema, ref_usage[0][0],
                                            [u[1] for u in ref_usage]))
        count = self._query(f"SELECT COUNT(*) FROM {self._fq(ref)}")
        return DatasetSchema(ref, f"databricks:{self.catalog}.{self.schema}.{ref.name}", cols, keys,
                             int(count[0][0]) if count else None)

    def _fq(self, ref: DatasetRef) -> str:
        return f"{_q(self.catalog)}.{_q(self.schema)}.{_q(ref.name)}"

    def read_sample(self, ref: DatasetRef, columns: list[str], limit: int | None) -> pa.Table:
        sql = f"SELECT {', '.join(_q(c) for c in columns)} FROM {self._fq(ref)}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
                if hasattr(cur, "fetchall_arrow"):
                    return cur.fetchall_arrow()
                return rows_to_table(columns, [tuple(r) for r in cur.fetchall()])
        except Exception as exc:
            raise SourceUnavailable(f"source {self.source_id}: read failed ({type(exc).__name__})") from exc

    def execute_read_only(self, sql: str, max_rows: int, timeout_seconds: float) -> QueryResult:
        """One query on the warehouse; use a token whose principal only has SELECT/USE privileges."""
        with self.conn.cursor() as cur:
            cur.execute(sql)
            description = cur.description or []
            rows = [tuple(r) for r in cur.fetchmany(max_rows + 1)]
        columns = [(d[0], str(d[1]) if len(d) > 1 and d[1] is not None else None) for d in description]
        return QueryResult(columns, rows[:max_rows], len(rows) > max_rows)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

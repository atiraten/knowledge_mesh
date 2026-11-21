"""Relational databases through SQLAlchemy.

Two ways to describe the connection:

* **Fields** (Settings page default): database type, hostname, port, database name, optional
  schema, username, and a password from the OS keychain or an environment variable. The URL
  is assembled with ``URL.create`` so special characters in passwords need no escaping.
* **Connection URL** (``connection_url_env`` or a keychain-stored URL): any SQLAlchemy URL,
  e.g. SQLite files or dialects not listed in ``DB_TYPES``.

Sessions are forced read-only where the dialect supports it (SQLite ``mode=ro`` +
``query_only``, PostgreSQL/MySQL read-only transactions); use a read-only account as well.
"""

from __future__ import annotations

import os
import time

import pyarrow as pa
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import NoSuchModuleError, SQLAlchemyError
from sqlalchemy.sql import column as sql_column
from sqlalchemy.sql import table as sql_table

from ...common.errors import SourceUnavailable
from ...common.secrets import mask_secrets
from .base import (ColumnInfo, Connector, DatasetRef, DatasetSchema, DeclaredKey, QueryResult, resolve_credential,
                   rows_to_table)

CONNECT_TIMEOUT_SECONDS = 10
_SYSTEM_SCHEMAS = {
    "information_schema", "pg_catalog", "sys", "guest", "mysql", "performance_schema", "sqlite_temp",
    # Oracle-maintained accounts
    "system", "sysaux", "outln", "xdb", "ctxsys", "mdsys", "ordsys", "ordplugins", "orddata", "olapsys",
    "dbsnmp", "wmsys", "lbacsys", "dvsys", "dvf", "audsys", "ojvmsys", "gsmadmin_internal", "gsmcatuser",
    "gsmuser", "appqossys", "dbsfwuser", "remote_scheduler_agent", "sysbackup", "sysdg", "syskm", "sysrac",
    "si_informtn_schema", "anonymous", "xs$null", "dip", "ggsys", "mddata", "spatial_csw_admin_usr",
}
_SYSTEM_PREFIXES = ("pg_temp", "pg_toast", "db_", "apex_", "flows_")


def _is_system_schema(name: str) -> bool:
    lowered = name.lower()
    return lowered in _SYSTEM_SCHEMAS or lowered.startswith(_SYSTEM_PREFIXES)

# db_type -> SQLAlchemy driver, default port, pip package, connect-timeout argument of the driver
DB_TYPES: dict[str, dict] = {
    "postgresql": {"label": "PostgreSQL", "driver": "postgresql+psycopg", "port": 5432,
                   "package": "psycopg[binary]", "timeout_arg": "connect_timeout"},
    "mysql": {"label": "MySQL / MariaDB", "driver": "mysql+pymysql", "port": 3306,
              "package": "pymysql", "timeout_arg": "connect_timeout"},
    "mssql": {"label": "SQL Server", "driver": "mssql+pymssql", "port": 1433,
              "package": "pymssql", "timeout_arg": "login_timeout"},
    "oracle": {"label": "Oracle", "driver": "oracle+oracledb", "port": 1521,
               "package": "oracledb", "timeout_arg": "tcp_connect_timeout"},
}


def build_url(db_type: str, host: str, port: int | None, database: str, username: str | None,
              password: str | None) -> URL:
    """SQLAlchemy URL from connection fields. For Oracle, ``database`` is the service name."""
    spec = DB_TYPES[db_type]
    if db_type == "oracle":
        return URL.create(spec["driver"], username=username, password=password, host=host,
                          port=port or spec["port"], query={"service_name": database})
    return URL.create(spec["driver"], username=username, password=password, host=host,
                      port=port or spec["port"], database=database)


def _why(exc: Exception, secret: str | None) -> str:
    detail = str(getattr(exc, "orig", None) or exc).splitlines()[0] if str(exc) else ""
    return mask_secrets(f"{type(exc).__name__}: {detail}"[:300], [secret] if secret else [])


def _password(settings, secrets) -> str | None:
    """Password for field-based connections: keychain, environment variable, or none."""
    if settings.credential_store == "secret_store":
        value = secrets.get(settings.id, "password") if secrets is not None else None
        if not value:
            raise SourceUnavailable(f"source {settings.id}: the password is not in the secret store; "
                                    "re-enter it in Settings")
        return value
    if settings.password_env:
        value = os.environ.get(settings.password_env)
        if not value:
            raise SourceUnavailable(f"source {settings.id}: environment variable {settings.password_env} "
                                    "(password) is not set", {"env": settings.password_env})
        return value
    return None  # e.g. trust/peer authentication


class DatabaseConnector(Connector):
    def __init__(self, settings, secrets=None):
        super().__init__(settings, secrets)
        self._all_schemas: list[str | None] | None = None
        connect_args: dict = {}
        if settings.host:
            spec = DB_TYPES[settings.db_type]
            secret = _password(settings, secrets)
            url = build_url(settings.db_type, settings.host, settings.port, settings.database, settings.username,
                            secret)
            connect_args[spec["timeout_arg"]] = CONNECT_TIMEOUT_SECONDS
        else:
            secret = resolve_credential(settings, "connection_url", "connection URL", secrets)
            try:
                url = make_url(secret)
            except Exception as exc:
                raise SourceUnavailable(f"source {settings.id}: invalid connection URL") from exc
        if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
            url = url.set(database=f"file:{url.database}", query={**url.query, "mode": "ro", "uri": "true"})
        try:
            self.engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        except (ModuleNotFoundError, NoSuchModuleError) as exc:
            spec = DB_TYPES.get(settings.db_type or "")
            hint = f" (pip install '{spec['package']}')" if spec else ""
            raise SourceUnavailable(f"source {settings.id}: the database driver is not installed{hint}: "
                                    f"{_why(exc, secret)}") from exc
        except Exception as exc:
            raise SourceUnavailable(f"source {settings.id}: cannot create engine ({_why(exc, secret)})") from exc
        raw_url = secret
        backend = url.get_backend_name()

        @event.listens_for(self.engine, "connect")
        def _read_only(dbapi_conn, _record):  # pragma: no cover - dialect specific
            cur = dbapi_conn.cursor()
            try:
                if backend == "sqlite":
                    cur.execute("PRAGMA query_only = ON")
                elif backend == "postgresql":
                    cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
                elif backend in {"mysql", "mariadb"}:
                    cur.execute("SET SESSION TRANSACTION READ ONLY")
            finally:
                cur.close()

        try:
            self._inspector = inspect(self.engine)
        except SQLAlchemyError as exc:
            raise SourceUnavailable(f"source {settings.id}: connection failed ({_why(exc, raw_url)})") from exc

    def _schemas(self) -> list[str | None]:
        """The configured schemas; when none are configured, every non-system schema.

        In MySQL/MariaDB a schema *is* a database, so "all schemas" means the connected one.
        """
        if self.settings.schemas:
            return list(self.settings.schemas)
        if self._all_schemas is None:
            default = self._inspector.default_schema_name
            if self.engine.url.get_backend_name() in {"mysql", "mariadb"}:
                self._all_schemas = [default]
            else:
                try:
                    names = [s for s in self._inspector.get_schema_names() if not _is_system_schema(s)]
                except SQLAlchemyError:
                    names = []
                self._all_schemas = names or [default]
        return self._all_schemas

    def scanned_schemas(self) -> list[str] | None:
        return [str(s) for s in self._schemas() if s]

    def schema_overview(self) -> list[tuple[str, int]] | None:
        out: list[tuple[str, int]] = []
        for schema in self._inspector.get_schema_names()[:200]:
            if _is_system_schema(schema):
                continue
            try:
                n = len(self._inspector.get_table_names(schema=schema)) + \
                    len(self._inspector.get_view_names(schema=schema))
            except SQLAlchemyError:
                continue  # no privilege on that schema
            if n:
                out.append((schema, n))
        return out

    def list_datasets(self) -> list[DatasetRef]:
        refs: list[DatasetRef] = []
        failures: list[str] = []
        schemas = self._schemas()
        for schema in schemas:
            try:
                names = [(n, "table") for n in self._inspector.get_table_names(schema=schema)]
                names += [(n, "view") for n in self._inspector.get_view_names(schema=schema)]
            except SQLAlchemyError as exc:
                failures.append(f"{schema}: {type(exc).__name__}")
                continue  # e.g. no privilege on one schema: keep the others
            refs += [DatasetRef(schema, n, kind) for n, kind in names]
        if failures and len(failures) == len(schemas):
            raise SourceUnavailable(f"source {self.source_id}: listing failed ({'; '.join(failures)[:300]})")
        return [r for r in refs if self.allowed(r)]

    def describe(self, ref: DatasetRef) -> DatasetSchema:
        insp = self._inspector
        cols = [
            ColumnInfo(c["name"], str(c["type"]), bool(c.get("nullable", True)), i)
            for i, c in enumerate(insp.get_columns(ref.name, schema=ref.schema))
        ]
        keys: list[DeclaredKey] = []
        if ref.kind == "table":
            pk = insp.get_pk_constraint(ref.name, schema=ref.schema) or {}
            if pk.get("constrained_columns"):
                keys.append(DeclaredKey("primary", list(pk["constrained_columns"]), pk.get("name")))
            seen = set()
            for uc in insp.get_unique_constraints(ref.name, schema=ref.schema):
                cols_t = tuple(uc["column_names"])
                if cols_t and cols_t not in seen:
                    seen.add(cols_t)
                    keys.append(DeclaredKey("unique", list(cols_t), uc.get("name")))
            for ix in insp.get_indexes(ref.name, schema=ref.schema):
                cols_t = tuple(c for c in ix.get("column_names") or [] if c)
                if ix.get("unique") and cols_t and cols_t not in seen:
                    seen.add(cols_t)
                    keys.append(DeclaredKey("unique", list(cols_t), ix.get("name")))
            for fk in insp.get_foreign_keys(ref.name, schema=ref.schema):
                if fk.get("constrained_columns") and fk.get("referred_table"):
                    keys.append(DeclaredKey(
                        "foreign", list(fk["constrained_columns"]), fk.get("name"),
                        fk.get("referred_schema") or ref.schema, fk["referred_table"], list(fk["referred_columns"]),
                    ))
        row_count = None
        try:
            with self.engine.connect() as conn:
                row_count = conn.execute(select(func.count()).select_from(sql_table(ref.name, schema=ref.schema))).scalar()
        except SQLAlchemyError:
            row_count = None
        physical = f"{self.engine.url.get_backend_name()}:{ref.qualified}"
        return DatasetSchema(ref, physical, cols, keys, row_count)

    def read_sample(self, ref: DatasetRef, columns: list[str], limit: int | None) -> pa.Table:
        stmt = select(*[sql_column(c) for c in columns]).select_from(sql_table(ref.name, schema=ref.schema))
        if limit:
            stmt = stmt.limit(limit)
        try:
            with self.engine.connect() as conn:
                rows = [tuple(r) for r in conn.execute(stmt)]
        except SQLAlchemyError as exc:
            raise SourceUnavailable(f"source {self.source_id}: read failed ({type(exc).__name__})") from exc
        return rows_to_table(columns, rows)

    def execute_read_only(self, sql: str, max_rows: int, timeout_seconds: float) -> QueryResult:
        """One query on the read-only engine (sessions are forced read-only at connect). The SQL goes to the
        driver as is (no bind-parameter parsing); a statement timeout is set where the dialect supports it."""
        backend = self.engine.url.get_backend_name()
        ms = max(1, int(timeout_seconds * 1000))
        with self.engine.connect() as conn:
            raw = conn.connection.dbapi_connection
            try:
                if backend == "postgresql":
                    conn.exec_driver_sql(f"SET LOCAL statement_timeout = {ms}")
                elif backend == "mysql":
                    try:  # MySQL only; MariaDB has no such variable (the read-only session still applies)
                        conn.exec_driver_sql(f"SET SESSION MAX_EXECUTION_TIME = {ms}")
                    except SQLAlchemyError:
                        pass
                elif backend == "sqlite":
                    deadline = time.monotonic() + timeout_seconds
                    raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
                result = conn.exec_driver_sql(sql)
                names = list(result.keys())
                description = result.cursor.description if result.cursor is not None else None
                types = [_type_name(d[1]) if description else None for d in (description or [None] * len(names))]
                rows = [tuple(r) for r in result.fetchmany(max_rows + 1)]
            finally:
                if backend == "sqlite":
                    raw.set_progress_handler(None, 0)
                conn.rollback()
        return QueryResult(list(zip(names, types)), rows[:max_rows], len(rows) > max_rows)

    def close(self) -> None:
        self.engine.dispose()


def _type_name(code) -> str | None:
    if code is None:
        return None
    if isinstance(code, type):
        return code.__name__
    return str(code)

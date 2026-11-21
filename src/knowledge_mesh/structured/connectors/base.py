"""Connector interface for structured sources (relational databases, Parquet, Databricks)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from ...common.errors import SourceUnavailable


@dataclass(frozen=True)
class DatasetRef:
    schema: str | None
    name: str
    kind: str = "table"  # table | view | file

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool | None
    ordinal: int


@dataclass
class DeclaredKey:
    key_type: str  # primary | unique | foreign
    columns: list[str]
    name: str | None = None
    ref_schema: str | None = None
    ref_table: str | None = None
    ref_columns: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    columns: list[tuple[str, str | None]]   # (name, driver type name when known)
    rows: list[tuple[Any, ...]]
    truncated: bool


@dataclass
class DatasetSchema:
    ref: DatasetRef
    physical_ref: str
    columns: list[ColumnInfo]
    keys: list[DeclaredKey] = field(default_factory=list)
    row_count: int | None = None


class Connector(ABC):
    """Read-only access to one configured source. Implementations must never write."""

    def __init__(self, settings, secrets=None):
        self.settings = settings
        self.source_id = settings.id
        self.secrets = secrets

    @abstractmethod
    def list_datasets(self) -> list[DatasetRef]: ...

    @abstractmethod
    def describe(self, ref: DatasetRef) -> DatasetSchema: ...

    @abstractmethod
    def read_sample(self, ref: DatasetRef, columns: list[str], limit: int | None) -> pa.Table:
        """Return up to ``limit`` rows (all rows when None) of ``columns``. Values never leave the process."""

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def execute_read_only(self, sql: str, max_rows: int, timeout_seconds: float) -> "QueryResult":
        """Run one already validated read-only query (``execute_sql``); at most ``max_rows`` rows are returned."""
        from ...common.errors import NotConfigured

        raise NotConfigured(f"source {self.source_id} ({self.settings.kind}) does not support SQL execution")

    def scanned_schemas(self) -> list[str] | None:
        """Schemas ``list_datasets`` looks in (None when the source has no schema concept)."""
        return None

    def schema_overview(self) -> list[tuple[str, int]] | None:
        """``(schema, tables + views)`` for each non-system schema with tables; None if unsupported."""
        return None

    def allowed(self, ref: DatasetRef) -> bool:
        allow = self.settings.datasets
        if not allow:
            return True
        return ref.name in allow or ref.qualified in allow

    def find(self, name: str) -> DatasetRef:
        for ref in self.list_datasets():
            if name in (ref.name, ref.qualified):
                return ref
        raise SourceUnavailable(f"dataset {name} not found in source {self.source_id}")


def resolve_credential(settings, field: str, purpose: str, secrets=None) -> str:
    """Value for ``field``: plain setting, else the ``<field>_env`` variable, else the secret store."""
    plain = getattr(settings, field, None)
    if plain:
        return str(plain)
    env_name = getattr(settings, f"{field}_env", None)
    if env_name and (settings.credential_store == "env" or field not in {"connection_url", "token"}):
        return env_required(env_name, purpose, settings.id)
    if settings.credential_store == "secret_store":
        value = secrets.get(settings.id, field) if secrets is not None else None
        if value:
            return value
        raise SourceUnavailable(f"source {settings.id}: {purpose} is not in the secret store; re-enter it in Settings")
    raise SourceUnavailable(f"source {settings.id}: no {purpose} configured")


def coverage_note(settings, connector: Connector, found: int) -> str | None:
    """Explain an empty result, or name schemas holding tables that were not scanned.

    Returns None when there is nothing to report. Never raises: the diagnostic must not turn a
    successful (or already failing) catalog run into a different error.
    """
    try:
        scanned = [s for s in (connector.scanned_schemas() or []) if s]
        overview = connector.schema_overview()
    except Exception:
        return "No tables or views found." if found == 0 else None
    database = getattr(settings, "database", None)
    target = f"database '{database}'" if database else f"source '{settings.id}'"
    shown = ", ".join(repr(s) for s in scanned[:8]) + (f" (+{len(scanned) - 8} more)" if len(scanned) > 8 else "")
    where = f"schema {shown} of {target}" if scanned else target
    others = [(s, n) for s, n in (overview or []) if s not in scanned]
    listing = ", ".join(f"{s} ({n})" for s, n in others[:20])
    if found == 0:
        msg = f"No tables or views found in {where}."
        if settings.datasets:
            msg += f" The dataset allowlist ({', '.join(settings.datasets[:10])}) may exclude them."
        if overview is not None:
            msg += (f" Schemas with tables this account can access: {listing}. Enter them in Settings → Schema."
                    if others else
                    " No schema in this database has tables this account can access: check the database name "
                    "and the account's privileges.")
        return msg
    if others and not getattr(settings, "schemas", None):
        return f"Schemas with tables that were not scanned: {listing}. Add them in Settings → Schema to catalog them."
    return None


def env_required(var: str | None, purpose: str, source_id: str) -> str:
    if not var:
        raise SourceUnavailable(f"source {source_id}: {purpose} environment variable is not configured")
    value = os.environ.get(var)
    if not value:
        raise SourceUnavailable(
            f"source {source_id}: environment variable {var} ({purpose}) is not set", {"env": var}
        )
    return value


def rows_to_table(columns: list[str], rows: list[tuple[Any, ...]]) -> pa.Table:
    arrays = []
    for i, _ in enumerate(columns):
        values = [r[i] for r in rows]
        try:
            arrays.append(pa.array(values))
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            arrays.append(pa.array([None if v is None else str(v) for v in values], type=pa.string()))
    return pa.table(arrays, names=columns) if columns else pa.table({})

"""Registry of structured sources: ``config/mesh.yaml`` (read-only) + the Settings page (editable).

Sources added from the Settings page are stored in SQLite (non-secret settings) and take
effect immediately — no restart. Their secrets (database connection URL, Databricks token)
go either to an environment variable reference or to the OS secret store, never to SQLite.
Parquet repositories must be under ``storage.allowed_source_roots``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from ..common.errors import Conflict, Forbidden, InvalidArgument, MeshError, NotConfigured, NotFound
from ..common.ids import utcnow
from ..common.paths import resolve_authorized_path
from ..common.secrets import MemorySecretStore, SecretStore, mask_secrets
from ..config import MeshConfig, StructuredSourceSettings
from ..storage.db import Database, dumps, loads
from .connectors import Connector, build_connector
from .connectors.base import coverage_note

# Secret fields per kind. Databases described by fields use "password"; full-URL ones "connection_url".
SECRET_FIELDS = {"database": ["password", "connection_url"], "databricks": ["token"], "parquet": []}


FIELD_LABELS = {
    "id": "Source ID", "description": "Description", "db_type": "Database type", "database": "Database name",
    "host": "Hostname", "port": "Port", "schemas": "Schema", "username": "Username", "password": "Password",
    "password_env": "Password environment variable", "connection_url": "Connection URL",
    "connection_url_env": "Connection URL environment variable", "path": "File or folder",
    "server_hostname": "Server hostname", "server_hostname_env": "Server hostname environment variable",
    "http_path": "HTTP path", "http_path_env": "HTTP path environment variable", "catalog": "Catalog",
    "schema": "Schema", "token": "Personal access token", "token_env": "Token environment variable",
    "datasets": "Datasets to include",
}
_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


def _required(fields: list[str]) -> InvalidArgument:
    labels = [FIELD_LABELS.get(f, f) for f in fields]
    return InvalidArgument(f"Required: {', '.join(dict.fromkeys(labels))}",
                           {"fields": fields, "errors": [f"{label} is required" for label in labels]})


def _missing_fields(kind: str, data: dict[str, Any], secrets: dict[str, str], field_mode: bool) -> list[str]:
    """Fields the user must fill in, by source type (and connection mode for databases)."""
    kept = data.get("credential_store") == "secret_store"  # a stored secret is reused
    missing: list[str] = []
    if not data.get("id"):
        missing.append("id")
    if kind == "database" and field_mode:
        missing += [f for f in ("db_type", "database", "host") if not data.get(f)]
    elif kind == "database":
        if not (secrets.get("connection_url") or data.get("connection_url_env") or kept):
            missing += ["connection_url", "connection_url_env"]
    elif kind == "parquet":
        if not data.get("path"):
            missing.append("path")
    elif kind == "databricks":
        for f in ("server_hostname", "http_path"):
            if not (data.get(f) or data.get(f"{f}_env")):
                missing.append(f)
        if not data.get("catalog"):
            missing.append("catalog")
        if not (secrets.get("token") or data.get("token_env") or kept):
            missing += ["token", "token_env"]
    return missing


def _secret_field(src: StructuredSourceSettings) -> str | None:
    if src.kind == "database":
        return "password" if src.host else "connection_url"
    return SECRET_FIELDS[src.kind][0] if SECRET_FIELDS[src.kind] else None
_SETTINGS_FIELDS = set(StructuredSourceSettings.model_fields) | {"schema"}


class SourceRegistry:
    def __init__(self, config: MeshConfig, db: Database, secrets: SecretStore,
                 connector_factory: Callable[..., Connector] = build_connector):
        self.config = config
        self.db = db
        self.secrets = secrets
        self.connector_factory = connector_factory
        for root in config.storage.allowed_source_roots:
            root.mkdir(parents=True, exist_ok=True)

    # -- reads ---------------------------------------------------------------------------------
    def _db_sources(self) -> list[StructuredSourceSettings]:
        out = []
        for row in self.db.query("SELECT settings_json FROM source_configs ORDER BY source_id"):
            try:
                out.append(StructuredSourceSettings.model_validate(loads(row["settings_json"])))
            except ValidationError:
                continue  # a row that no longer validates is skipped rather than breaking the catalog
        return out

    def all(self) -> list[StructuredSourceSettings]:
        config_ids = {s.id for s in self.config.structured_sources}
        return list(self.config.structured_sources) + [s for s in self._db_sources() if s.id not in config_ids]

    def get(self, source_id: str) -> StructuredSourceSettings | None:
        return next((s for s in self.all() if s.id == source_id), None)

    def require(self, source_id: str) -> StructuredSourceSettings:
        src = self.get(source_id)
        if src is None:
            raise NotFound(f"structured source not configured: {source_id}")
        return src

    def origin(self, source_id: str) -> str:
        return "config_file" if self.config.source(source_id) else "settings"

    def connector(self, source_id: str) -> Connector:
        return self.connector_factory(self.require(source_id), self.secrets)

    def describe(self, src: StructuredSourceSettings) -> dict[str, Any]:
        origin = self.origin(src.id)
        view = {"source_id": src.id, "kind": src.kind, "description": src.description, "origin": origin,
                "editable": origin == "settings", "target": self._target(src), "credentials": self._credentials(src),
                "settings": src.public_view()}
        if src.kind == "parquet":
            view["settings"]["path"] = str(src.path)
        return view

    @staticmethod
    def _target(src: StructuredSourceSettings) -> str:
        if src.kind == "database":
            if src.host:
                return src.connection_display or f"{src.db_type}://{src.host}:{src.port or ''}/{src.database}"
            return src.connection_display or f"${src.connection_url_env}"
        if src.kind == "parquet":
            return str(src.path)
        host = src.server_hostname or f"${src.server_hostname_env}"
        path = src.http_path or f"${src.http_path_env}"
        return f"{host} {path} · {src.catalog}.{src.schema_ or 'default'}"

    def _credentials(self, src: StructuredSourceSettings) -> dict[str, Any]:
        field = _secret_field(src)
        if field is None:
            return {"store": "none", "reference": None, "status": "not_required"}
        if field == "password" and src.credential_store == "env" and not src.password_env:
            return {"store": "none", "reference": None, "status": "no_password"}
        if src.credential_store == "secret_store":
            present = self.secrets.get(src.id, field) is not None
            return {"store": "secret_store", "reference": f"{self.secrets.backend}: {src.id}:{field}",
                    "status": "present" if present else "missing"}
        env = getattr(src, f"{field}_env")
        return {"store": "env", "reference": env, "status": "present" if os.environ.get(env or "") else "missing"}

    # -- writes --------------------------------------------------------------------------------
    def _build(self, payload: dict[str, Any], existing: StructuredSourceSettings | None
               ) -> tuple[StructuredSourceSettings, dict[str, str]]:
        payload = {k: v for k, v in payload.items() if v not in (None, "", [])}
        kind = payload.get("kind") or (existing.kind if existing else None)
        if kind not in SECRET_FIELDS:
            raise InvalidArgument("kind must be database, parquet or databricks")
        secrets: dict[str, str] = {}
        data: dict[str, Any] = {k: v for k, v in payload.items() if k in _SETTINGS_FIELDS}
        data["kind"] = kind
        data["credential_store"] = "env"
        # Only the secret used by this source's mode counts: a database described by fields uses a
        # password; one described by a URL uses the URL. The other mode's values are dropped.
        field_mode = kind == "database" and bool(data.get("host") or data.get("db_type") or data.get("database"))
        if kind == "database":
            unused = "connection_url" if field_mode else "password"
            data.pop(f"{unused}_env", None)
        active = ("password" if field_mode else "connection_url") if kind == "database" else \
            (SECRET_FIELDS[kind][0] if SECRET_FIELDS[kind] else None)
        if active:
            value = str(payload.get(active) or "").strip()
            if value:
                secrets[active] = value
                data["credential_store"] = "secret_store"
                data.pop(f"{active}_env", None)
            elif (not data.get(f"{active}_env") and existing and existing.credential_store == "secret_store"
                  and _secret_field(existing) == active):
                data["credential_store"] = "secret_store"  # keep the stored secret
        if field_mode:
            from .connectors.database import DB_TYPES, build_url

            if data.get("db_type") in DB_TYPES and data.get("database") and data.get("host"):
                has_password = data["credential_store"] == "secret_store" or bool(data.get("password_env"))
                data["connection_display"] = build_url(
                    data["db_type"], data["host"], data.get("port"), data["database"], data.get("username"),
                    "x" if has_password else None).render_as_string(hide_password=True)
        elif kind == "database":
            url = secrets.get("connection_url")
            if url:
                from sqlalchemy.engine import make_url

                try:
                    data["connection_display"] = make_url(url).render_as_string(hide_password=True)
                except Exception as exc:
                    raise InvalidArgument("connection URL is not a valid SQLAlchemy URL "
                                          "(e.g. sqlite:////absolute/path/crm.db)",
                                          {"fields": ["connection_url"]}) from exc
            elif data["credential_store"] == "secret_store" and existing:
                data["connection_display"] = existing.connection_display
        missing = _missing_fields(kind, data, secrets, field_mode)
        if missing:
            raise _required(missing)
        if data.get("id") and not _ID_RE.match(str(data["id"])):
            raise InvalidArgument("Source ID may only contain letters, digits, _ . - (max 64 characters)",
                                  {"fields": ["id"]})
        if kind == "parquet":
            roots = self.config.storage.allowed_source_roots
            try:
                data["path"] = str(resolve_authorized_path(data["path"], roots))
            except Forbidden as exc:
                raise Forbidden(
                    "Parquet path is outside the allowed source roots; add its parent directory to "
                    "storage.allowed_source_roots in config/mesh.yaml",
                    {"path": payload.get("path"), "allowed_roots": [str(r) for r in roots],
                     "fields": ["path"]}) from exc
        try:
            settings = StructuredSourceSettings.model_validate(data)
        except ValidationError as exc:
            fields = [str(e["loc"][0]) for e in exc.errors() if e.get("loc")]
            errors = [f"{FIELD_LABELS.get(str(e['loc'][0]), e['loc'][0])}: {e['msg']}" if e.get("loc")
                      else e["msg"].removeprefix("Value error, ") for e in exc.errors()]
            raise InvalidArgument("; ".join(errors)[:500], {"errors": errors, "fields": fields}) from exc
        if secrets and not self.secrets.available:
            raise NotConfigured(f"the secret store is unavailable ({getattr(self.secrets, 'reason', '')}); "
                                "reference an environment variable instead")
        return settings, secrets

    def create(self, payload: dict[str, Any], created_by: str | None = None) -> dict[str, Any]:
        source_id = str(payload.get("id") or "").strip()
        if source_id and self.get(source_id) is not None:
            raise Conflict(f"a source with id {source_id!r} already exists", {"fields": ["id"]})
        settings, secrets = self._build(payload, None)
        self._save(settings, secrets, created_by, insert=True)
        return self.describe(settings)

    def update(self, source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        existing = self.require(source_id)
        if self.origin(source_id) == "config_file":
            raise Forbidden("this source is defined in config/mesh.yaml; edit the file and restart to change it")
        payload = {**payload, "id": source_id, "kind": existing.kind}
        settings, secrets = self._build(payload, existing)
        keep = _secret_field(settings) if settings.credential_store == "secret_store" else None
        for field in SECRET_FIELDS[settings.kind]:
            if field != keep:
                self.secrets.delete(source_id, field)  # secret no longer used (env var, no password, other mode)
        self._save(settings, secrets, None, insert=False)
        return self.describe(settings)

    def _save(self, settings: StructuredSourceSettings, secrets: dict[str, str], created_by: str | None,
              insert: bool) -> None:
        for field, value in secrets.items():
            self.secrets.set(settings.id, field, value)
        data = dumps(settings.model_dump(mode="json", by_alias=True, exclude_none=True))
        now = utcnow()
        if insert:
            self.db.execute("INSERT INTO source_configs(source_id, kind, settings_json, created_by, created_at, "
                            "updated_at) VALUES (?,?,?,?,?,?)",
                            (settings.id, settings.kind, data, created_by, now, now))
        else:
            self.db.execute("UPDATE source_configs SET settings_json = ?, updated_at = ? WHERE source_id = ?",
                            (data, now, settings.id))
        self.db.execute("INSERT INTO structured_sources(source_id, kind, status) VALUES (?,?,?) "
                        "ON CONFLICT(source_id) DO UPDATE SET kind = excluded.kind",
                        (settings.id, settings.kind, "configured"))

    def delete(self, source_id: str) -> StructuredSourceSettings:
        src = self.require(source_id)
        if self.origin(source_id) == "config_file":
            raise Forbidden("this source is defined in config/mesh.yaml; remove it there and restart")
        for field in SECRET_FIELDS[src.kind]:
            self.secrets.delete(source_id, field)
        self.db.execute("DELETE FROM source_configs WHERE source_id = ?", (source_id,))
        return src

    # -- connection test -------------------------------------------------------------------------
    def test(self, payload: dict[str, Any] | None = None, source_id: str | None = None) -> dict[str, Any]:
        """Connect and list datasets without saving anything. Error messages never contain secrets."""
        secrets: dict[str, str] = {}
        try:
            if source_id:
                settings = self.require(source_id)
                store: SecretStore = self.secrets
            else:
                payload = dict(payload or {})
                payload["id"] = payload.get("id") or "connection-test"
                existing = self.get(payload["id"]) if self.origin(payload["id"]) == "settings" else None
                settings, secrets = self._build(payload, existing)
                store = MemorySecretStore({(settings.id, k): v for k, v in secrets.items()}, fallback=self.secrets)
            connector = self.connector_factory(settings, store)
            try:
                refs = connector.list_datasets()
                note = coverage_note(settings, connector, len(refs))
            finally:
                connector.close()
        except MeshError as exc:
            error = {"code": exc.code, "message": mask_secrets(exc.message, list(secrets.values()))}
            if exc.details.get("fields"):
                error["fields"] = exc.details["fields"]
            return {"ok": False, "error": error}
        except Exception as exc:  # driver-specific failures
            return {"ok": False, "error": {"code": "connection_failed", "message": mask_secrets(
                f"{type(exc).__name__}: {str(exc)[:300]}", list(secrets.values()))}}
        out = {"ok": True, "dataset_count": len(refs), "datasets": [r.qualified for r in refs[:25]]}
        if note:
            out["warning"] = note
        return out


def describe_roots(config: MeshConfig) -> list[str]:
    return [str(Path(r)) for r in config.storage.allowed_source_roots]

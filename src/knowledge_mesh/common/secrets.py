"""Secret store for connector credentials entered on the Settings page.

Secrets (database connection URLs, Databricks tokens) live in the OS secret store via
``keyring`` — the macOS Keychain, Windows Credential Locker or Linux Secret Service — under
service ``secrets.service_name`` and account ``<source_id>:<field>``. They are never written
to SQLite, config files, logs or API responses.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Protocol

from .errors import NotConfigured

log = logging.getLogger(__name__)
_URL_PASSWORD = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")


def mask_secrets(text: str, secrets: list[str] | None = None) -> str:
    """Remove known secret values and URL passwords from a message before showing or logging it."""
    for secret in secrets or []:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***")
    return _URL_PASSWORD.sub(r"\1***@", text)


class SecretStore(Protocol):
    backend: str

    @property
    def available(self) -> bool: ...

    def get(self, source_id: str, field: str) -> str | None: ...

    def set(self, source_id: str, field: str, value: str) -> None: ...

    def delete(self, source_id: str, field: str) -> None: ...


class KeyringSecretStore:
    backend = "keyring"

    def __init__(self, service_name: str):
        self.service = service_name
        self._keyring = None
        self._reason: str | None = None
        try:
            import keyring
            from keyring.backends import fail

            kr = keyring.get_keyring()
            if isinstance(kr, fail.Keyring):
                self._reason = "no OS secret store backend is available"
            else:
                self._keyring = keyring
                self.backend = f"keyring ({type(kr).__module__.rsplit('.', 1)[-1]})"
        except ImportError:
            self._reason = "the keyring package is not installed"

    @property
    def available(self) -> bool:
        return self._keyring is not None

    @property
    def reason(self) -> str | None:
        return self._reason

    def _require(self):
        if self._keyring is None:
            raise NotConfigured(f"secret store unavailable: {self._reason}; use an environment variable instead")
        return self._keyring

    def get(self, source_id: str, field: str) -> str | None:
        if self._keyring is None:
            return None
        try:
            return self._keyring.get_password(self.service, f"{source_id}:{field}")
        except Exception as exc:  # keychain locked, access denied, ...
            log.warning("secret store read failed", extra={"source_id": source_id, "error": type(exc).__name__})
            return None

    def set(self, source_id: str, field: str, value: str) -> None:
        self._require().set_password(self.service, f"{source_id}:{field}", value)

    def delete(self, source_id: str, field: str) -> None:
        if self._keyring is None:
            return
        try:
            self._keyring.delete_password(self.service, f"{source_id}:{field}")
        except Exception:  # already absent
            pass


class MemorySecretStore:
    """In-process store for tests and for testing a connection before saving it."""

    backend = "memory"

    def __init__(self, initial: dict[tuple[str, str], str] | None = None, fallback: "SecretStore | None" = None):
        self._data = dict(initial or {})
        self._fallback = fallback
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return True

    def get(self, source_id: str, field: str) -> str | None:
        with self._lock:
            if (source_id, field) in self._data:
                return self._data[(source_id, field)]
        return self._fallback.get(source_id, field) if self._fallback else None

    def set(self, source_id: str, field: str, value: str) -> None:
        with self._lock:
            self._data[(source_id, field)] = value

    def delete(self, source_id: str, field: str) -> None:
        with self._lock:
            self._data.pop((source_id, field), None)


class NullSecretStore:
    backend = "none"
    reason = "secret store disabled in configuration (secrets.backend: none)"

    @property
    def available(self) -> bool:
        return False

    def get(self, source_id: str, field: str) -> str | None:
        return None

    def set(self, source_id: str, field: str, value: str) -> None:
        raise NotConfigured(f"{self.reason}; use an environment variable instead")

    def delete(self, source_id: str, field: str) -> None:
        return None


def build_secret_store(settings) -> SecretStore:
    if settings.backend == "none":
        return NullSecretStore()
    return KeyringSecretStore(settings.service_name)

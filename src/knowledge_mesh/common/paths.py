"""Filesystem boundary checks for local ingestion and uploads."""

from __future__ import annotations

import re
from pathlib import Path

from .errors import Forbidden, InvalidArgument, NotFound

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def resolve_authorized_path(path: str | Path, roots: list[Path]) -> Path:
    """Resolve ``path`` (following symlinks) and require it to live under an authorized root."""
    if not str(path).strip():
        raise InvalidArgument("path is required")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        # Relative paths are interpreted against the first authorized root.
        if not roots:
            raise Forbidden("no ingestion roots are configured")
        candidate = roots[0] / candidate
    resolved = candidate.resolve()
    for root in roots:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            if not resolved.exists():
                raise NotFound(f"path not found: {path}")
            return resolved
    raise Forbidden(
        "path is outside the authorized ingestion roots",
        {"path": str(path), "allowed_roots": [str(r) for r in roots]},
    )


def safe_filename(name: str, default: str = "document") -> str:
    """Strip directory components and unsafe characters from a user-supplied filename."""
    base = Path(str(name).replace("\\", "/")).name
    base = _SAFE_NAME.sub("_", base).strip(" .")
    if not base or base in {".", ".."}:
        base = default
    return base[:180]

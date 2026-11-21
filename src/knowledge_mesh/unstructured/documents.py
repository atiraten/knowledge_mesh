"""Document registry: originals, extracted text, provenance and dedup state in SQLite.

* ``document_id`` is derived from the source URI, so updating a source keeps its ID.
* Unchanged content (same hash) is a no-op; identical content under another active source
  is recorded as a skipped duplicate so it never produces duplicate embeddings.
* Originals are written with 0600 permissions under ``storage.originals_dir`` and are only
  ever served back as downloads, never executed or rendered inline.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..common.errors import ExtractionFailed, InvalidArgument, NotFound, PayloadTooLarge, UnsupportedMediaType
from ..common.ids import document_id_for, sha256_bytes, utcnow
from ..common.paths import resolve_authorized_path, safe_filename
from ..observability import get_metrics, span
from ..storage.db import Database, dumps, loads
from .loaders import MIME_BY_EXT, ExtractedDocument, detect_type, extract

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: str
    status: str  # new | updated | unchanged | duplicate | failed
    reason: str | None = None
    duplicate_of: str | None = None
    extracted: ExtractedDocument | None = None

    @property
    def needs_indexing(self) -> bool:
        return self.status in {"new", "updated"}

    def to_dict(self) -> dict[str, Any]:
        out = {"document_id": self.document_id, "status": self.status}
        if self.reason:
            out["reason"] = self.reason
        if self.duplicate_of:
            out["duplicate_of"] = self.duplicate_of
        return out


class DocumentService:
    def __init__(self, storage_settings, crawler_settings, db: Database):
        self.cfg = storage_settings
        self.dedupe_content = crawler_settings.deduplicate_content_sha256
        self.db = db
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.cfg.originals_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.extracted_dir.mkdir(parents=True, exist_ok=True)
        for root in self.cfg.allowed_ingest_roots:
            root.mkdir(parents=True, exist_ok=True)

    # -- ingestion -------------------------------------------------------------------
    def ingest_bytes(self, data: bytes, filename: str, *, source_type: str, source_uri: str,
                     declared_mime: str | None = None, metadata: dict[str, Any] | None = None,
                     max_bytes: int | None = None) -> IngestResult:
        metrics = get_metrics()
        limit = max_bytes or self.cfg.max_upload_bytes
        if len(data) > limit:
            raise PayloadTooLarge(f"document exceeds {limit} bytes", {"size": len(data), "limit": limit})
        filename = safe_filename(filename)
        document_id = document_id_for(source_uri)
        ext, mime = detect_type(filename, data, declared_mime, self.cfg.allowed_extensions)
        content_hash = sha256_bytes(data)
        # Concurrent crawl workers may fetch identical content under different URLs: the dedup
        # check and the registration of the canonical copy must be atomic per content hash.
        with self._lock_for(content_hash), self._lock_for(document_id):
            return self._ingest_locked(data, filename, ext, mime, content_hash, document_id, source_type,
                                       source_uri, metadata, metrics)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                if len(self._locks) > 10_000:
                    self._locks = {k: v for k, v in self._locks.items() if v.locked()}
                lock = self._locks[key] = threading.Lock()
            return lock

    def _ingest_locked(self, data: bytes, filename: str, ext: str, mime: str, content_hash: str, document_id: str,
                       source_type: str, source_uri: str, metadata: dict[str, Any] | None, metrics) -> IngestResult:
        with span("document.ingest", "CHAIN", {"document.id": document_id, "document.source_type": source_type}) as sp:
            existing = self.db.query_one("SELECT * FROM documents WHERE document_id = ?", (document_id,))
            if existing and existing["active"] and existing["content_hash"] == content_hash \
                    and existing["status"] in {"indexed", "extracted"}:
                status = "unchanged" if existing["status"] == "indexed" else "updated"
                sp.set_attribute("document.status", status)
                metrics.documents.add(1, {"status": status, "source_type": source_type})
                return IngestResult(document_id, status, "content unchanged" if status == "unchanged" else None)

            if self.dedupe_content:
                dup = self.db.query_one(
                    "SELECT document_id FROM documents WHERE content_hash = ? AND document_id != ? AND active = 1 "
                    "AND status IN ('indexed','extracted') LIMIT 1",
                    (content_hash, document_id),
                )
                if dup:
                    self._upsert(document_id, source_type, source_uri, filename, mime, content_hash, len(data),
                                 None, None, "skipped", "duplicate_content", metadata, None, dup["document_id"])
                    metrics.documents.add(1, {"status": "duplicate", "source_type": source_type})
                    sp.set_attribute("document.status", "duplicate")
                    return IngestResult(document_id, "duplicate", "identical content already indexed",
                                        duplicate_of=dup["document_id"])

            try:
                with span("document.extract", "CHAIN", {"document.mime_type": mime}):
                    extracted = extract(data, mime, filename)
            except ExtractionFailed as exc:
                self._upsert(document_id, source_type, source_uri, filename, mime, content_hash, len(data),
                             None, None, "failed", exc.message, metadata, None, None)
                metrics.documents.add(1, {"status": "failed", "source_type": source_type})
                sp.set_attribute("document.status", "failed")
                return IngestResult(document_id, "failed", exc.message)

            original_path = self._store_original(document_id, ext, data, existing)
            extracted_path = self.cfg.extracted_dir / f"{document_id}.json"
            _atomic_write(extracted_path, extracted.to_json().encode("utf-8"))
            status = "updated" if existing else "new"
            self._upsert(document_id, source_type, source_uri, filename, mime, content_hash, len(data),
                         original_path, extracted_path, "extracted", None, metadata, extracted, None)
            metrics.documents.add(1, {"status": status, "source_type": source_type})
            sp.set_attribute("document.status", status)
            log.info("document ingested", extra={"document_id": document_id, "status": status, "mime_type": mime,
                                                 "sections": len(extracted.sections)})
            return IngestResult(document_id, status, extracted=extracted)

    def ingest_path(self, path: str | Path, metadata: dict[str, Any] | None = None) -> IngestResult:
        resolved = resolve_authorized_path(path, self.cfg.allowed_ingest_roots)
        if not resolved.is_file():
            raise InvalidArgument("path is not a file", {"path": str(path)})
        size = resolved.stat().st_size
        if size > self.cfg.max_upload_bytes:
            raise PayloadTooLarge(f"file exceeds {self.cfg.max_upload_bytes} bytes", {"path": str(path)})
        return self.ingest_bytes(resolved.read_bytes(), resolved.name, source_type="file",
                                 source_uri=resolved.as_uri(), metadata=metadata)

    def iter_directory(self, path: str | Path) -> Iterator[Path]:
        root = resolve_authorized_path(path, self.cfg.allowed_ingest_roots)
        if not root.is_dir():
            raise InvalidArgument("path is not a directory", {"path": str(path)})
        for p in sorted(root.rglob("*")):
            if p.name.startswith(".") or any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            if p.is_file():
                yield p

    def _store_original(self, document_id: str, ext: str, data: bytes, existing: dict | None) -> Path:
        target = self.cfg.originals_dir / f"{document_id}{ext}"
        if existing and existing.get("original_path") and existing["original_path"] != str(target):
            Path(existing["original_path"]).unlink(missing_ok=True)
        _atomic_write(target, data)
        return target

    def _upsert(self, document_id, source_type, source_uri, filename, mime, content_hash, size,
                original_path, extracted_path, status, reason, metadata, extracted: ExtractedDocument | None,
                duplicate_of) -> None:
        now = utcnow()
        self.db.execute(
            """INSERT INTO documents(document_id, source_type, source_uri, title, filename, mime_type, content_hash,
                   size_bytes, original_path, extracted_path, status, status_reason, duplicate_of, page_count,
                   section_count, metadata_json, ingested_at, updated_at, active)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
               ON CONFLICT(document_id) DO UPDATE SET
                   source_type=excluded.source_type, source_uri=excluded.source_uri,
                   title=COALESCE(excluded.title, documents.title), filename=excluded.filename,
                   mime_type=excluded.mime_type, content_hash=excluded.content_hash, size_bytes=excluded.size_bytes,
                   original_path=COALESCE(excluded.original_path, documents.original_path),
                   extracted_path=COALESCE(excluded.extracted_path, documents.extracted_path),
                   status=excluded.status, status_reason=excluded.status_reason, duplicate_of=excluded.duplicate_of,
                   page_count=excluded.page_count, section_count=excluded.section_count,
                   metadata_json=excluded.metadata_json, updated_at=excluded.updated_at, active=1""",
            (document_id, source_type, source_uri, extracted.title if extracted else None, filename, mime,
             content_hash, size, str(original_path) if original_path else None,
             str(extracted_path) if extracted_path else None, status, reason, duplicate_of,
             extracted.page_count if extracted else None, len(extracted.sections) if extracted else None,
             dumps(metadata or {}), now, now),
        )

    # -- state transitions --------------------------------------------------------------
    def mark_indexed(self, document_id: str, node_count: int, leaf_count: int, index_version: str) -> None:
        self.db.execute(
            "UPDATE documents SET status='indexed', status_reason=NULL, node_count=?, leaf_count=?, index_version=?,"
            " indexed_content_hash=content_hash, indexed_at=?, updated_at=? WHERE document_id=?",
            (node_count, leaf_count, index_version, utcnow(), utcnow(), document_id),
        )

    def mark_failed(self, document_id: str, reason: str) -> None:
        self.db.execute("UPDATE documents SET status='failed', status_reason=?, updated_at=? WHERE document_id=?",
                        (reason[:500], utcnow(), document_id))

    def mark_deleted(self, document_id: str) -> None:
        row = self.get_row(document_id)
        for key in ("original_path", "extracted_path"):
            if row.get(key):
                Path(row[key]).unlink(missing_ok=True)
        self.db.execute(
            "UPDATE documents SET status='deleted', active=0, node_count=0, leaf_count=0, original_path=NULL,"
            " extracted_path=NULL, updated_at=? WHERE document_id=?",
            (utcnow(), document_id),
        )

    # -- reads ---------------------------------------------------------------------------
    def get_row(self, document_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM documents WHERE document_id = ?", (document_id,))
        if row is None:
            raise NotFound(f"document not found: {document_id}")
        return row

    def get(self, document_id: str) -> dict[str, Any]:
        return self._public(self.get_row(document_id))

    def list(self, offset: int = 0, limit: int = 50, status: str | None = None) -> dict[str, Any]:
        limit = max(1, min(limit, 200))
        where, params = "active = 1", []
        if status:
            where = "status = ?"
            params.append(status)
        total = self.db.scalar(f"SELECT COUNT(*) FROM documents WHERE {where}", params)
        rows = self.db.query(
            f"SELECT * FROM documents WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            [*params, limit, max(0, offset)],
        )
        return {"total": total, "offset": offset, "limit": limit, "documents": [self._public(r) for r in rows]}

    def load_extracted(self, document_id: str) -> ExtractedDocument:
        row = self.get_row(document_id)
        if not row.get("extracted_path") or not Path(row["extracted_path"]).exists():
            raise NotFound(f"extracted text not available for {document_id}", {"status": row["status"]})
        return ExtractedDocument.from_json(Path(row["extracted_path"]).read_text(encoding="utf-8"))

    def original_file(self, document_id: str) -> tuple[Path, str, str]:
        row = self.get_row(document_id)
        if not row["active"] or not row.get("original_path") or not Path(row["original_path"]).exists():
            raise NotFound(f"original not available for {document_id}")
        return Path(row["original_path"]), row["mime_type"] or "application/octet-stream", row["filename"] or "document"

    @staticmethod
    def _public(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "document_id": row["document_id"],
            "source_type": row["source_type"],
            "source_uri": row["source_uri"],
            "title": row["title"],
            "filename": row["filename"],
            "mime_type": row["mime_type"],
            "content_hash": row["content_hash"],
            "size_bytes": row["size_bytes"],
            "status": row["status"],
            "status_reason": row["status_reason"],
            "duplicate_of": row["duplicate_of"],
            "page_count": row["page_count"],
            "section_count": row["section_count"],
            "node_count": row["node_count"],
            "leaf_count": row["leaf_count"],
            "index_version": row["index_version"],
            "ingested_at": row["ingested_at"],
            "updated_at": row["updated_at"],
            "indexed_at": row["indexed_at"],
            "metadata": loads(row["metadata_json"], {}),
            "original_ref": f"/api/documents/{row['document_id']}/original" if row.get("original_path") else None,
        }

    @staticmethod
    def supported_extension(path: Path) -> bool:
        return path.suffix.lower() in MIME_BY_EXT


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


__all__ = ["DocumentService", "IngestResult", "UnsupportedMediaType"]

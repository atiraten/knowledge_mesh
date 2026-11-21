"""Structured error types shared by services, the MCP server and the HTTP API."""

from __future__ import annotations

from typing import Any


class MeshError(Exception):
    code = "mesh_error"
    http_status = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return body


class InvalidArgument(MeshError):
    code = "invalid_argument"
    http_status = 400


class BlockedURL(MeshError):
    code = "blocked_url"
    http_status = 400


class Forbidden(MeshError):
    code = "forbidden"
    http_status = 403


class NotFound(MeshError):
    code = "not_found"
    http_status = 404


class Conflict(MeshError):
    code = "conflict"
    http_status = 409


class PayloadTooLarge(MeshError):
    code = "payload_too_large"
    http_status = 413


class UnsupportedMediaType(MeshError):
    code = "unsupported_media_type"
    http_status = 415


class ExtractionFailed(MeshError):
    code = "extraction_failed"
    http_status = 422


class GraphValidationError(MeshError):
    code = "graph_validation_failed"
    http_status = 422


class IncompatibleIndex(MeshError):
    code = "incompatible_index"
    http_status = 409


class NotConfigured(MeshError):
    code = "not_configured"
    http_status = 503


class SourceUnavailable(MeshError):
    code = "source_unavailable"
    http_status = 503


class OperationFailed(MeshError):
    code = "operation_failed"
    http_status = 500


class TransientError(MeshError):
    """Retryable failure (network hiccup, rate limit, timeout)."""

    code = "transient_failure"
    http_status = 503

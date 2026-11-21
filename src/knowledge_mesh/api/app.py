"""FastAPI application: JSON API + lightweight browser UI.

* Every response carries ``X-Trace-Id``; errors use ``{"error": {code, message, details}, "trace_id"}``.
* If the environment variable named by ``api.api_key_env`` is set, ``/api/*`` requires the
  ``X-API-Key`` header (or ``Authorization: Bearer``). Bind to localhost otherwise.
* Uploads enforce the configured size limit and allowed types; filenames are sanitized.
* Originals are served with ``nosniff`` and a sandbox CSP; HTML originals are never rendered inline.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from opentelemetry import context as otel_context
from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..common.errors import ExtractionFailed, Forbidden, MeshError, PayloadTooLarge
from ..config import load_config
from ..mesh import KnowledgeMesh
from ..observability import current_trace_id, get_metrics, tracer
from ..observability.audit import RequestContext, caller_identity

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


# -- request models -----------------------------------------------------------------------------
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CrawlRequest(_Model):
    url: str = Field(min_length=1, max_length=2048)
    max_depth: int | None = Field(default=None, ge=1, le=10)
    max_pages: int | None = Field(default=None, ge=1, le=10_000)


class DirectoryRequest(_Model):
    path: str | None = Field(default=None, max_length=4096)


class CatalogRequest(_Model):
    datasets: list[str] | None = Field(default=None, max_length=1000)


class KeysRequest(_Model):
    dataset_ids: list[str] = Field(min_length=1, max_length=200)
    max_rows: int | None = Field(default=None, ge=0)


class TermsRequest(_Model):
    dataset_ids: list[str] = Field(min_length=1, max_length=200)
    force: bool = False


class IndexRequest(_Model):
    document_ids: list[str] = Field(min_length=1, max_length=1000)
    chunk_sizes: list[int] | None = None
    force: bool = False


class DefinitionRequest(_Model):
    name: str = Field(min_length=1, max_length=200)
    definition: str = Field(max_length=5000)
    entities: list[str] = Field(default_factory=list, max_length=50)
    terms: list[str] = Field(default_factory=list, max_length=50)
    author: str | None = Field(default=None, max_length=200)
    source_ref: str | None = Field(default=None, max_length=500)


class SourcePayload(_Model):
    """A structured source from the Settings page. ``password``, ``connection_url`` and ``token`` are
    secrets: they go to the OS secret store and are never returned."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(default="", max_length=64)
    kind: str | None = None
    description: str | None = Field(default=None, max_length=500)
    db_type: str | None = Field(default=None, max_length=32)
    host: str | None = Field(default=None, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    database: str | None = Field(default=None, max_length=255)
    username: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, max_length=1000, repr=False)
    password_env: str | None = Field(default=None, max_length=128)
    connection_url_env: str | None = Field(default=None, max_length=128)
    connection_url: str | None = Field(default=None, max_length=4000, repr=False)
    schemas: list[str] | None = Field(default=None, max_length=100)
    path: str | None = Field(default=None, max_length=4096)
    server_hostname: str | None = Field(default=None, max_length=255)
    server_hostname_env: str | None = Field(default=None, max_length=128)
    http_path: str | None = Field(default=None, max_length=512)
    http_path_env: str | None = Field(default=None, max_length=128)
    token_env: str | None = Field(default=None, max_length=128)
    token: str | None = Field(default=None, max_length=4000, repr=False)
    catalog: str | None = Field(default=None, max_length=255)
    schema_: str | None = Field(default=None, alias="schema", max_length=255)
    datasets: list[str] | None = Field(default=None, max_length=1000)

    def data(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}
ALERT_TOKEN_ENV = "KNOWLEDGE_MESH_ALERT_TOKEN"
_SESSION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class FeedbackRequest(_Model):
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


def _supplied_key(request: Request) -> str | None:
    supplied = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = supplied or auth[7:]
    return supplied or None


def request_context(request: Request) -> RequestContext:
    """Caller identity (client address + API-key fingerprint) and the UI session, for traces and audit."""
    session = request.headers.get("x-session-id") or ""
    host = request.client.host if request.client else None
    return RequestContext(caller=caller_identity(host, _supplied_key(request)),
                          session_id=session if _SESSION_RE.match(session) else None)


class SearchRequest(_Model):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = None
    evidence_types: list[str] | None = None
    document_ids: list[str] | None = None
    source_types: list[str] | None = None
    structured_source_ids: list[str] | None = None
    dataset_ids: list[str] | None = None
    expand_parents: bool | None = None
    rerank: bool | None = None  # None = retrieval.reranker_enabled from config
    output_format: Literal["text", "html"] = "text"  # /api/search: "html" asks the agent for generate_html output


class ClarificationReply(_Model):
    reply: str = Field(min_length=1, max_length=4000)


class ReviewRequest(_Model):
    object_type: Literal["foreign_key", "relationship", "term_relation", "entity_mapping", "term_mapping"]
    object_id: str = Field(min_length=1, max_length=200)
    status: Literal["verified", "rejected", "inferred"]
    note: str | None = Field(default=None, max_length=1000)


class RelationshipsRequest(_Model):
    dataset_ids: list[str] = Field(min_length=1, max_length=200)


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}, "trace_id": current_trace_id()}
    if details:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status)


def create_app(mesh: KnowledgeMesh | None = None, config_path: str | None = None) -> FastAPI:
    owned = mesh is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.mesh is None:
            app.state.mesh = KnowledgeMesh(load_config(config_path))
        threading.Thread(target=app.state.mesh.warmup, name="mesh-warmup", daemon=True).start()
        yield
        if owned:
            app.state.mesh.close()

    app = FastAPI(title="Knowledge Mesh", version=__version__, lifespan=lifespan)
    app.state.mesh = mesh

    def m() -> KnowledgeMesh:
        return app.state.mesh

    # -- middleware: tracing, metrics, auth, security headers ------------------------------------------
    @app.middleware("http")
    async def telemetry_middleware(request: Request, call_next):
        token = otel_context.attach(extract(dict(request.headers)))
        metrics = get_metrics()
        start = time.perf_counter()
        try:
            with tracer().start_as_current_span(f"HTTP {request.method}", kind=SpanKind.SERVER) as sp:
                sp.set_attribute("openinference.span.kind", "CHAIN")
                sp.set_attribute("http.method", request.method)
                ctx = request_context(request)
                sp.set_attribute("user.id", ctx.caller)
                if ctx.session_id:
                    sp.set_attribute("session.id", ctx.session_id)
                response = _auth_check(request)
                if response is None:
                    try:
                        response = await call_next(request)
                    except Exception:
                        log.exception("unhandled error", extra={"path": request.url.path})
                        response = _error(500, "internal_error", "unexpected server error")
                route = request.scope.get("route")
                template = getattr(route, "path", None) or ("static" if request.url.path.startswith("/static")
                                                            else "unmatched")
                sp.update_name(f"{request.method} {template}")
                sp.set_attribute("http.route", template)
                sp.set_attribute("http.status_code", response.status_code)
                elapsed = time.perf_counter() - start
                labels = {"method": request.method, "route": template, "status": str(response.status_code)}
                metrics.http_requests.add(1, labels)
                metrics.http_duration.record(elapsed, {"method": request.method, "route": template})
                if response.status_code == 403:
                    metrics.guardrail_events.add(1, {"kind": "acl_denied"})
                if response.status_code in (401, 403):
                    reason = "unauthorized" if response.status_code == 401 else "forbidden"
                    metrics.access_denied.add(1, {"reason": reason, "route": template})
                    target = template if template != "unmatched" else request.url.path  # rejected before routing
                    m().audit.record("access", caller=ctx.caller, decision="denied", resource_type="route",
                                     resource_ids=[f"{request.method} {target}"], reason=reason)
                trace_id = format(sp.get_span_context().trace_id, "032x")
                response.headers["X-Trace-Id"] = trace_id
                response.headers.setdefault("X-Content-Type-Options", "nosniff")
                response.headers.setdefault("Referrer-Policy", "no-referrer")
                response.headers.setdefault("X-Frame-Options", "DENY")
                if response.headers.get("content-type", "").startswith("text/html"):
                    response.headers.setdefault(
                        "Content-Security-Policy",
                        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                        "object-src 'none'; frame-ancestors 'none'")
                if response.status_code >= 500 or response.status_code in (401, 403):
                    log.warning("request failed", extra={"route": template, "status": response.status_code})
                return response
        finally:
            otel_context.detach(token)

    def _auth_check(request: Request) -> Response | None:
        env = m().config.api.api_key_env
        expected = os.environ.get(env) if env else None
        if not expected or not request.url.path.startswith("/api") or request.url.path == "/api/health":
            return None
        if request.url.path == "/api/ops/alerts" and request.method == "POST":
            return None  # Alertmanager authenticates with its own webhook token (see receive_alerts)
        supplied = _supplied_key(request) or ""
        if not supplied or not hmac.compare_digest(supplied, expected):
            return _error(401, "unauthorized", "missing or invalid API key")
        return None

    def _alert_token() -> str | None:
        token = os.environ.get(ALERT_TOKEN_ENV)
        path = m().config.observability.alert_token_file
        if not token and path and Path(path).exists():
            token = Path(path).read_text(encoding="utf-8").strip()
        return token or None

    # -- error handlers --------------------------------------------------------------------------------
    @app.exception_handler(MeshError)
    async def mesh_error(_request: Request, exc: MeshError):
        if exc.http_status >= 500:
            log.warning("service error", extra={"code": exc.code})
        return _error(exc.http_status, exc.code, exc.message, exc.details or None)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        fields = [str(e["loc"][1]) for e in errors if len(e["loc"]) > 1 and e["loc"][0] == "body"]
        first = errors[0] if errors else {}
        where = f"{first['loc'][-1]}: " if first.get("loc") else ""
        return _error(422, "invalid_argument", f"{where}{first.get('msg', 'request validation failed')}",
                      {"errors": errors[:20], "fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    # -- UI --------------------------------------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/health")
    def health():
        return m().health()

    # -- documents -------------------------------------------------------------------------------------
    @app.post("/api/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...), metadata: str | None = Form(None),
                              index: bool = Form(True)):
        limit = m().config.storage.max_upload_bytes
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise PayloadTooLarge(f"upload exceeds {limit} bytes", {"limit": limit})
        meta = None
        if metadata:
            try:
                meta = json.loads(metadata)
                if not isinstance(meta, dict):
                    raise ValueError
            except ValueError:
                return _error(422, "invalid_argument", "metadata must be a JSON object")
        result = await run_in_threadpool(m().ingest_document, None, data, file.filename or "upload", meta, index,
                                         file.content_type)
        if result["status"] == "failed":
            raise ExtractionFailed(result.get("reason") or "extraction failed",
                                   {"document_id": result["document_id"]})
        return result

    @app.get("/api/documents")
    def list_documents(offset: int = 0, limit: int = 50, status: str | None = None):
        return m().documents.list(offset, limit, status)

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str):
        return m().documents.get(document_id)

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str):
        return m().delete_document(document_id)

    @app.get("/api/documents/{document_id}/text")
    def document_text(document_id: str, page: int | None = None, section: str | None = None,
                      chunk_id: str | None = None, offset: int = 0, max_chars: int = 4000):
        return m().read_document(document_id, page, section, chunk_id, offset, max_chars)

    @app.get("/api/documents/{document_id}/original")
    def document_original(document_id: str):
        path, mime, filename = m().documents.original_file(document_id)
        headers = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"}
        if mime == "application/pdf":
            return FileResponse(path, media_type=mime, headers=headers,
                                content_disposition_type="inline", filename=filename)
        if mime in {"text/plain", "text/markdown"}:
            return FileResponse(path, media_type="text/plain; charset=utf-8", headers=headers,
                                content_disposition_type="inline", filename=filename)
        return FileResponse(path, media_type="application/octet-stream", headers=headers, filename=filename)

    @app.post("/api/documents/index", status_code=202)
    def index_documents(body: IndexRequest):
        return m().index_documents(body.document_ids, body.chunk_sizes, body.force)

    @app.post("/api/ingest-directory", status_code=202)
    def ingest_directory(body: DirectoryRequest):
        return m().ingest_directory(body.path)

    @app.get("/api/chunks/{chunk_id}")
    def get_chunk(chunk_id: str):
        return m().get_chunk(chunk_id)

    @app.post("/api/crawl", status_code=202)
    def crawl(body: CrawlRequest):
        return m().crawl_website(body.url, body.max_depth, body.max_pages)

    # -- jobs ------------------------------------------------------------------------------------------
    @app.get("/api/jobs")
    def list_jobs(limit: int = 50):
        return {"jobs": m().job_store.list(limit)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return m().get_job(job_id)

    @app.get("/api/jobs/{job_id}/items")
    def job_items(job_id: str, offset: int = 0, limit: int = 50, status: str | None = None):
        return m().job_items(job_id, offset, limit, status)

    # -- structured catalog ----------------------------------------------------------------------------
    @app.get("/api/structured-sources")
    def structured_sources():
        return {"sources": m().list_structured_sources()}

    @app.get("/api/structured-sources/{source_id}")
    def structured_source(source_id: str):
        return m().catalog.source_tree(source_id)

    @app.post("/api/structured-sources/{source_id}/catalog", status_code=202)
    def catalog_source(source_id: str, body: CatalogRequest | None = None):
        return m().catalog_source(source_id, body.datasets if body else None)

    @app.get("/api/catalog")
    def catalog_tree():
        return {"sources": m().catalog_tree()}

    # -- settings: structured sources --------------------------------------------------------------
    def require_admin(request: Request) -> None:
        env = m().config.api.api_key_env
        if env and os.environ.get(env):
            return  # the middleware already authenticated this request with the API key
        host = request.client.host if request.client else ""
        if host not in LOCAL_CLIENTS:
            raise Forbidden("Settings changes are only accepted from this machine unless an API key is configured "
                            f"(set ${env or 'KNOWLEDGE_MESH_API_KEY'})", {"client": host})

    @app.get("/api/settings")
    def settings_overview():
        return m().settings_overview()

    @app.post("/api/settings/sources", status_code=201)
    def add_source(body: SourcePayload, request: Request):
        require_admin(request)
        out = m().add_source(body.data(), request.headers.get("x-user") or (request.client.host if request.client else None))
        m().audit.record("settings.add_source", caller=request_context(request).caller, resource_type="source",
                         resource_ids=[body.id])
        return out

    @app.put("/api/settings/sources/{source_id}")
    def update_source(source_id: str, body: SourcePayload, request: Request):
        require_admin(request)
        out = m().update_source(source_id, body.data())
        m().audit.record("settings.update_source", caller=request_context(request).caller, resource_type="source",
                         resource_ids=[source_id])
        return out

    @app.delete("/api/settings/sources/{source_id}")
    def remove_source(source_id: str, request: Request):
        require_admin(request)
        out = m().remove_source(source_id)
        m().audit.record("settings.remove_source", caller=request_context(request).caller, resource_type="source",
                         resource_ids=[source_id])
        return out

    @app.post("/api/settings/sources/test")
    def test_new_source(body: SourcePayload, request: Request):
        require_admin(request)
        return m().test_source(payload=body.data())

    @app.post("/api/settings/sources/{source_id}/test")
    def test_saved_source(source_id: str, request: Request):
        require_admin(request)
        return m().test_source(source_id=source_id)

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        return m().get_dataset(dataset_id)

    @app.post("/api/datasets/discover-keys", status_code=202)
    def discover_keys(body: KeysRequest):
        return m().discover_keys(body.dataset_ids, body.max_rows)

    @app.post("/api/datasets/generate-terms", status_code=202)
    def generate_terms(body: TermsRequest):
        return m().generate_terms(body.dataset_ids, body.force)

    @app.get("/api/columns/{column_id}")
    def get_column(column_id: str):
        return m().catalog.get_column(column_id)

    @app.get("/api/profiling-runs/{run_id}")
    def profiling_run(run_id: str):
        return m().get_profiling_run(run_id)

    @app.get("/api/business-terms/{slug}")
    def business_term(slug: str):
        return m().get_business_term(slug)

    @app.get("/api/business-entities/{slug}")
    def business_entity(slug: str):
        return m().get_business_entity(slug)

    # -- definitions -----------------------------------------------------------------------------------
    @app.get("/api/business-definitions")
    def list_definitions():
        return {"definitions": m().definitions.list()}

    @app.post("/api/business-definitions", status_code=201)
    def capture_definition(body: DefinitionRequest, request: Request):
        author = body.author or request.headers.get("x-user") or "anonymous"
        return m().capture_definition(body.name, body.definition, body.entities, body.terms, author, body.source_ref)

    @app.get("/api/business-definitions/{definition_id}")
    def get_definition(definition_id: str):
        return m().definitions.get(definition_id)

    @app.delete("/api/business-definitions/{definition_id}")
    def delete_definition(definition_id: str):
        return m().definitions.delete(definition_id)

    # -- search / chat ---------------------------------------------------------------------------------
    @app.post("/api/search")
    def search(body: SearchRequest, request: Request):
        """Agent search: planner -> task executor loops -> cited answer (direct retrieval without an agent model)."""
        return m().search(body.query, body.top_k, body.evidence_types, body.document_ids, body.source_types,
                          body.structured_source_ids, body.dataset_ids, body.expand_parents, body.rerank,
                          context=request_context(request), output_format=body.output_format)

    @app.post("/api/searches/{trace_id}/clarification")
    def search_clarification(trace_id: str, body: ClarificationReply, request: Request):
        """Answer a search's clarification question: the search continues with the original request + the reply."""
        return m().continue_search(trace_id, body.reply, context=request_context(request))

    @app.get("/api/searches/{trace_id}")
    def get_search(trace_id: str):
        """One agent search: usage, rule checks, LLM-judge scores (once ready), feedback, Phoenix trace link."""
        return m().get_search(trace_id)

    @app.post("/api/searches/{trace_id}/feedback", status_code=201)
    def search_feedback(trace_id: str, body: FeedbackRequest, request: Request):
        return m().add_feedback(trace_id, body.rating, body.comment, request_context(request))

    @app.get("/api/ops/alerts")
    def list_alerts(status: str | None = None, limit: int = 50):
        return {"alerts": m().list_alerts(status, limit)}

    @app.post("/api/ops/alerts")
    async def receive_alerts(request: Request):
        """Alertmanager webhook. Requires the shared webhook token (or a local caller when none is set)."""
        token = _alert_token()
        host = request.client.host if request.client else ""
        if token:
            if not hmac.compare_digest(_supplied_key(request) or "", token):
                return _error(401, "unauthorized", "missing or invalid alert webhook token")
        elif host not in LOCAL_CLIENTS:
            return _error(403, "forbidden", "alert webhook accepts local callers only when no token is configured")
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "invalid_argument", "body must be JSON")
        return await run_in_threadpool(m().record_alerts, payload if isinstance(payload, dict) else {})

    @app.get("/api/audit")
    def audit_events(request: Request, limit: int = 100, caller: str | None = None):
        require_admin(request)
        return {"events": m().audit.recent(limit, caller)}

    @app.post("/api/retrieve")
    def retrieve(body: SearchRequest):
        """Direct mixed retrieval (documents, glossary, graph) without the agent."""
        return m().retrieve_context(body.query, body.top_k, body.evidence_types, body.document_ids,
                                    body.source_types, body.structured_source_ids, body.dataset_ids,
                                    body.expand_parents, body.rerank)

    @app.post("/api/chat")
    def chat(body: SearchRequest):
        return m().answer(body.query, top_k=body.top_k, evidence_types=body.evidence_types,
                          document_ids=body.document_ids, source_types=body.source_types,
                          structured_source_ids=body.structured_source_ids, dataset_ids=body.dataset_ids,
                          expand_parents=body.expand_parents, rerank=body.rerank)

    @app.get("/api/graph/stats")
    def graph_stats():
        return m().graph.stats()

    @app.get("/api/graph/view")
    def graph_view(source_id: str, schema: str | None = None, dataset_id: str | None = None):
        """Knowledge graph of a catalog selection (database = source, schema, or table) for the catalog view."""
        return m().graph_view(source_id, schema, dataset_id)

    @app.post("/api/graph/reload")
    def graph_reload(request: Request):
        """Re-read the persisted graph; the current graph stays in use if the file does not parse."""
        require_admin(request)
        out = m().graph.reload()
        m().audit.record("graph.reload", caller=request_context(request).caller, resource_type="graph")
        return out

    # -- reviews and proposed relationships ---------------------------------------------------------
    @app.post("/api/reviews")
    def review(body: ReviewRequest, request: Request):
        """Verify or reject an inferred FK, relationship, term relation or mapping (``inferred`` clears it)."""
        ctx = request_context(request)
        actor = request.headers.get("x-user") or ctx.caller
        out = m().review(body.object_type, body.object_id, body.status, actor, body.note)
        m().audit.record("review", caller=ctx.caller, resource_type=body.object_type, resource_ids=[body.object_id],
                         reason=body.status)
        return out

    @app.get("/api/datasets/{dataset_id}/relationships")
    def dataset_relationships(dataset_id: str):
        return m().dataset_relationships(dataset_id)

    @app.post("/api/datasets/generate-relationships", status_code=202)
    def generate_relationships(body: RelationshipsRequest):
        return m().generate_relationships(body.dataset_ids)

    return app

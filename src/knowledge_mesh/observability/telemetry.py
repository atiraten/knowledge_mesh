"""OpenTelemetry traces, metrics and structured logs.

* Traces go to an OTLP endpoint (an OpenTelemetry Collector forwarding to Arize Phoenix).
  Spans carry OpenInference attributes (``openinference.span.kind``, ``llm.*``,
  ``retrieval.documents.*``) so Phoenix can render RAG, LLM and tool steps.
* Metrics are exposed in Prometheus format at ``/metrics`` for Prometheus/Grafana.
* Logs are JSON with trace/span IDs; with ``loki_enabled`` they are also exported over OTLP
  so the collector can push them to Loki.

A tracer provider is always installed (even when export is disabled) so every API
response can carry a real trace ID.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .. import __version__

TRACER_NAME = "knowledge_mesh"
_lock = threading.Lock()
_configured = False
_capture_sensitive = False
_SENSITIVE_KEYS = ("token", "password", "secret", "api_key", "apikey", "authorization", "credential")


def setup_telemetry(settings) -> None:
    """Install tracer, meter and log providers once per process."""
    global _configured, _capture_sensitive
    with _lock:
        if _configured:
            return
        _configured = True
        _capture_sensitive = bool(settings.capture_sensitive_content)
        resource = Resource.create({"service.name": settings.service_name, "service.version": __version__,
                                    # Phoenix files traces under this project (instead of "default")
                                    "openinference.project.name": settings.service_name})
        wants_otlp = settings.enabled and (settings.phoenix_enabled or settings.loki_enabled)
        otlp_ok = wants_otlp and endpoint_reachable(settings.otlp_endpoint)

        tracer_provider = TracerProvider(resource=resource)
        if otlp_ok and settings.phoenix_enabled:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                tracer_provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True, timeout=5))
                )
            except Exception as exc:  # pragma: no cover - exporter optional at runtime
                logging.getLogger(__name__).warning("OTLP trace exporter unavailable: %s", exc)
        trace.set_tracer_provider(tracer_provider)

        readers = []
        if settings.enabled and settings.prometheus_enabled:
            try:
                from opentelemetry.exporter.prometheus import PrometheusMetricReader

                readers.append(PrometheusMetricReader())
            except Exception as exc:  # pragma: no cover
                logging.getLogger(__name__).warning("Prometheus reader unavailable: %s", exc)
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers, views=_histogram_views()))

        _configure_logging(settings, resource, otlp_ok)
        if wants_otlp and not otlp_ok:
            logging.getLogger(__name__).warning(
                "OTLP endpoint %s is not reachable; trace and log export are disabled for this run. Start the "
                "collector (docker compose -f deploy/docker-compose.yml up -d) and restart, or set "
                "observability.enabled: false.", settings.otlp_endpoint)
        get_metrics().register_process_gauges()


_SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300)
_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
_USD = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5)
_COUNTS = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100)
_SIZES = (1000, 2000, 5000, 10000, 20000, 40000, 60000, 100000, 200000)


def _histogram_views() -> list:
    """Bucket boundaries per histogram, so percentiles are meaningful (the OTel default 0..10000 is not)."""
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View

    plan = {
        "mesh_http_request_duration": _SECONDS, "mesh_stage_duration": _SECONDS, "mesh_job_duration": _SECONDS,
        "mesh_llm_latency": _SECONDS, "mesh_agent_search_duration": _SECONDS, "mesh_tool_duration": _SECONDS,
        "mesh_judge_score": _RATIOS, "mesh_eval_compliance": _RATIOS, "mesh_eval_citation_coverage": _RATIOS,
        "mesh_retrieval_top_score": _RATIOS, "mesh_search_cost_usd": _USD,
        "mesh_agent_iterations": _COUNTS, "mesh_agent_intents": _COUNTS, "mesh_search_evidence": _COUNTS,
        "mesh_retrieval_results": _COUNTS,
        "mesh_agent_context_chars": _SIZES, "mesh_tool_result_chars": _SIZES,
        "mesh_search_tokens": (500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000),
    }
    return [View(instrument_name=name, aggregation=ExplicitBucketHistogramAggregation(bounds))
            for name, bounds in plan.items()]


def endpoint_reachable(endpoint: str, timeout: float = 0.5) -> bool:
    """TCP probe of the OTLP endpoint, so a missing collector costs one warning instead of endless retries."""
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(endpoint if "://" in endpoint else f"http://{endpoint}")
    try:
        with socket.create_connection((parts.hostname or "localhost", parts.port or 4317), timeout=timeout):
            return True
    except OSError:
        return False


class _ExcludeTelemetryLoggers(logging.Filter):
    """Keep the exporters' own log records out of OTLP log export (otherwise a failing export
    logs an error, which is queued for export, fails, and logs again — forever)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(("opentelemetry", "grpc"))


class _ThrottleExporterWarnings(logging.Filter):
    """When the collector goes away after startup, the OTLP exporters warn on every batch and retry.
    Show one clear line per window instead, with the number of similar warnings suppressed since."""

    PREFIXES = ("opentelemetry.exporter", "opentelemetry.sdk.trace.export", "opentelemetry.sdk._logs",
                "opentelemetry.sdk._shared_internal", "grpc")

    def __init__(self, endpoint: str, window_seconds: float = 300.0, clock=time.monotonic):
        super().__init__()
        self.endpoint = endpoint
        self.window = window_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._until = 0.0
        self._suppressed = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING or not record.name.startswith(self.PREFIXES):
            return True
        now = self.clock()
        with self._lock:
            if now < self._until:
                self._suppressed += 1
                return False
            suppressed, self._suppressed = self._suppressed, 0
            self._until = now + self.window
        detail = record.getMessage().split(". Error details")[0][:160]
        msg = (f"Telemetry export to {self.endpoint} is failing ({detail}); traces and logs are dropped until it is "
               "reachable. Start the stack with scripts/observability.sh up.")
        if suppressed:
            msg += f" {suppressed} similar warnings were suppressed in the last {int(self.window // 60)} min."
        record.msg, record.args = msg, ()
        return True


def _configure_logging(settings, resource: Resource, otlp_ok: bool) -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if settings.log_format == "json" else _TextFormatter())
    handler.set_name("knowledge_mesh_stream")
    handler.addFilter(_ThrottleExporterWarnings(settings.otlp_endpoint))
    root.handlers = [h for h in root.handlers if h.get_name() != "knowledge_mesh_stream"] + [handler]
    if otlp_ok and settings.loki_enabled:
        try:
            from opentelemetry._logs import set_logger_provider
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

            provider = LoggerProvider(resource=resource)
            provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=settings.otlp_endpoint, insecure=True, timeout=5))
            )
            set_logger_provider(provider)
            otlp_handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
            otlp_handler.addFilter(_ExcludeTelemetryLoggers())
            root.addHandler(otlp_handler)
        except Exception as exc:  # pragma: no cover
            logging.getLogger(__name__).warning("OTLP log exporter unavailable: %s", exc)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers",
                  "opentelemetry", "grpc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_STD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            payload["trace_id"] = format(ctx.trace_id, "032x")
            payload["span_id"] = format(ctx.span_id, "016x")
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME, __version__)


def capture_sensitive() -> bool:
    return _capture_sensitive


@contextmanager
def span(name: str, kind: str = "CHAIN", attributes: dict[str, Any] | None = None,
         links: list[trace.SpanContext] | None = None) -> Iterator[trace.Span]:
    """Start a span tagged with an OpenInference span kind (AGENT, CHAIN, TOOL, LLM, RETRIEVER, EMBEDDING).
    ``links`` point at related spans that are not parents (e.g. the tasks a task depends on)."""
    span_links = [trace.Link(ctx) for ctx in links or [] if ctx is not None and ctx.is_valid]
    with tracer().start_as_current_span(name, links=span_links) as sp:
        sp.set_attribute("openinference.span.kind", kind)
        set_attributes(sp, attributes or {})
        yield sp


def add_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add an event to the current span (attribute values are flattened like ``set_attributes``)."""
    flat: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            flat[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value):
            flat[key] = list(value)
        else:
            flat[key] = json.dumps(value, default=str)[:4000]
    trace.get_current_span().add_event(name, flat)


def set_attributes(sp: trace.Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            sp.set_attribute(key, value)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, (str, int, float, bool)) for v in value):
            sp.set_attribute(key, list(value))
        else:
            sp.set_attribute(key, json.dumps(value, default=str)[:4000])


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def current_span_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.span_id, "016x") if ctx.is_valid else None


def current_span_context() -> trace.SpanContext | None:
    ctx = trace.get_current_span().get_span_context()
    return ctx if ctx.is_valid else None


def sanitize_params(params: dict[str, Any], max_str: int = 200) -> dict[str, Any]:
    """Redact secrets and bulky content from tool/API parameters before logging or tracing."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        lowered = key.lower()
        if any(s in lowered for s in _SENSITIVE_KEYS):
            out[key] = "***"
        elif lowered in {"content", "content_base64", "data", "text", "definition"} and not _capture_sensitive:
            out[key] = f"<{len(value) if hasattr(value, '__len__') else '?'} chars redacted>"
        elif isinstance(value, bytes):
            out[key] = f"<{len(value)} bytes>"
        elif isinstance(value, str):
            out[key] = value if len(value) <= max_str else value[:max_str] + "..."
        elif isinstance(value, dict):
            out[key] = sanitize_params(value, max_str)
        elif isinstance(value, (list, tuple)):
            out[key] = [sanitize_params(v, max_str) if isinstance(v, dict) else v for v in list(value)[:20]]
        else:
            out[key] = value
    return out


class Metrics:
    """Application metrics (exported to Prometheus by the OTel Prometheus reader)."""

    def __init__(self) -> None:
        m = metrics.get_meter(TRACER_NAME, __version__)
        self.meter = m
        self.http_requests = m.create_counter("mesh_http_requests", description="HTTP requests by route and status")
        self.http_duration = m.create_histogram("mesh_http_request_duration", unit="s", description="HTTP latency")
        self.documents = m.create_counter("mesh_documents_ingested", description="Documents by ingestion outcome")
        self.chunks_indexed = m.create_counter("mesh_chunks_indexed", description="Embedded leaf chunks")
        self.nodes_stored = m.create_counter("mesh_nodes_stored", description="Hierarchical nodes stored")
        self.stage_duration = m.create_histogram("mesh_stage_duration", unit="s", description="Pipeline stage latency")
        self.crawler_pages = m.create_counter("mesh_crawler_pages", description="Crawled pages by status")
        self.blocked_urls = m.create_counter("mesh_blocked_urls", description="URLs blocked by crawl safety rules")
        self.datasets = m.create_counter("mesh_datasets_cataloged", description="Datasets cataloged")
        self.keys = m.create_counter("mesh_keys_discovered", description="Keys recorded by type and origin")
        self.terms = m.create_counter("mesh_terms_generated", description="Business terms by outcome")
        self.graph_validations = m.create_counter("mesh_graph_validations", description="SHACL validations by outcome")
        self.graph_validation_failures = m.create_counter(
            "mesh_graph_validation_failures", description="Graph updates rejected by SHACL"
        )
        self.jobs = m.create_counter("mesh_jobs", description="Jobs by kind and terminal status")
        self.job_duration = m.create_histogram("mesh_job_duration", unit="s", description="Job duration")
        self.retrieval_results = m.create_histogram("mesh_retrieval_results", description="Results per index")
        self.llm_requests = m.create_counter("mesh_llm_requests", description="LLM calls by purpose and status")
        self.llm_tokens = m.create_counter("mesh_llm_tokens", description="LLM tokens by purpose and direction")
        self.llm_cost = m.create_counter("mesh_llm_estimated_cost_usd", description="Estimated LLM spend")
        self.llm_latency = m.create_histogram("mesh_llm_latency", unit="s", description="LLM call latency")
        self.guardrail_events = m.create_counter("mesh_guardrail_events", description="Guardrail signals by kind")
        self.retries = m.create_counter("mesh_retries", description="Retries by operation")
        self.rerank_outcomes = m.create_counter("mesh_rerank_outcomes", description="Reranker outcomes "
                                                "(selected / no_response / failed)")
        self.llm_errors = m.create_counter("mesh_llm_errors", description="Failed LLM calls by purpose and error type")
        # -- agent: tracing, planning, reliability ---------------------------------------------------
        self.agent_runs = m.create_counter("mesh_agent_runs", description="Agent searches by outcome")
        self.agent_search_duration = m.create_histogram("mesh_agent_search_duration", unit="s",
                                                        description="End-to-end agent search latency")
        self.agent_iterations = m.create_histogram("mesh_agent_iterations",
                                                   description="Executor LLM calls per task, by final task status")
        self.agent_tasks = m.create_counter("mesh_agent_tasks", description="Task runs by final status")
        self.agent_plans = m.create_counter("mesh_agent_plans", description="Plans by kind (initial/replan) and outcome")
        self.agent_intents = m.create_histogram("mesh_agent_intents", description="Intents per plan")
        self.agent_recoveries = m.create_counter(
            "mesh_agent_recoveries", description="Recovery events (invalid reply, tool error, replan, iteration cap, "
                                                 "plan fallback, retrieval fallback)")
        self.search_tokens = m.create_histogram("mesh_search_tokens", description="LLM tokens per agent search")
        self.search_cost = m.create_histogram("mesh_search_cost_usd", description="Estimated LLM cost per agent search")
        self.search_evidence = m.create_histogram("mesh_search_evidence", description="Evidence items per agent search")
        # -- context ---------------------------------------------------------------------------------
        self.agent_context_chars = m.create_histogram("mesh_agent_context_chars",
                                                      description="Characters sent per executor call")
        self.agent_context_pruned = m.create_counter(
            "mesh_agent_context_pruned", description="Context pruning by kind (stubbed, truncated, seen_before)")
        # -- tools -----------------------------------------------------------------------------------
        self.tool_calls = m.create_counter("mesh_tool_calls", description="Tool calls by source (mcp/agent), tool, "
                                                                          "status and error code")
        self.tool_duration = m.create_histogram("mesh_tool_duration", unit="s", description="Tool call latency")
        self.tool_result_chars = m.create_histogram("mesh_tool_result_chars", description="Tool result size")
        # -- RAG -------------------------------------------------------------------------------------
        self.retrieval_top_score = m.create_histogram("mesh_retrieval_top_score",
                                                      description="Best similarity per retrieval and index")
        self.retrieval_empty = m.create_counter("mesh_retrieval_empty", description="Retrievals with no result, by index")
        # -- quality and evaluation ------------------------------------------------------------------
        self.eval_checks = m.create_counter("mesh_eval_checks", description="Online rule checks by check and result")
        self.compliance = m.create_histogram("mesh_eval_compliance", description="Rule compliance per search (0-1)")
        self.citation_coverage = m.create_histogram("mesh_eval_citation_coverage",
                                                    description="Share of answer sentences with a valid citation")
        self.judge_runs = m.create_counter("mesh_judge_runs", description="LLM judge runs by status")
        self.judge_score = m.create_histogram("mesh_judge_score", description="LLM judge scores (0-1) by metric")
        self.hallucinations = m.create_counter("mesh_hallucinations", description="Answers with unsupported claims")
        self.feedback = m.create_counter("mesh_user_feedback", description="User feedback by rating")
        # -- security and operations -----------------------------------------------------------------
        self.policy_violations = m.create_counter("mesh_policy_violations", description="Policy violations by policy")
        self.sensitive_findings = m.create_counter("mesh_sensitive_findings",
                                                   description="PII/secret findings by kind and where")
        self.access_denied = m.create_counter("mesh_access_denied", description="Denied requests by reason")
        # knowledge graph retrieval and SQL record retrieval
        self.graph_queries = m.create_counter("mesh_graph_queries", description="query_knowledge_graph calls by status")
        self.graph_query_duration = m.create_histogram("mesh_graph_query_duration", unit="s",
                                                       description="query_knowledge_graph latency")
        self.graph_paths = m.create_histogram("mesh_graph_paths", description="Candidate paths per graph query")
        self.sql_queries = m.create_counter("mesh_sql_queries", description="execute_sql calls by status")
        self.sql_rows = m.create_histogram("mesh_sql_rows", description="Rows returned per execute_sql call")
        self.alerts_received = m.create_counter("mesh_alerts_received", description="Alerts received from Alertmanager")
        self._gauges_registered = False
        self._mesh_gauges_registered = False
        self.mesh_source = None  # callable returning the live KnowledgeMesh (for infrastructure gauges)

    def register_process_gauges(self) -> None:
        if self._gauges_registered:
            return
        self._gauges_registered = True
        try:
            import psutil
        except ImportError:  # pragma: no cover
            return
        proc = psutil.Process()
        proc.cpu_percent(None)

        def _cpu(_options):
            yield metrics.Observation(proc.cpu_percent(None))

        def _mem(_options):
            yield metrics.Observation(proc.memory_info().rss)

        self.meter.create_observable_gauge("mesh_process_cpu_percent", callbacks=[_cpu], description="Process CPU %")
        self.meter.create_observable_gauge(
            "mesh_process_memory_rss", callbacks=[_mem], unit="By", description="Process resident memory"
        )

    def register_mesh_gauges(self, mesh_source) -> None:
        """Infrastructure gauges read from the live mesh: DB size, index nodes, job queue depth, threads."""
        self.mesh_source = mesh_source
        if self._mesh_gauges_registered:
            return
        self._mesh_gauges_registered = True

        def _mesh():
            try:
                return self.mesh_source() if self.mesh_source else None
            except Exception:  # pragma: no cover
                return None

        def _db_size(_options):
            mesh = _mesh()
            path = getattr(getattr(getattr(mesh, "config", None), "storage", None), "metadata_db", None)
            if path is not None and Path(path).exists():
                yield metrics.Observation(Path(path).stat().st_size)

        def _index_nodes(_options):
            mesh = _mesh()
            for name in ("doc_index", "glossary_index"):
                idx = getattr(mesh, name, None)
                if idx is not None and idx._index is not None:  # only once loaded: never force a load here
                    yield metrics.Observation(len(idx._index.index_struct.nodes_dict), {"index": idx.name})

        def _jobs(_options):
            mesh = _mesh()
            if mesh is not None:
                try:
                    n = mesh.db.scalar("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')")
                    yield metrics.Observation(int(n or 0))
                except Exception:  # pragma: no cover
                    return

        def _threads(_options):
            yield metrics.Observation(threading.active_count())

        m = self.meter
        m.create_observable_gauge("mesh_db_size", callbacks=[_db_size], unit="By", description="Metadata DB file size")
        m.create_observable_gauge("mesh_index_nodes", callbacks=[_index_nodes], description="Embedded nodes per index")
        m.create_observable_gauge("mesh_jobs_pending", callbacks=[_jobs], description="Queued or running jobs")
        m.create_observable_gauge("mesh_threads", callbacks=[_threads], description="Process threads")


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics

import logging
import socket

from knowledge_mesh.observability.telemetry import (_ExcludeTelemetryLoggers, _ThrottleExporterWarnings,
                                                 endpoint_reachable)


def test_endpoint_probe():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert endpoint_reachable(f"http://127.0.0.1:{port}")
    finally:
        server.close()
    assert not endpoint_reachable(f"http://127.0.0.1:{port}", timeout=0.2)


def test_exporter_logs_are_not_fed_back_into_otlp_export():
    f = _ExcludeTelemetryLoggers()
    rec = lambda name: logging.LogRecord(name, logging.ERROR, __file__, 1, "x", None, None)  # noqa: E731
    assert not f.filter(rec("opentelemetry.exporter.otlp.proto.grpc.exporter"))
    assert not f.filter(rec("grpc._channel"))
    assert f.filter(rec("knowledge_mesh.jobs"))


def test_exporter_warnings_are_throttled_to_one_clear_line_per_window():
    now = [1000.0]
    f = _ThrottleExporterWarnings("http://localhost:4317", window_seconds=300, clock=lambda: now[0])
    msg = ("Transient error StatusCode.UNAVAILABLE encountered while exporting traces to localhost:4317, retrying in "
           "0.82s. Error details: failed to connect to all addresses")
    rec = lambda name="opentelemetry.exporter.otlp.proto.grpc.exporter": logging.LogRecord(  # noqa: E731
        name, logging.WARNING, __file__, 1, msg, None, None)
    first = rec()
    assert f.filter(first)
    text = first.getMessage()
    assert text.startswith("Telemetry export to http://localhost:4317 is failing")
    assert "scripts/observability.sh up" in text and "Error details" not in text
    assert [f.filter(rec()) for _ in range(3)] == [False, False, False]  # same window: suppressed
    assert f.filter(logging.LogRecord("knowledge_mesh.jobs", logging.WARNING, __file__, 1, "job", None, None))
    now[0] += 301
    later = rec()
    assert f.filter(later) and "3 similar warnings were suppressed in the last 5 min" in later.getMessage()


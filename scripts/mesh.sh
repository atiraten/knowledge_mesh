#!/usr/bin/env bash
# Start and stop Knowledge Mesh: the app server and (unless --no-observability) the observability stack.
#
# Usage:
#   scripts/mesh.sh start   [--no-observability]   stack on Podman (scripts/observability.sh up), then the app
#   scripts/mesh.sh stop    [--no-observability]   the app, then the stack (the Podman VM keeps running)
#   scripts/mesh.sh restart [--no-observability]
#   scripts/mesh.sh status
#
# Environment: PORT (default 9801), HOST (default 0.0.0.0, so the containers can scrape /metrics).
# The app writes data/run/server.pid and logs to data/logs/server.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-9801}"
HOST="${HOST:-0.0.0.0}"
PY="$ROOT/.venv/bin/python"
RUN_DIR="$ROOT/data/run"
LOG_DIR="$ROOT/data/logs"
PID_FILE="$RUN_DIR/server-$PORT.pid"
LOG_FILE="$LOG_DIR/server.log"
OBS="$ROOT/scripts/observability.sh"

die() { echo "error: $*" >&2; exit 1; }

app_pid() { [ -f "$PID_FILE" ] && cat "$PID_FILE" || true; }
app_running() { local pid; pid="$(app_pid)"; [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; }
port_owner() { lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1 || true; }

start_app() {
  [ -x "$PY" ] || die "no virtualenv at $ROOT/.venv (python3 -m venv .venv && .venv/bin/pip install -e '.[google,dev]')"
  if app_running; then
    echo "Knowledge Mesh is already running (pid $(app_pid)) on port $PORT."
    return 0
  fi
  local owner
  owner="$(port_owner)"
  [ -z "$owner" ] || die "port $PORT is used by pid $owner ($(ps -o comm= -p "$owner")). Stop it or set PORT."
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  echo "Starting Knowledge Mesh on $HOST:$PORT (log: $LOG_FILE)..."
  # Detached launch: its own session, no inherited descriptors, stdin from /dev/null, output to the log.
  # The app survives the shell or terminal that started it, and this script returns at once.
  "$PY" - "$ROOT" "$LOG_FILE" "$PY" "$ROOT/src/harness.py" serve --host "$HOST" --port "$PORT" > "$PID_FILE" <<'EOF'
import subprocess, sys
root, log, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
with open(log, "ab") as out:
    proc = subprocess.Popen(cmd, cwd=root, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                            start_new_session=True, close_fds=True)
print(proc.pid)
EOF
  local deadline=$((SECONDS + 120))
  until [ "$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/api/health" || true)" = "200" ]; do
    app_running || [ -n "$(port_owner)" ] || { rm -f "$PID_FILE"; tail -n 30 "$LOG_FILE" >&2; die "the app exited during startup"; }
    [ "$SECONDS" -lt "$deadline" ] || die "the app did not answer on port $PORT within 120 s (see $LOG_FILE)"
    sleep 2
  done
  # The python.org Python launcher runs the interpreter as a child process: record the process that listens.
  port_owner > "$PID_FILE"
  echo "Knowledge Mesh is up: http://localhost:$PORT  (pid $(app_pid))"
}

stop_app() {
  if ! app_running; then
    rm -f "$PID_FILE"
    echo "Knowledge Mesh is not running."
    return 0
  fi
  local pid
  pid="$(app_pid)"
  echo "Stopping Knowledge Mesh (pid $pid)..."
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "Still running after 20 s; sending SIGKILL."
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

status_app() {
  if app_running; then
    local code
    code="$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/api/health" || true)"
    echo "Knowledge Mesh: running (pid $(app_pid), port $PORT, health HTTP $code)"
  else
    echo "Knowledge Mesh: stopped$( [ -n "$(port_owner)" ] && echo " (port $PORT is used by pid $(port_owner), not started by this script)")"
  fi
}

WITH_OBS=1
ACTION="${1:-}"
shift || true
for arg in "$@"; do
  case "$arg" in
    --no-observability) WITH_OBS=0 ;;
    *) die "unknown option: $arg" ;;
  esac
done

case "$ACTION" in
  start)
    [ "$WITH_OBS" -eq 1 ] && "$OBS" up
    start_app ;;
  stop)
    stop_app
    [ "$WITH_OBS" -eq 1 ] && "$OBS" down
    true ;;
  restart)
    stop_app
    start_app ;;
  status)
    status_app
    "$OBS" status ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac

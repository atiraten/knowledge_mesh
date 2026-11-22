#!/usr/bin/env bash
# Start and stop the local observability stack (deploy/docker-compose.yml) on Podman:
#   OTel Collector, Phoenix, Loki, Prometheus, Alertmanager, Grafana
#
# Usage:
#   scripts/observability.sh up                     start the Podman VM (if needed) and the stack, wait until healthy
#   scripts/observability.sh down [--stop-engine]   stop the stack (and the Podman VM with --stop-engine)
#   scripts/observability.sh status                 VM, containers and endpoint health
#   scripts/observability.sh logs [service]         last container logs (all services, or one)
#
# Podman is the default engine; a running Docker engine is used only when Podman is not installed.
# Environment: MESH_HOST_IP overrides the address containers use to reach the app on this Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT/deploy/docker-compose.yml"
SECRETS_DIR="$ROOT/deploy/secrets"
TOKEN_FILE="$SECRETS_DIR/alert_webhook_token"
export PATH="/opt/podman/bin:/usr/local/bin:$PATH"   # the Podman installer does not add itself to PATH
export PODMAN_COMPOSE_WARNING_LOGS=false            # silence "Executing external compose provider"

# name|url (any HTTP answer below 500 counts as up)
ENDPOINTS=(
  "Phoenix|http://127.0.0.1:6006/"
  "Grafana|http://127.0.0.1:3000/api/health"
  "Prometheus|http://127.0.0.1:9090/-/ready"
  "Alertmanager|http://127.0.0.1:9093/-/ready"
  "Loki|http://127.0.0.1:3100/ready"
  "OTel Collector (OTLP HTTP)|http://127.0.0.1:4318/"
)

die() { echo "error: $*" >&2; exit 1; }

engine() {
  if command -v podman >/dev/null 2>&1; then echo podman
  elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then echo docker
  else echo none
  fi
}

compose() {
  case "$(engine)" in
    podman)
      # Private client config: ~/.docker/config.json may name a credential helper from an old Docker Desktop
      # install ("credsStore": "desktop") that no longer exists and breaks every pull. All images are public.
      local cfg="$ROOT/data/run/docker-config"
      mkdir -p "$cfg"
      [ -s "$cfg/config.json" ] || echo '{}' > "$cfg/config.json"
      DOCKER_CONFIG="$cfg" podman compose -f "$COMPOSE_FILE" "$@" ;;
    docker) docker compose -f "$COMPOSE_FILE" "$@" ;;
    *) die "no container engine. Install Podman (https://podman.io, macOS arm64 installer), then:
  podman machine init --cpus 4 --memory 4096 && podman machine start" ;;
  esac
}

engine_ready() {
  case "$(engine)" in
    podman) podman info >/dev/null 2>&1 ;;
    docker) docker info >/dev/null 2>&1 ;;
    *) return 1 ;;
  esac
}

start_engine() {
  [ "$(engine)" = "none" ] && compose version   # prints the install instructions and exits
  engine_ready && return 0
  if [ "$(engine)" = "podman" ]; then
    if [ -z "$(podman machine list --format '{{.Name}}' 2>/dev/null)" ]; then
      echo "Creating the Podman VM (4 CPUs, 4 GiB)..."
      podman machine init --cpus 4 --memory 4096
    fi
    echo "Starting the Podman VM..."
    podman machine start
  fi
  local deadline=$((SECONDS + 120))
  until engine_ready; do
    [ "$SECONDS" -lt "$deadline" ] || die "the container engine did not become ready within 120 s"
    sleep 2
  done
}

ensure_token() {
  mkdir -p "$SECRETS_DIR"
  chmod 700 "$SECRETS_DIR"
  if [ ! -s "$TOKEN_FILE" ]; then
    (umask 077 && python3 -c 'import secrets; print(secrets.token_hex(32))' > "$TOKEN_FILE")
    echo "Created the Alertmanager webhook token: $TOKEN_FILE"
  fi
  chmod 644 "$TOKEN_FILE"  # mounted as a single file; Alertmanager runs as another user in its container
}

http_code() { curl -s -m 3 -o /dev/null -w "%{http_code}" "$1" || true; }

health() {
  for entry in "${ENDPOINTS[@]}"; do
    local name="${entry%%|*}" url="${entry#*|}" code
    code="$(http_code "$url")"
    if [ "$code" = "000" ] || [ "$code" -ge 500 ]; then
      printf "  %-28s DOWN  (%s)\n" "$name" "$url"
    else
      printf "  %-28s up    (%s)\n" "$name" "$url"
    fi
  done
}

wait_healthy() {
  local deadline=$((SECONDS + 240)) pending code
  echo -n "Waiting for the services"
  while :; do
    pending=0
    for entry in "${ENDPOINTS[@]}"; do
      code="$(http_code "${entry#*|}")"
      if [ "$code" = "000" ] || [ "$code" -ge 500 ]; then pending=$((pending + 1)); fi
    done
    [ "$pending" -eq 0 ] && { echo " ready."; return 0; }
    [ "$SECONDS" -ge "$deadline" ] && { echo; echo "Some services are not ready yet:"; health; return 1; }
    echo -n "."
    sleep 3
  done
}

cmd_up() {
  start_engine
  ensure_token
  [ -n "${MESH_HOST_IP:-}" ] && echo "Containers reach the app at host.docker.internal -> $MESH_HOST_IP"
  compose up -d --remove-orphans
  wait_healthy || true
  cat <<EOF

Observability stack ($(engine)):
  Phoenix (traces, evaluations)   http://localhost:6006
  Grafana (dashboards, logs)      http://localhost:3000   user admin, password \$GRAFANA_ADMIN_PASSWORD (default change-me)
  Prometheus (metrics, rules)     http://localhost:9090
  Alertmanager (alert routing)    http://localhost:9093
  Loki (logs, via Grafana)        http://localhost:3100
The app must listen on 0.0.0.0:8000 for Prometheus and Alertmanager to reach it (scripts/mesh.sh does this).
EOF
}

cmd_down() {
  if engine_ready; then
    compose down
  else
    echo "The container engine is not running; no containers to stop."
  fi
  if [ "${1:-}" = "--stop-engine" ] && [ "$(engine)" = "podman" ]; then
    echo "Stopping the Podman VM..."
    podman machine stop
  fi
}

cmd_status() {
  case "$(engine)" in
    podman) echo "Engine: Podman $(podman --version | awk '{print $3}')"; podman machine list ;;
    docker) echo "Engine: Docker" ;;
    *) echo "Engine: none installed (install Podman: https://podman.io)" ;;
  esac
  if engine_ready; then
    compose ps
  else
    echo "The container engine is not running."
  fi
  echo "Endpoints:"
  health
}

cmd_logs() {
  engine_ready || die "the container engine is not running"
  compose logs --tail=200 "$@"
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  down) shift; cmd_down "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs) shift; cmd_logs "$@" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac

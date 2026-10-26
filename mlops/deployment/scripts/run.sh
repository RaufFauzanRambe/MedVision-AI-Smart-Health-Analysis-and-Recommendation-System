#!/usr/bin/env bash
# =============================================================================
# MedVision-AI — local container run script
#
# Usage:
#   bash scripts/run.sh                        # run with sane defaults
#   bash scripts/run.sh v1.2.3                 # run a specific tag
#   APP_PORT=9000 bash scripts/run.sh          # override host port
# =============================================================================
set -euo pipefail

# --- Configuration -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-medvision-ai}"
TAG="${1:-${TAG:-latest}}"
CONTAINER_NAME="${CONTAINER_NAME:-medvision-api-run}"

HOST_PORT="${APP_PORT:-8000}"
CONTAINER_PORT="8000"

DATA_DIR="${DATA_DIR:-${HOME}/.medvision/data}"
LOG_DIR="${LOG_DIR:-${HOME}/.medvision/logs}"

# --- Pre-flight checks ------------------------------------------------------
command -v docker >/dev/null 2>&1 || {
    echo "ERROR: 'docker' not found in PATH." >&2
    exit 127
}

mkdir -p "${DATA_DIR}" "${LOG_DIR}"

echo "==> Running ${IMAGE_NAME}:${TAG}"
echo "    Container : ${CONTAINER_NAME}"
echo "    Port map : ${HOST_PORT}->${CONTAINER_PORT}"
echo "    Data vol : ${DATA_DIR}"
echo "    Log vol  : ${LOG_DIR}"
echo "    Ctrl-C to stop."

# --- Run ---------------------------------------------------------------------
# --rm     : auto-remove on exit
# -i -t    : interactive TTY for graceful Ctrl-C
# -p       : publish host:container port
# -e ...   : pass through minimal runtime env
# -v ...   : bind-mount data & logs for persistence
docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    -e APP_HOST=0.0.0.0 \
    -e APP_PORT="${CONTAINER_PORT}" \
    -e LOG_LEVEL="${LOG_LEVEL:-INFO}" \
    -e MODEL_NAME="${MODEL_NAME:-medvision-v1}" \
    -e MODEL_VERSION="${TAG}" \
    -v "${DATA_DIR}:/opt/medvision/data" \
    -v "${LOG_DIR}:/opt/medvision/logs" \
    "${IMAGE_NAME}:${TAG}"

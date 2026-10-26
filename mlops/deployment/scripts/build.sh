#!/usr/bin/env bash
# =============================================================================
# MedVision-AI — image build script
#
# Usage:
#   bash scripts/build.sh                    # default tag = latest
#   bash scripts/build.sh v1.2.3             # custom version tag
#   IMAGE_REGISTRY=ghcr.io/org bash scripts/build.sh v1.2.3
# =============================================================================
set -euo pipefail

# --- Configuration -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DEPLOY_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-medvision-ai}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-}"
TAG="${1:-${TAG:-latest}}"

if [[ -n "${IMAGE_REGISTRY}" ]]; then
    FULL_IMAGE="${IMAGE_REGISTRY}/${IMAGE_NAME}:${TAG}"
else
    FULL_IMAGE="${IMAGE_NAME}:${TAG}"
fi

# --- Pre-flight checks ------------------------------------------------------
command -v docker >/dev/null 2>&1 || {
    echo "ERROR: 'docker' not found in PATH." >&2
    exit 127
}

echo "==> Building image: ${FULL_IMAGE}"
echo "    Context: ${PROJECT_ROOT}"
echo "    Dockerfile: ${DEPLOY_DIR}/Dockerfile"

# --- Build -------------------------------------------------------------------
docker build \
    -f "${DEPLOY_DIR}/Dockerfile" \
    -t "${FULL_IMAGE}" \
    --build-arg PYTHON_VERSION=3.11-slim \
    --label "org.opencontainers.image.title=MedVision-AI" \
    --label "org.opencontainers.image.version=${TAG}" \
    --label "org.opencontainers.image.source=$(git -C "${PROJECT_ROOT}" remote get-url origin 2>/dev/null || echo local)" \
    --label "org.opencontainers.image.revision=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    "${PROJECT_ROOT}"

echo "==> Build complete: ${FULL_IMAGE}"

# Optional push
if [[ "${PUSH_IMAGE:-0}" == "1" ]]; then
    echo "==> Pushing ${FULL_IMAGE} ..."
    docker push "${FULL_IMAGE}"
    echo "==> Push complete."
fi

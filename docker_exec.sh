#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="${IMAGE_NAME:-people-counter:gpu-final-nvdec}"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "❌ Docker image $IMAGE_NAME not found."
    echo "   Build/prepare it first (e.g. ./1_prepare.sh then ./2_prepare_nvdec.sh)."
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 python -m pytest -q app_v2/tests/test_frame_telemetry.py"
    exit 1
fi

PYTEST_ADDOPTS_DEFAULT="-s -rs -W ignore:.*pynvml.*deprecated.*:FutureWarning"
PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-$PYTEST_ADDOPTS_DEFAULT}"
NVDEC_DEBUG_LOGS="${NVDEC_DEBUG_LOGS:-1}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,video}"

DOCKER_ARGS=(
    --rm -it
    --gpus all
    -e "DISPLAY=${DISPLAY:-}"
    -e "PYTHONPATH=/app"
    -e "PYTEST_ADDOPTS=$PYTEST_ADDOPTS"
    -e "NVDEC_DEBUG_LOGS=$NVDEC_DEBUG_LOGS"
    -e "NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES"
    -v "$PWD:/app"
    -w /app
)

if [[ -n "${NVDEC_TEST_STREAM_URL:-}" ]]; then
    DOCKER_ARGS+=( -e "NVDEC_TEST_STREAM_URL=$NVDEC_TEST_STREAM_URL" )
fi

echo "🐳 Running in $IMAGE_NAME: $*"
docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" "$@"

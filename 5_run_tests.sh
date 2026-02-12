#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="${IMAGE_NAME:-people-counter:gpu-final-nvdec}"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Step 2 (./2_prepare_nvdec.sh) is mandatory after ./1_prepare.sh—rebuild both before retrying or set IMAGE_NAME to an existing NVDEC-ready tag."
    exit 1
fi

echo "🧪 Running Python build + pytest inside $IMAGE_NAME..."

CONFIG_PATH="app_v2/config/test_config.yaml"
if [[ -z "${NVDEC_TEST_STREAM_URL:-}" && -f "$CONFIG_PATH" ]]; then
    NVDEC_TEST_STREAM_URL="$(python3 - <<'PY'
from pathlib import Path

path = Path("app_v2/config/test_config.yaml")
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = line.split(":", 1)
        if len(candidate) == 2 and candidate[0].strip() == "nvdec_test_stream_url":
            value = candidate[1].strip().strip('"').strip("'")
            if value:
                print(value)
                break
PY
    )"
fi

docker_args=(
    --rm -it
    --gpus all
    -e "DISPLAY=$DISPLAY"
    -e "PYTHONPATH=/app"
    -v "$PWD:/app"
    -w "/app"
)
PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-"-s -rs"}"
NVDEC_DEBUG_LOGS="${NVDEC_DEBUG_LOGS:-1}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,video}"
docker_args+=(-e "PYTEST_ADDOPTS=$PYTEST_ADDOPTS")
docker_args+=(-e "NVDEC_DEBUG_LOGS=$NVDEC_DEBUG_LOGS")
docker_args+=(-e "NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES")
if [[ -n "${NVDEC_TEST_STREAM_URL:-}" ]]; then
    export NVDEC_TEST_STREAM_URL
    docker_args+=("-e" "NVDEC_TEST_STREAM_URL=$NVDEC_TEST_STREAM_URL")
fi

docker run "${docker_args[@]}" \
    "$IMAGE_NAME" bash -c "set -e; if command -v nvidia-smi >/dev/null 2>&1; then echo '[containérisé] nvidia-smi'; nvidia-smi || true; else echo '[containérisé] nvidia-smi absent'; fi; if command -v nvcc >/dev/null 2>&1; then echo '[containérisé] nvcc --version'; nvcc --version || true; else echo '[containérisé] nvcc absent'; fi; echo '[containérisé] NVDEC_DEBUG_LOGS=$NVDEC_DEBUG_LOGS'; echo '[containérisé] NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES'; python -m compileall app_v2; python -m pytest $PYTEST_ADDOPTS app_v2/tests"

echo "✅ Tests completed inside $IMAGE_NAME."

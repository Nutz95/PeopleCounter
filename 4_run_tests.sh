#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="people-counter:gpu-final"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Run ./1_prepare.sh to build and prep the image."
    exit 1
fi

echo "🧪 Running Python build + pytest inside $IMAGE_NAME..."

docker run --rm -it \
    --gpus all \
    -e "DISPLAY=$DISPLAY" \
    -e "PYTHONPATH=/app" \
    -v "$PWD:/app" \
    -w "/app" \
    "$IMAGE_NAME" bash -c "set -e; if command -v nvidia-smi >/dev/null 2>&1; then echo '[containérisé] nvidia-smi'; nvidia-smi || true; else echo '[containérisé] nvidia-smi absent'; fi; if command -v nvcc >/dev/null 2>&1; then echo '[containérisé] nvcc --version'; nvcc --version || true; else echo '[containérisé] nvcc absent'; fi; python -m compileall app_v2; python -m pytest app_v2/tests"

echo "✅ Tests completed inside $IMAGE_NAME."

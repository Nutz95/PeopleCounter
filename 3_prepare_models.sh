#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="${IMAGE_NAME:-people-counter:gpu-final}"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Run ./0_build_image.sh followed by ./1_prepare.sh before preparing models."
    exit 1
fi

echo "🧪 Launching container to prepare models..."
docker run --rm -it --gpus all -e DISPLAY=$DISPLAY -v "$PWD:/app" -w /app "$IMAGE_NAME" bash -c "set -e; python3 prepare_models.py"

echo "✅ Models prepared inside $IMAGE_NAME."

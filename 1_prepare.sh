#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="people-counter:gpu-final"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Run ./0_build_image.sh to build it once before prepping."
    exit 1
fi

action() {
    echo "🧪 Running prep commands inside the container..."
    local docker_args=("--gpus" "all" "-e" "DISPLAY=$DISPLAY" "-v" "$PWD:/app" "-w" "/app")
    local container_id
    container_id=$(docker create "${docker_args[@]}" "$IMAGE_NAME" bash -c "set -e; apt-get update -qq; apt-get install -y --no-install-recommends git unzip ninja-build build-essential pkg-config cmake || true; if ! command -v nvcc >/dev/null 2>&1; then echo '⚠️ nvcc introuvable : votre image doit fournir un CUDA 13.x compatible'; fi; pip install --no-cache-dir 'cuda-python>=13.1.0,<14.0'; pip install --no-cache-dir flask screeninfo psutil matplotlib ninja pytest; exit 0")

    docker start -a "$container_id"
    echo "💾 Committing prep container to $IMAGE_NAME..."
    docker commit "$container_id" "$IMAGE_NAME" >/dev/null
    docker rm "$container_id" >/dev/null
}

action

echo "✅ Prepared image $IMAGE_NAME is ready. Run ./2_prepare_models.sh if you need to rebuild models, then ./3_run_app.sh --profile ..."

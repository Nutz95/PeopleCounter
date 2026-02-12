#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="${IMAGE_NAME:-people-counter:gpu-final}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-people-counter:gpu-final-nvdec}"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Build/prep first with ./0_build_image.sh then ./1_prepare.sh"
    exit 1
fi

echo "🧪 Building NVDEC/VPF layer from image: $IMAGE_NAME"

docker_args=("--gpus" "all" "-e" "DISPLAY=$DISPLAY" "-v" "$PWD:/app" "-w" "/app")
container_id=$(docker create "${docker_args[@]}" "$IMAGE_NAME" bash -c "set -e; chmod +x /app/scripts/nvdec/install_vpf.sh; /app/scripts/nvdec/install_vpf.sh")

cleanup() {
    if [[ -n "${container_id:-}" ]]; then
        docker rm -f "$container_id" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

docker start -a "$container_id"

echo "💾 Committing NVDEC layer to $OUTPUT_IMAGE..."
docker commit "$container_id" "$OUTPUT_IMAGE" >/dev/null
docker rm "$container_id" >/dev/null
container_id=""

echo "✅ NVDEC-enabled image ready: $OUTPUT_IMAGE"
echo "   You can run tests with: IMAGE_NAME=$OUTPUT_IMAGE ./5_run_tests.sh"
echo "   To run the app with this image, set IMAGE_NAME in your launch script or retag manually."

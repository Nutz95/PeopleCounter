#!/usr/bin/env bash
set -euo pipefail

# run_app.sh - Run the PeopleCounter application in Docker
# Usage: ./run_app.sh [/dev/video0 | http://...]

SOURCE="${1:-/dev/video0}"
IMAGE_NAME="people-counter:gpu-final"

echo "🚀 Launching PeopleCounter Container..."
echo "📸 Source: $SOURCE"

# Detection of source type
DOCKER_ARGS=("--gpus" "all" "-p" "5000:5000" "-e" "DISPLAY=$DISPLAY")

# --- MODE DEVELOPPEMENT ---
# On monte le dossier actuel dans /app pour que les modifs soient instantanées
# On définit PYTHONPATH pour utiliser /app
DOCKER_ARGS+=("-v" "$PWD:/app" "-w" "/app" "-e" "PYTHONPATH=/app")

if [[ "$SOURCE" == http* ]] || [[ "$SOURCE" == rtsp* ]]; then
    echo "🌐 Using Network Stream mode"
    DOCKER_ARGS+=("-e" "CAPTURE_MODE=rtsp" "-e" "RTSP_URL=$SOURCE")
else
    echo "🔌 Using Local Device mode"
    DOCKER_ARGS+=("--device" "$SOURCE:$SOURCE" "-e" "CAPTURE_MODE=usb" "-e" "CAMERA_INDEX=$SOURCE")
fi

echo "[DEBUG] Installation/Verif des dependances rapides..."
# Run prepare_models.py inside the container to clone/install local lwcc if LWCC_GIT_URL is set
# and pre-download optional YOLO models (controlled by YOLO_PREPARE env var).
# We avoid installing 'lwcc' from PyPI here to prevent overriding local editable installs.
docker run --rm -it \
    "${DOCKER_ARGS[@]}" \
    "$IMAGE_NAME" bash -c "set -e; apt-get update -qq && apt-get install -y git unzip || true; python3 prepare_models.py || true; pip install --no-cache-dir flask screeninfo psutil matplotlib; python3 main.py"

echo "⏹️ Container stopped."

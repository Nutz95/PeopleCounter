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

if [[ "$SOURCE" == http* ]] || [[ "$SOURCE" == rtsp* ]]; then
    echo "🌐 Using Network Stream mode"
    DOCKER_ARGS+=("-e" "CAPTURE_MODE=rtsp" "-e" "RTSP_URL=$SOURCE")
else
    echo "🔌 Using Local Device mode"
    DOCKER_ARGS+=("--device" "$SOURCE:$SOURCE" "-e" "CAPTURE_MODE=usb" "-e" "CAMERA_INDEX=$SOURCE")
fi

docker run --rm -it \
    "${DOCKER_ARGS[@]}" \
    "$IMAGE_NAME" python3 main.py

echo "⏹️ Container stopped."

#!/usr/bin/env bash
set -euo pipefail

# run_app.sh - Run the PeopleCounter application in Docker
# Usage: ./run_app.sh [camera_device] (default: /dev/video0)

VIDEO_DEVICE="${1:-/dev/video0}"
IMAGE_NAME="people-counter:gpu-final"

echo "🚀 Launching PeopleCounter Container..."
echo "📸 Camera device: $VIDEO_DEVICE"

# Note: We expose port 5000 for the upcoming web server
docker run --rm -it \
    --gpus all \
    --device "$VIDEO_DEVICE":"$VIDEO_DEVICE" \
    -p 5000:5000 \
    -e DISPLAY=$DISPLAY \
    -e CAPTURE_MODE=usb \
    "$IMAGE_NAME" python3 main.py

echo "⏹️ Container stopped."

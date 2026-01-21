#!/usr/bin/env bash
set -euo pipefail

# Run pipeline in USB camera mode (1080p)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

export CAPTURE_MODE=usb
export CAMERA_INDEX=${CAMERA_INDEX:-0}
export CAMERA_WIDTH=${CAMERA_WIDTH:-1920}
export CAMERA_HEIGHT=${CAMERA_HEIGHT:-1080}
export OPENVINO_DEVICE=${OPENVINO_DEVICE:-GPU}
export LWCC_BACKEND=${LWCC_BACKEND:-openvino}
export YOLO_BACKEND=${YOLO_BACKEND:-openvino_native}
export YOLO_DEVICE=${YOLO_DEVICE:-GPU.0}

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment .venv..."
  python -m venv .venv
fi

# Activate venv
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  PYTHON=".venv/bin/python"
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
  PYTHON=".venv/Scripts/python"
else
  echo "Virtual environment not found in .venv"
  exit 1
fi

# Upgrade pip if we just created it or if INSTALL_REQS is set
if [ "${INSTALL_REQS:-0}" = "1" ]; then
  $PYTHON -m pip install --upgrade pip
  $PYTHON -m pip install -r requirements.txt
fi

# Run
$PYTHON camera_app_pipeline.py

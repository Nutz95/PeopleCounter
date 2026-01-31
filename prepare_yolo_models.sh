#!/usr/bin/env bash
# Helper to download YOLO11/12 and segmentation models into models/ (uses ultralytics)
ROOT=$(cd "$(dirname "$0")" && pwd)
PY="$ROOT/.venv/Scripts/python.exe"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

if [ ! -f "$PY" ]; then
  echo "Virtualenv not found. Run ./setup.sh first to create .venv and install requirements."
  exit 1
fi

# List of YOLO v26 models to ensure downloaded as .pt
models=(
  "yolo26n.pt"
  "yolo26s.pt"
  "yolo26m.pt"
  "yolo26l.pt"
  "yolo26x.pt"
  "yolo26n-seg.pt"
  "yolo26s-seg.pt"
  "yolo26m-seg.pt"
  "yolo26l-seg.pt"
  "yolo26x-seg.pt"
)

for m in "${models[@]}"; do
  echo "Ensuring $m is available (download via ultralytics if needed)"
  "$PY" -c "from ultralytics import YOLO; YOLO('$m')"
  # Move any .pt downloaded into models dir
  if [ -f "$m" ]; then
    mv -n "$m" "$MODELS_DIR/"
  fi
done

echo "Downloaded models are in $MODELS_DIR"

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

# List of models to ensure downloaded as .pt
models=(
  "yolo11n.pt"
  "yolo11s.pt"
  "yolo11m.pt"
  "yolo11l.pt"
  "yolo11x.pt"
  "yolo12n.pt"
  "yolo12s.pt"
  "yolo12m.pt"
  "yolo12l.pt"
  "yolo12x.pt"
  "yolo11n-seg.pt"
  "yolo11s-seg.pt"
  "yolo11m-seg.pt"
  "yolo11l-seg.pt"
  "yolo11x-seg.pt"
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

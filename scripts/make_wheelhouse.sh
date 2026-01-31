#!/usr/bin/env bash
set -euo pipefail
# make_wheelhouse.sh - download wheels for offline installation
# Usage: ./scripts/make_wheelhouse.sh [requirements-file] [dest-dir]
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REQ_FILE="${1:-$ROOT/requirements.cuda.txt}"
DEST_DIR="${2:-$ROOT/wheelhouse}"
EXTRA_INDEX="https://download.pytorch.org/whl/cu131"

mkdir -p "$DEST_DIR"
echo "Downloading wheels listed in $REQ_FILE to $DEST_DIR"

# Create a filtered requirements list for wheel download to avoid dependency resolver conflicts
# Exclude torch stack, ultralytics, and opencv to handle them specifically
TMP_REQ="$(mktemp)"
grep -vE '^(torch|torchvision|torchaudio|ultralytics|ultralytics-thop|opencv-python)(==|=| )' "$REQ_FILE" > "$TMP_REQ" || cp "$REQ_FILE" "$TMP_REQ"

echo "Downloading non-torch packages to wheelhouse..."
python3 -m pip download --dest "$DEST_DIR" -r "$TMP_REQ"

echo "Downloading torch stack (from cu131 index) into wheelhouse..."
python3 -m pip download --dest "$DEST_DIR" --extra-index-url "$EXTRA_INDEX" \
	torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 || true

echo "Downloading ultralytics wheels into wheelhouse..."
python3 -m pip download --dest "$DEST_DIR" --no-deps ultralytics==8.4.8 ultralytics-thop==2.0.18 || true

echo "Downloading build helper wheels (pip/setuptools/wheel/wheel_stub) into wheelhouse..."
python3 -m pip download --dest "$DEST_DIR" pip setuptools wheel wheel_stub || true

echo "Downloading TensorRT-related artifacts into wheelhouse (if available)..."
# Try common TensorRT packages (metapackage, cu13 binary layer, bindings, libs)
python3 -m pip download --dest "$DEST_DIR" tensorrt tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs || true

rm -f "$TMP_REQ"

echo "Wheelhouse populated: $DEST_DIR"

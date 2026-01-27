#!/usr/bin/env bash
# setup.sh - create venv, install requirements, download and convert models
# Usage: ./setup.sh
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
LOG_FILE="$ROOT/setup.log"

# Redirect stdout and stderr to both console and log file
# Using 'tee' to keep output visible in terminal while saving to file
echo "Logging setup to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

VENV="$ROOT/.venv"
PY="$VENV/Scripts/python.exe"

# Define subdirectory structure for models
MODELS_DIR="$ROOT/models"
PT_DIR="$MODELS_DIR/pt"
ONNX_DIR="$MODELS_DIR/onnx"
TRT_DIR="$MODELS_DIR/tensorrt"
OV_DIR="$MODELS_DIR/openvino"

mkdir -p "$PT_DIR" "$ONNX_DIR" "$TRT_DIR" "$OV_DIR"

# 0) Clone LWCC if missing (external dependency)
EXTERNAL_DIR="$ROOT/external"
LWCC_REPO_DIR="$EXTERNAL_DIR/lwcc"
mkdir -p "$EXTERNAL_DIR"
if [ ! -d "$LWCC_REPO_DIR" ]; then
    echo "Cloning LWCC from fork..."
    git clone https://github.com/Nutz95/lwcc.git "$LWCC_REPO_DIR"
else
    echo "LWCC already exists in $LWCC_REPO_DIR"
fi

# Set PYTHONPATH to include lwcc library
export PYTHONPATH="$LWCC_REPO_DIR:$PYTHONPATH"

# Migrate existing files if any
echo "Migrating existing models to new structure..."
# Root moves (files)
mv "$ROOT"/*.pt "$PT_DIR/" 2>/dev/null || true
mv "$ROOT"/*.onnx "$ONNX_DIR/" 2>/dev/null || true
mv "$ROOT"/*.onnx.data "$ONNX_DIR/" 2>/dev/null || true
mv "$ROOT"/*.engine "$TRT_DIR/" 2>/dev/null || true

# Root moves (folders)
for d in "$ROOT"/*_openvino_model/; do
    [ -d "$d" ] || continue
    target="$OV_DIR/$(basename "$d")"
    echo "Relocating $d -> $target"
    if [ -d "$target" ]; then rm -rf "$target"; fi
    mv "$d" "$OV_DIR/" 2>/dev/null || true
done

# Old models directory moves (flat structure)
mv "$MODELS_DIR"/*.pt "$PT_DIR/" 2>/dev/null || true
mv "$MODELS_DIR"/*.onnx "$ONNX_DIR/" 2>/dev/null || true
mv "$MODELS_DIR"/*.onnx.data "$ONNX_DIR/" 2>/dev/null || true
mv "$MODELS_DIR"/*.engine "$TRT_DIR/" 2>/dev/null || true
mv "$MODELS_DIR"/*.xml "$OV_DIR/" 2>/dev/null || true
mv "$MODELS_DIR"/*.bin "$OV_DIR/" 2>/dev/null || true

# Move openvino folders from models/
for d in "$MODELS_DIR"/*_openvino_model/; do
    [ -d "$d" ] || continue
    target="$OV_DIR/$(basename "$d")"
    echo "Relocating $d -> $target"
    if [ -d "$target" ]; then rm -rf "$target"; fi
    mv "$d" "$OV_DIR/" 2>/dev/null || true
done

# 1) Create virtualenv if missing
if [ ! -f "$PY" ]; then
    echo "Creating virtualenv in $VENV"
    python -m venv "$VENV"
fi

# 2) Activate and upgrade pip
echo "Activating virtualenv and installing requirements..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

# 3) Save current environment packages (optional)
echo "To save your environment packages run:"
echo "  $PY -m pip freeze > requirements.txt"

# 4) Download and prepare YOLO models (11, 12, 26 and -seg variants)
echo "Preparing YOLO models (Download + ONNX + OpenVINO + TensorRT)..."
MODELS_DIR="$MODELS_DIR" PT_DIR="$PT_DIR" ONNX_DIR="$ONNX_DIR" TRT_DIR="$TRT_DIR" OV_DIR="$OV_DIR" "$PY" prepare_yolo_all.py

# Download LWCC weights into ~/.lwcc/weights (used by export_density_to_onnx.py)
echo "Downloading LWCC pretrained weights..."
"$PY" download_lwcc_weights.py || echo "Warning: LWCC weight download failed or partial."

# 5) Export LWCC density models to ONNX
echo "Exporting LWCC models to ONNX..."
"$PY" export_density_to_onnx.py

# 6) Convert LWCC PTH to OpenVINO IR and ONNX
echo "Converting LWCC models to OpenVINO..."
"$PY" convert_pth_to_openvino.py

# 7) Convert any remaining or new ONNX to TRT (TensorRT)
echo "Ensuring all ONNX models are converted to TensorRT if CUDA is present..."
for onnx in "$ONNX_DIR"/*.onnx; do
    [ -e "$onnx" ] || continue
    engine="$TRT_DIR/$(basename "$onnx" .onnx).engine"
    if [ ! -f "$engine" ]; then
        echo "Converting $(basename "$onnx") -> $(basename "$engine")"
        "$PY" convert_onnx_to_trt.py "$onnx" "$engine" 32
    else
        echo "Skipping: $engine already exists."
    fi
done

echo "Setup completed successfully."
echo "Models are organized in $MODELS_DIR:"
echo "  - PyTorch:  $PT_DIR"
echo "  - ONNX:     $ONNX_DIR"
echo "  - TensorRT: $TRT_DIR"
echo "  - OpenVINO: $OV_DIR"

echo "Setup complete. Run the app with: ./run_people_counter_rtx.sh 4k yolo11s-seg.engine 0.5 1 0 25 1 1"

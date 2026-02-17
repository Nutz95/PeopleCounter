#!/usr/bin/env bash
set -euo pipefail

# 3c_prepare_int8_optimized.sh
# Build INT8 engines with MINIMAL COCO download (1GB instead of 18GB+)

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="${IMAGE_NAME:-people-counter:gpu-final}"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Run ./0_build_image.sh first."
    exit 1
fi

# Parameters
MODEL_VARIANT="${1:-yolo26n-seg}"
NUM_IMAGES="${2:-1000}"
BATCH_SIZE="${3:-100}"
CLEAN_CACHE="${4:-yes}"

echo "🚀 Building INT8 TensorRT engine (OPTIMIZED)"
echo "   Model: ${MODEL_VARIANT}"
echo "   Calibration images: ${NUM_IMAGES} (downloads ~300MB instead of 18GB!)"
echo "   Batch size: ${BATCH_SIZE} (much faster than batch=1)"
echo "   Clean cache: ${CLEAN_CACHE}"
echo ""

# Build command
CMD="python3 prepare_yolo_int8_fixed.py \
  --model models/pt/${MODEL_VARIANT}.pt \
  --output models/tensorrt/${MODEL_VARIANT}-int8.engine \
  --calibration-images ${NUM_IMAGES} \
  --batch-size ${BATCH_SIZE}"

if [ "$CLEAN_CACHE" = "yes" ]; then
    CMD="$CMD --clean-cache"
fi

echo "⏱️  Estimated time: 5-15 minutes (first run downloads 1GB COCO val2017)"
echo ""

docker run --rm -it \
    --gpus all \
    --shm-size=4g \
    -v "$PWD:/app" \
    -w /app \
    "$IMAGE_NAME" \
    bash -c "set -e; $CMD"

echo ""
echo "✅ INT8 engine ready: models/tensorrt/${MODEL_VARIANT}-int8.engine"
echo ""
echo "📝 Update app_v2/config/pipeline.yaml:"
echo "   models:"
echo "     yolo_global:"
echo "       engine: models/tensorrt/${MODEL_VARIANT}-int8.engine"
echo "     yolo_tiles:"
echo "       engine: models/tensorrt/${MODEL_VARIANT}-int8.engine"
echo ""
echo "🧪 Test: ./5_run_tests.sh --app-version v2 -k test_pipeline_e2e"

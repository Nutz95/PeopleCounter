#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 3_prepare_models.sh — Build all YOLO26 model variants in every format:
#
#   Step 1 — FP32 baseline (ONNX + OpenVINO + TRT FP32) via prepare_models.py
#             Called for all yolo26{n,s,m,l,x} and yolo26{n,s,m,l,x}-seg
#
#   Step 2 — FP16 mixed-precision TRT engine for each *-seg variant
#             Uses prepare_yolo_autocast_fp16.py (TRT FP16 builder flag)
#             Best compute-time at batch=1 for yolo26m-seg (global tile)
#
#   Step 3 — FP8-QDQ TRT engine for each *-seg variant
#             Uses prepare_yolo_modelopt_fp8.py (ModelOpt Q/DQ insertion)
#             Best throughput at batch>=8 for yolo26m-seg (−13% vs FP32)
#
# Environment variables:
#   IMAGE_NAME          Docker image to use (default: people-counter:gpu-final-nvdec)
#   SKIP_FP16=1         Skip FP16 generation
#   SKIP_FP8=1          Skip FP8-QDQ generation
#   SEG_MODELS          Space-separated list of -seg base names to quantise
#                       (default: yolo26n-seg yolo26s-seg yolo26m-seg yolo26l-seg yolo26x-seg)
# ---------------------------------------------------------------------------
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="${IMAGE_NAME:-people-counter:gpu-final-nvdec}"
SKIP_FP16="${SKIP_FP16:-0}"
SKIP_FP8="${SKIP_FP8:-0}"
SKIP_DENSITY="${SKIP_DENSITY:-0}"
# DENSITY_CALIB_DIR: path to UCF-QNRF Train/img folder for FP8 calibration
# Leave unset to build FP16 only (FP8 requires UCF-QNRF_ECCV18.zip dataset)
DENSITY_CALIB_DIR="${DENSITY_CALIB_DIR:-}"
SEG_MODELS="${SEG_MODELS:-yolo26n-seg yolo26s-seg yolo26m-seg yolo26l-seg yolo26x-seg}"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️  Docker image $IMAGE_NAME not found."
    echo "    Run ./0_build_image.sh then ./1_prepare.sh (and ./2_prepare_nvdec.sh) first."
    exit 1
fi

DOCKER_RUN="docker run --rm -it --gpus all -v $PWD:/app -w /app $IMAGE_NAME"

# ---------------------------------------------------------------------------
# Step 1 — FP32 baseline (ONNX + OpenVINO + TRT FP32)
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3 — FP32 baseline (all yolo26 variants)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$DOCKER_RUN bash -c "set -e; python3 prepare_models.py"

# ---------------------------------------------------------------------------
# Step 2 — FP16 mixed-precision engines for each *-seg variant
# ---------------------------------------------------------------------------
if [[ "$SKIP_FP16" == "1" ]]; then
    echo ""
    echo "Step 2/3 — FP16 skipped (SKIP_FP16=1)"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 2/3 — FP16 engines (TRT FP16 builder flag)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for model in $SEG_MODELS; do
        onnx_in="models/onnx/${model}.onnx"
        engine_out="models/tensorrt/${model}-fp16-mixed.engine"
        echo "  → $model : $engine_out"
        $DOCKER_RUN bash -c "set -e
            if [ -f '$engine_out' ]; then
                echo '     [skip] engine already exists: $engine_out'
            else
                python3 prepare_yolo_autocast_fp16.py \
                    --onnx-in '$onnx_in' \
                    --engine-out '$engine_out' \
                    --timing-cache models/tensorrt/timing_cache.bin
            fi"
    done
fi

# ---------------------------------------------------------------------------
# Step 3 — FP8-QDQ engines for each *-seg variant
# ---------------------------------------------------------------------------
if [[ "$SKIP_FP8" == "1" ]]; then
    echo ""
    echo "Step 3/3 — FP8-QDQ skipped (SKIP_FP8=1)"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 3/3 — FP8-QDQ engines (ModelOpt Q/DQ calibration)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Note: requires people-counter:gpu-final-nvdec (includes ModelOpt + polygraphy)"
    for model in $SEG_MODELS; do
        model_base="${model%-seg}-seg"   # normalise to e.g. yolo26n-seg
        engine_out="models/tensorrt/${model_base}-fp8-qdq.engine"
        echo "  → $model_base : $engine_out"
        $DOCKER_RUN bash -c "set -e
            if [ -f '$engine_out' ]; then
                echo '     [skip] engine already exists: $engine_out'
            else
                python3 prepare_yolo_modelopt_fp8.py --model '$model_base'
            fi"
    done
fi

echo ""
echo "✅ All YOLO models prepared inside $IMAGE_NAME."

# ---------------------------------------------------------------------------
# Step 4 — DM-Count QNRF density model
# ---------------------------------------------------------------------------
if [[ "$SKIP_DENSITY" == "1" ]]; then
    echo ""
    echo "Step 4/4 — Density model skipped (SKIP_DENSITY=1)"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 4/4 — DM-Count QNRF density model (FP16 default)"
    if [[ -n "$DENSITY_CALIB_DIR" ]]; then
        echo "           FP8 calibration: $DENSITY_CALIB_DIR"
    else
        echo "           FP8 skipped — set DENSITY_CALIB_DIR to enable"
        echo "           (download UCF-QNRF_ECCV18.zip from crcv.ucf.edu/data/ucf-qnrf/)"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    DENSITY_ARGS="--skip-fp8"
    [[ -n "$DENSITY_CALIB_DIR" ]] && DENSITY_ARGS="--calib-dir $DENSITY_CALIB_DIR"

    $DOCKER_RUN bash -c "set -e
        # 640×720 reference engine (18 tiles, ~92 ms, high accuracy)
        if [ -f 'models/tensorrt/dm_count_qnrf.engine' ]; then
            echo '     [skip] density 640×720 engine already exists'
        else
            python3 prepare_density_models.py $DENSITY_ARGS --tile-size 640x720
        fi
        # 1920×1088 fast engine (4 tiles, 2×2 over 4K, to benchmark)
        if [ -f 'models/tensorrt/dm_count_qnrf_1920x1088.engine' ]; then
            echo '     [skip] density 1920×1088 engine already exists'
        else
            python3 prepare_density_models.py $DENSITY_ARGS --tile-size 1920x1088
        fi"
fi

echo ""
echo "Engines available in models/tensorrt/:"
docker run --rm --gpus all -v "$PWD:/app" -w /app "$IMAGE_NAME" \
    bash -c "ls -lh models/tensorrt/*.engine 2>/dev/null || echo '  (none found)'"

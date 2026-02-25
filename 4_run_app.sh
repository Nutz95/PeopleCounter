#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

DEFAULT_SOURCE="/dev/video0"
SOURCE=""
PROFILE_NAME=""
POSITIONAL=()
DOCKER_ARGS_EXTRA=()
CUDA_PROFILE=0
APP_VERSION="${APP_VERSION:-v1}"
APP_VERSION_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile|-p)
            PROFILE_NAME="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE_NAME="${1#*=}"
            shift
            ;;
        --app-version)
            APP_VERSION_ARG="$2"
            shift 2
            ;;
        --app-version=*)
            APP_VERSION_ARG="${1#*=}"
            shift
            ;;
        --cuda-profile|--cuda-profiling)
            CUDA_PROFILE=1
            shift
            ;;
        --perf-log)
            DOCKER_ARGS_EXTRA+=("-e" "PERF_LOG=1")
            shift
            ;;
        -e|--env)
            DOCKER_ARGS_EXTRA+=("-e" "$2")
            shift 2
            ;;
        -e=*|--env=*)
            DOCKER_ARGS_EXTRA+=("-e" "${1#*=}")
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [source] [--profile name] [--app-version v1|v2] [--cuda-profile] [--perf-log] [-e KEY=VALUE]"
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

SOURCE="${POSITIONAL[0]:-$DEFAULT_SOURCE}"
SELECTED_APP_VERSION="${APP_VERSION_ARG:-$APP_VERSION}"
SELECTED_APP_VERSION="$(echo "$SELECTED_APP_VERSION" | tr '[:upper:]' '[:lower:]')"
if [[ "$SELECTED_APP_VERSION" != "v1" && "$SELECTED_APP_VERSION" != "v2" ]]; then
    echo "❌ Invalid app version: $SELECTED_APP_VERSION. Choose v1 or v2."
    exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-people-counter:gpu-final-nvdec}"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Run ./1_prepare.sh then ./2_prepare_nvdec.sh (or set IMAGE_NAME to an existing tag)."
    exit 1
fi

APP_DIR="app_v${SELECTED_APP_VERSION:1}"
if [[ ! -d "$APP_DIR" ]]; then
    echo "❌ Application folder $APP_DIR missing inside the workspace."
    exit 1
fi

echo "🚀 Launching PeopleCounter Container (app v$SELECTED_APP_VERSION)..."
echo "📸 Source: $SOURCE"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🖥️ GPU status (hôte) :"
    nvidia-smi
else
    echo "⚠️ nvidia-smi introuvable sur l’hôte"
fi

# Detection of source type
DOCKER_ARGS=("--gpus" "all" "-p" "5000:5000" "-e" "DISPLAY=$DISPLAY" "-v" "$PWD:/app" "-w" "/app/$APP_DIR" "-e" "PYTHONPATH=/app:/app/app_v1:/app/app_v2" "-e" "APP_VERSION=$SELECTED_APP_VERSION")

if [[ "$SOURCE" == http* ]] || [[ "$SOURCE" == rtsp* ]]; then
    echo "🌐 Using Network Stream mode"
    DOCKER_ARGS+=("-e" "CAPTURE_MODE=rtsp" "-e" "RTSP_URL=$SOURCE")
else
    echo "🔌 Using Local Device mode"
    DOCKER_ARGS+=("--device" "$SOURCE:$SOURCE" "-e" "CAPTURE_MODE=usb" "-e" "CAMERA_INDEX=$SOURCE")
fi

# --- CONFIGURATION GPU PAR DÉFAUT ---
DOCKER_ARGS+=("-e" "YOLO_BACKEND=tensorrt_native")
DOCKER_ARGS+=("-e" "LWCC_BACKEND=tensorrt")
DOCKER_ARGS+=("-e" "YOLO_MODEL=yolo26m-seg")
DOCKER_ARGS+=("-e" "YOLO_TILING=1")
DOCKER_ARGS+=("-e" "DENSITY_TILING=1")
DOCKER_ARGS+=("-e" "PEOPLE_COUNTER_CUDA_ARCH_LIST=8.6")

PROFILE_NAME="${PROFILE_NAME:-rtx_extreme}"
CONFIG_FILE="$BASE_DIR/scripts/configs/${PROFILE_NAME}.env"
CLEANUP_ENV=""
cleanup_profile_env() {
    [[ -n "$CLEANUP_ENV" ]] && rm -f "$CLEANUP_ENV"
}
trap cleanup_profile_env EXIT
if [[ -f "$CONFIG_FILE" ]]; then
    SANITIZED_ENV=$(mktemp)
    CLEANUP_ENV="$SANITIZED_ENV"
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line##+( )}"
        line="${line%%+( )}"
        line="${line#export }"
        if [[ -z "$line" ]]; then
            continue
        fi
        echo "$line" >> "$SANITIZED_ENV"
    done < "$CONFIG_FILE"
    if [[ -s "$SANITIZED_ENV" ]]; then
        DOCKER_ARGS+=("--env-file" "$SANITIZED_ENV")
        echo "🧭 Profile: $PROFILE_NAME (env file: $CONFIG_FILE)"
    else
        echo "⚠️ Profile '$PROFILE_NAME' contains no usable variables; skipping."
    fi
else
    echo "⚠️ Profile '$PROFILE_NAME' not found (looking for $CONFIG_FILE); proceeding with defaults."
fi
DOCKER_ARGS+=("-e" "ACTIVE_PROFILE=$PROFILE_NAME")
if [[ "$CUDA_PROFILE" -eq 1 ]]; then
    DOCKER_ARGS+=("-e" "ENABLE_CUDA_PROFILING=1")
fi
# Append any extra -e flags passed by the user (e.g. --perf-log or -e PERF_LOG=1)
if [[ ${#DOCKER_ARGS_EXTRA[@]} -gt 0 ]]; then
    DOCKER_ARGS+=("${DOCKER_ARGS_EXTRA[@]}")
fi

ENTRY_COMMAND="python3 main.py"
if [[ "$CUDA_PROFILE" -eq 1 ]]; then
    if [[ "$SELECTED_APP_VERSION" == "v1" ]]; then
        ENTRY_COMMAND="nsys profile --force-overwrite true --trace=cuda --capture-range=cudaProfilerApi --output=people_counter python camera_app_pipeline.py && nsys export --report summary people_counter.qdstrm"
    else
        ENTRY_COMMAND="nsys profile --force-overwrite true --trace=cuda --capture-range=cudaProfilerApi --output=people_counter python main.py && nsys export --report summary people_counter.qdstrm"
    fi
fi

echo "🌐 Web UI at http://localhost:5000 exposes profile controls, FPS, and the debug toggle."

docker run --rm -it \
    "${DOCKER_ARGS[@]}" \
    "$IMAGE_NAME" bash -c "set -e; if command -v nvidia-smi >/dev/null 2>&1; then echo '[containérisé] nvidia-smi'; nvidia-smi || true; else echo '[containérisé] nvidia-smi absent'; fi; if command -v nvcc >/dev/null 2>&1; then echo '[containérisé] nvcc --version'; nvcc --version || true; else echo '[containérisé] nvcc absent'; fi; ${ENTRY_COMMAND}"

echo "⏹️ Container stopped."

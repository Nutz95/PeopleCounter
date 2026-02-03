#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

# run_app.sh - Run the PeopleCounter application in Docker
# Usage: ./run_app.sh [/dev/video0 | http://...] [--profile balanced_tri_chip]

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

DEFAULT_SOURCE="/dev/video0"
SOURCE=""
PROFILE_NAME=""
POSITIONAL=()

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
        --help|-h)
            echo "Usage: $0 [source] [--profile name]"
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

SOURCE="${POSITIONAL[0]:-$DEFAULT_SOURCE}"
IMAGE_NAME="people-counter:gpu-final"

echo "🚀 Launching PeopleCounter Container..."
echo "📸 Source: $SOURCE"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🖥️ GPU status (hôte) :"
    nvidia-smi
else
    echo "⚠️ nvidia-smi introuvable sur l’hôte"
fi

# Detection of source type
DOCKER_ARGS=("--gpus" "all" "-p" "5000:5000" "-e" "DISPLAY=$DISPLAY")

# --- MODE DEVELOPPEMENT ---
# On monte le dossier actuel dans /app pour que les modifs soient instantanées
# On définit PYTHONPATH pour utiliser /app
DOCKER_ARGS+=("-v" "$PWD:/app" "-w" "/app" "-e" "PYTHONPATH=/app")

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
DOCKER_ARGS+=("-e" "YOLO_MODEL=yolo26s-seg")
DOCKER_ARGS+=("-e" "YOLO_TILING=1")
DOCKER_ARGS+=("-e" "DENSITY_TILING=1")

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
        line="${line##+( )}" # remove leading spaces
        line="${line%%+( )}" # remove trailing spaces
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

echo "🌐 Web UI at http://localhost:5000 exposes profile controls, FPS, and the debug toggle."

echo "[DEBUG] Installation/Verif des dependances rapides..."
# Run prepare_models.py inside the container to clone/install local lwcc if LWCC_GIT_URL is set
# and pre-download optional YOLO models (controlled by YOLO_PREPARE env var).
# We avoid installing 'lwcc' from PyPI here to prevent overriding local editable installs.
docker run --rm -it \
    "${DOCKER_ARGS[@]}" \
    "$IMAGE_NAME" bash -c "set -e; if command -v nvidia-smi >/dev/null 2>&1; then echo '[containérisé] nvidia-smi'; nvidia-smi || true; else echo '[containérisé] nvidia-smi absent'; fi; if command -v nvcc >/dev/null 2>&1; then echo '[containérisé] nvcc --version'; nvcc --version || true; else echo '[containérisé] nvcc absent'; fi; apt-get update -qq && apt-get install -y git unzip || true; python3 prepare_models.py || true; pip install --no-cache-dir flask screeninfo psutil matplotlib; python3 main.py"

echo "🌐 Visit http://localhost:5000 in your browser; pick a profile, start profiling, and watch the FPS overlay plus debug metrics without tailing logs."

# --- FIX PERMISSIONS ---
# Comme le conteneur tourne en root, on rend la main à l'utilisateur sur les fichiers créés (engines, models)
echo "🔧 Synchronisation des permissions..."
sudo chown -R $(id -u):$(id -g) models/ vendors/ 2>/unknown || true

echo "⏹️ Container stopped."

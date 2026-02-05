#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="people-counter:gpu-final"
PROFILE_NAME="rtx_extreme"
POSITIONAL=()
COMMAND=()

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
            echo "Usage: $0 [--profile name] [-- command arguments...]"
            echo "Helper to run an arbitrary command inside the PeopleCounter GPU container."
            exit 0
            ;;
        --)
            shift
            COMMAND=("$@")
            break
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac

done

if [[ ${#COMMAND[@]} -eq 0 ]]; then
    if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
        COMMAND=("${POSITIONAL[@]}")
    else
        COMMAND=("bash")
    fi
fi

CONFIG_FILE="$BASE_DIR/scripts/configs/${PROFILE_NAME}.env"
SANITIZED_ENV=""
if [[ -f "$CONFIG_FILE" ]]; then
    SANITIZED_ENV=$(mktemp)
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line##+( )}"
        line="${line%%+( )}"
        line="${line#export }"
        if [[ -n "$line" ]]; then
            echo "$line" >> "$SANITIZED_ENV"
        fi
    done < "$CONFIG_FILE"
fi

DOCKER_ARGS=("--gpus" "all" "-v" "$PWD:/app" "-w" "/app" "-e" "PYTHONPATH=/app" "-e" "ACTIVE_PROFILE=${PROFILE_NAME}")
if [[ -n "$SANITIZED_ENV" && -s "$SANITIZED_ENV" ]]; then
    trap 'rm -f "$SANITIZED_ENV"' EXIT
    DOCKER_ARGS+=("--env-file" "$SANITIZED_ENV")
fi

echo "Running inside container ${IMAGE_NAME} with profile ${PROFILE_NAME}: ${COMMAND[*]}"

docker run --rm -it "${DOCKER_ARGS[@]}" "$IMAGE_NAME" "${COMMAND[@]}"

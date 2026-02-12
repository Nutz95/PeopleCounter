#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="people-counter:gpu-final"

echo "🧱 Building base image ($IMAGE_NAME)..."
./build_image.sh

echo "✅ Base image built. You can now run:"
echo "   1) ./1_prepare.sh          -> python/runtime toolchain layers"
echo "   2) ./2_prepare_nvdec.sh    -> optional VPF/PyNvCodec layer (NVDEC support)"
echo "   3) ./3_prepare_models.sh   -> optional model conversion/build"

#!/usr/bin/env bash
set -euo pipefail

VPF_REPO_URL="${VPF_REPO_URL:-https://github.com/NVIDIA/VideoProcessingFramework.git}"
VPF_REF="${VPF_REF:-master}"
VPF_DIR="${VPF_DIR:-/opt/VideoProcessingFramework}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCHS="${CUDA_ARCHS:-86}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ Python introuvable: $PYTHON_BIN"
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "❌ nvcc introuvable. Cette étape doit tourner dans une image CUDA devel (pas runtime-only)."
    exit 1
fi

echo "📦 Installing VPF build dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build pkg-config \
    python3-dev \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev libavdevice-dev libswscale-dev \
    libnuma-dev libdrm-dev zlib1g-dev \
    ca-certificates

# NVIDIA codec headers package name differs depending on distro snapshot.
if apt-cache show ffnvcodec-dev >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends ffnvcodec-dev
elif apt-cache show libffmpeg-nvenc-dev >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends libffmpeg-nvenc-dev
else
    echo "⚠️ No distro package found for NVIDIA codec headers (ffnvcodec-dev/libffmpeg-nvenc-dev)."
    echo "   If build fails on nvcuvid/nvEncodeAPI headers, install Video Codec SDK headers manually."
fi

$PYTHON_BIN -m pip install --no-cache-dir --upgrade pip setuptools wheel scikit-build pybind11 "numpy<2.3"

if [[ -d "$VPF_DIR" ]]; then
    echo "🧹 Removing previous VPF directory: $VPF_DIR"
    rm -rf "$VPF_DIR"
fi

echo "📥 Cloning VPF from $VPF_REPO_URL ($VPF_REF)..."
git clone --depth 1 --branch "$VPF_REF" "$VPF_REPO_URL" "$VPF_DIR"

cd "$VPF_DIR"

if [[ ! -d "$CUDA_HOME" ]]; then
    echo "❌ CUDA_HOME not found: $CUDA_HOME"
    exit 1
fi

echo "🩹 Patching FFmpeg demuxer bitstream filter selection..."
"$PYTHON_BIN" - <<PY
from pathlib import Path

path = Path("$VPF_DIR/src/TC/src/FFmpegDemuxer.cpp")
if not path.exists():
    raise SystemExit(f"FFmpegDemuxer source missing: {path}")
old = """  const string bfs_name =
      is_mp4H264 ? "h264_mp4toannexb"
                 : is_mp4HEVC ? "hevc_mp4toannexb" : is_VP9 ? string() : "unknown";
"""
new = """  const string bfs_name =
      is_mp4H264 ? "h264_mp4toannexb"
                 : is_mp4HEVC ? "hevc_mp4toannexb"
                              : string();
"""
content = path.read_text()
if old not in content:
    print("🧷 FFmpegDemuxer filter patch already applied")
else:
    path.write_text(content.replace(old, new, 1))
    print("✅ FFmpegDemuxer now skips unknown bitstream filters")
PY

echo "🔧 Building VPF C++ core..."
export CUDAARCHS="$CUDA_ARCHS"
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(command -v "$PYTHON_BIN")" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
cmake --build build -j"$(nproc)"

if [[ -d "$VPF_DIR/build/src/PyNvCodec" ]]; then
    echo "🐍 Installing compiled PyNvCodec module into site-packages..."
    SITE_PACKAGES="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

    MODULE_PATH="$(find "$VPF_DIR/build/src/PyNvCodec" -maxdepth 1 -name "_PyNvCodec*.so" | head -n 1)"
    if [[ -z "$MODULE_PATH" ]]; then
        echo "❌ Compiled _PyNvCodec module not found in build/src/PyNvCodec"
        exit 1
    fi

    cp "$MODULE_PATH" "$SITE_PACKAGES/"

    if [[ -f "$VPF_DIR/src/PyNvCodec/PyNvCodec.py" ]]; then
        cp "$VPF_DIR/src/PyNvCodec/PyNvCodec.py" "$SITE_PACKAGES/"
    elif [[ -f "$VPF_DIR/build/src/PyNvCodec/PyNvCodec.py" ]]; then
        cp "$VPF_DIR/build/src/PyNvCodec/PyNvCodec.py" "$SITE_PACKAGES/"
    elif [[ ! -f "$SITE_PACKAGES/PyNvCodec.py" ]]; then
        cat > "$SITE_PACKAGES/PyNvCodec.py" <<'PY'
from _PyNvCodec import *
PY
    fi
else
    echo "❌ Could not find build/src/PyNvCodec in VPF build tree."
    exit 1
fi

echo "🧪 Verifying PyNvCodec import..."
$PYTHON_BIN - <<'PY'
import importlib
m = importlib.import_module("PyNvCodec")
print("✅ PyNvCodec import OK:", m.__name__)
PY

echo "✅ VPF + PyNvCodec installed successfully."

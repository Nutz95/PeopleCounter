#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VPF_REPO_URL="${VPF_REPO_URL:-https://github.com/NVIDIA/VideoProcessingFramework.git}"
VPF_REF="${VPF_REF:-master}"
VPF_DIR="${VPF_DIR:-/opt/VideoProcessingFramework}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ARCHS="${CUDA_ARCHS:-86}"
EXTERNALS_DIR="${EXTERNALS_DIR:-/app/externals}"
VIDEO_CODEC_SDK_VERSION="${VIDEO_CODEC_SDK_VERSION:-13.0.37}"
VIDEO_CODEC_SDK_ARCHIVE="NVIDIA_Video_Codec_SDK_${VIDEO_CODEC_SDK_VERSION}.zip"
VIDEO_CODEC_SDK_DOWNLOAD_URL="${VIDEO_CODEC_SDK_DOWNLOAD_URL:-https://developer.nvidia.com/downloads/video-codec-sdk/${VIDEO_CODEC_SDK_VERSION}/video_codec_sdk_${VIDEO_CODEC_SDK_VERSION}.zip}"
DEFAULT_VIDEO_CODEC_SDK_DIR="${EXTERNALS_DIR}/NVIDIA_Video_Codec_SDK_${VIDEO_CODEC_SDK_VERSION}"
REPO_EXTERNAL_SDK_DIR="${REPO_ROOT}/external/Video_Codec_SDK_${VIDEO_CODEC_SDK_VERSION}"
USER_VIDEO_CODEC_SDK_DIR="${VIDEO_CODEC_SDK_DIR:-}"
VIDEO_CODEC_SDK_DIR=""
VIDEO_CODEC_SDK_ZIP_PATH="${EXTERNALS_DIR}/${VIDEO_CODEC_SDK_ARCHIVE}"

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
    ca-certificates curl unzip

download_video_codec_sdk() {
    if [[ -n "$USER_VIDEO_CODEC_SDK_DIR" ]]; then
        if [[ ! -d "$USER_VIDEO_CODEC_SDK_DIR" ]]; then
            echo "❌ Provided VIDEO_CODEC_SDK_DIR does not exist: $USER_VIDEO_CODEC_SDK_DIR"
            exit 1
        fi
        VIDEO_CODEC_SDK_DIR="$USER_VIDEO_CODEC_SDK_DIR"
        echo "✅ Using provided Video Codec SDK at $VIDEO_CODEC_SDK_DIR"
        return
    fi

    if [[ -d "$REPO_EXTERNAL_SDK_DIR" ]]; then
        VIDEO_CODEC_SDK_DIR="$REPO_EXTERNAL_SDK_DIR"
        echo "✅ Using Video Codec SDK from repository external folder: $VIDEO_CODEC_SDK_DIR"
        return
    fi

    if [[ -d "$DEFAULT_VIDEO_CODEC_SDK_DIR" ]]; then
        VIDEO_CODEC_SDK_DIR="$DEFAULT_VIDEO_CODEC_SDK_DIR"
        echo "✅ Video Codec SDK already available at $VIDEO_CODEC_SDK_DIR"
        return
    fi

    mkdir -p "$EXTERNALS_DIR"
    echo "📥 Downloading Video Codec SDK ${VIDEO_CODEC_SDK_VERSION}..."
    if ! curl -fsSL "$VIDEO_CODEC_SDK_DOWNLOAD_URL" -o "$VIDEO_CODEC_SDK_ZIP_PATH"; then
        echo "⚠️ Unable to reach $VIDEO_CODEC_SDK_DOWNLOAD_URL (status 404 or network error)."
        echo "   Download the archive manually from https://developer.nvidia.com/downloads/video-codec-sdk/${VIDEO_CODEC_SDK_VERSION}/video_codec_sdk_${VIDEO_CODEC_SDK_VERSION}.zip"
        echo "   (you might need to sign in with a NVIDIA account), extract it into external/Video_Codec_SDK_${VIDEO_CODEC_SDK_VERSION}, or set VIDEO_CODEC_SDK_DIR before rerunning."
        exit 1
    fi

    echo "📦 Extracting $VIDEO_CODEC_SDK_ARCHIVE to $EXTERNALS_DIR..."
    unzip -q "$VIDEO_CODEC_SDK_ZIP_PATH" -d "$EXTERNALS_DIR"
    rm -f "$VIDEO_CODEC_SDK_ZIP_PATH"

    if [[ ! -d "$DEFAULT_VIDEO_CODEC_SDK_DIR" ]]; then
        echo "❌ Extraction did not produce expected directory: $DEFAULT_VIDEO_CODEC_SDK_DIR"
        exit 1
    fi

    VIDEO_CODEC_SDK_DIR="$DEFAULT_VIDEO_CODEC_SDK_DIR"
}

copy_nvcuvid_from_sdk() {
    if ldconfig -p | grep -q libnvcuvid >/dev/null 2>&1; then
        echo "✅ libnvcuvid already present"
        return 0
    fi

    if [[ -z "$VIDEO_CODEC_SDK_DIR" ]]; then
        echo "❌ NVIDIA Video Codec SDK directory not available; libnvcuvid is missing"
        exit 1
    fi

    if [[ ! -d "$VIDEO_CODEC_SDK_DIR" ]]; then
        echo "❌ VIDEO_CODEC_SDK_DIR does not point to a directory: $VIDEO_CODEC_SDK_DIR"
        exit 1
    fi

    LIB_SRC_DIR=""
    if [[ -d "$VIDEO_CODEC_SDK_DIR/lib/x64" ]]; then
        LIB_SRC_DIR="$VIDEO_CODEC_SDK_DIR/lib/x64"
    elif [[ -d "$VIDEO_CODEC_SDK_DIR/Lib/linux/stubs/x86_64" ]]; then
        LIB_SRC_DIR="$VIDEO_CODEC_SDK_DIR/Lib/linux/stubs/x86_64"
    else
        echo "❌ Video Codec SDK directory lacks a lib/x64 or Lib/linux/stubs/x86_64 folder"
        exit 1
    fi

    echo "📦 Copying libnvcuvid from Video Codec SDK..."
    shopt -s nullglob
    lib_files=("$LIB_SRC_DIR"/libnvcuvid.so*)
    shopt -u nullglob

    if [[ ${#lib_files[@]} -eq 0 ]]; then
        echo "❌ libnvcuvid files not found under $LIB_SRC_DIR"
        exit 1
    fi

    for f in "${lib_files[@]}"; do
        cp -a "$f" "$CUDA_HOME/lib64/"
    done

    HEADER_SRC_DIR=""
    if [[ -d "$VIDEO_CODEC_SDK_DIR/include" ]]; then
        HEADER_SRC_DIR="$VIDEO_CODEC_SDK_DIR/include"
    elif [[ -d "$VIDEO_CODEC_SDK_DIR/Interface" ]]; then
        HEADER_SRC_DIR="$VIDEO_CODEC_SDK_DIR/Interface"
    fi

    for header in nvcuvid.h nvEncodeAPI.h cuviddec.h; do
        if [[ -n "$HEADER_SRC_DIR" && -f "$HEADER_SRC_DIR/$header" ]]; then
            cp -a "$HEADER_SRC_DIR/$header" "$CUDA_HOME/include/"
        fi
    done

    ldconfig
    echo "✅ libnvcuvid is now available in $CUDA_HOME/lib64"
}

download_video_codec_sdk


# NVIDIA codec headers package name differs depending on distro snapshot.
if apt-cache show ffnvcodec-dev >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends ffnvcodec-dev
elif apt-cache show libffmpeg-nvenc-dev >/dev/null 2>&1; then
    apt-get install -y --no-install-recommends libffmpeg-nvenc-dev
else
    echo "⚠️ No distro package found for NVIDIA codec headers (ffnvcodec-dev/libffmpeg-nvenc-dev)."
    echo "   If build fails on nvcuvid/nvEncodeAPI headers, install Video Codec SDK headers manually."
fi

copy_nvcuvid_from_sdk

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

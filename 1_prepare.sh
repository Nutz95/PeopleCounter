#!/usr/bin/env bash
set -euo pipefail
shopt -s extglob

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

IMAGE_NAME="people-counter:gpu-final"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "⚠️ Docker image $IMAGE_NAME not found. Run ./0_build_image.sh to build it once before prepping."
    exit 1
fi

action() {
    echo "🧪 Running prep commands inside the container..."
    local docker_args=("--gpus" "all" "-e" "DISPLAY=$DISPLAY" "-v" "$PWD:/app" "-w" "/app")
    local container_id
    container_id=$(docker create "${docker_args[@]}" "$IMAGE_NAME" bash -c "set -e; 
    
    # System packages
    apt-get update -qq
    apt-get install -y --no-install-recommends git curl unzip ninja-build build-essential pkg-config cmake || true
    
    # Verify / repair CUDA toolchain
    if ! command -v nvcc >/dev/null 2>&1; then 
        echo '⚠️ nvcc introuvable dans l image de base; tentative d installation du toolkit CUDA 13.1...'
        apt-get install -y --no-install-recommends \
            cuda-nvcc-13-1 \
            cuda-compiler-13-1 \
            cuda-cudart-dev-13-1 \
            cuda-libraries-dev-13-1 || \
        apt-get install -y --no-install-recommends \
            cuda-toolkit-13-1 || \
        echo '⚠️ Installation du toolkit CUDA échouée'
    fi
    # VPF also needs CUDA development libraries such as NPP (CUDA::npps)
    if [[ ! -e /usr/local/cuda/lib64/libnpps.so && ! -e /usr/local/cuda/targets/x86_64-linux/lib/libnpps.so ]]; then
        echo '⚠️ Bibliothèques CUDA de développement incomplètes; installation de cuda-libraries-dev-13-1...'
        apt-get install -y --no-install-recommends cuda-libraries-dev-13-1 libnpp-dev-13-1 || \
        apt-get install -y --no-install-recommends cuda-toolkit-13-1 || \
        echo '⚠️ Installation des bibliothèques CUDA dev échouée'
    fi
    if command -v nvcc >/dev/null 2>&1; then
        echo '✅ nvcc détecté:'
        nvcc --version | tail -n 1 || true
        ls -ld /usr/local/cuda /usr/local/cuda/bin || true
    else
        echo '❌ nvcc reste introuvable après prepare : la couche NVDEC échouera. Rebuild recommandé via ./0_build_image.sh.'
        exit 1
    fi
    
    # Install TensorRT 10.15 runtime + trtexec (Blackwell sm_120 FP8/FP16 fixes)
    # TRT 10.15 resolves Blackwell perf regressions: +24% FP8 ConvNets, +9% FP16 FLUX
    echo '🔧 Installing TensorRT 10.15 (trtexec + runtime)...'
    apt-get install -y --no-install-recommends libnvinfer-bin || echo '⚠️ libnvinfer-bin apt install failed'
    if ! command -v trtexec >/dev/null 2>&1; then
        TRTEXEC_PATH=\$(find /usr -name trtexec 2>/dev/null | head -1)
        [ -n \"\$TRTEXEC_PATH\" ] && ln -sf \"\$TRTEXEC_PATH\" /usr/local/bin/trtexec || echo '⚠️ trtexec not found'
    fi
    if command -v trtexec >/dev/null 2>&1; then
        echo \"✅ trtexec: \$(trtexec --version 2>&1 | grep 'TensorRT v' | head -1 || echo 'version unknown')\"
    fi

    # Upgrade Python TensorRT bindings to 10.15 (needed to match new runtime)
    # NOTE: existing .engine files will be INCOMPATIBLE after this upgrade -> rebuild all engines
    echo '📦 Upgrading Python TensorRT to 10.15...'
    APT_TRT_VER=\$(dpkg -l libnvinfer-bin 2>/dev/null | awk '/^ii/{print \$3}' | cut -d- -f1 || echo 'unknown')
    echo \"ℹ️ apt libnvinfer version: \$APT_TRT_VER\"
    pip install --no-cache-dir --upgrade \
        'tensorrt==10.15.1.29' \
        'tensorrt_cu13==10.15.1.29' \
        'tensorrt_cu13_bindings==10.15.1.29' \
        'tensorrt_cu13_libs==10.15.1.29' \
        --extra-index-url https://pypi.ngc.nvidia.com || \
    pip install --no-cache-dir --upgrade \
        'tensorrt>=10.15,<10.16' \
        --extra-index-url https://pypi.ngc.nvidia.com || \
    echo '⚠️ TRT 10.15 pip wheels not found - engines will use existing TRT version'
    python3 -c 'import tensorrt; print(f\"✅ Python TRT: {tensorrt.__version__}\")' || \
        echo '⚠️ tensorrt import failed'

    # NVIDIA ModelOpt: quantization toolkit (FP8/INT8 with proper Q/DQ nodes)
    # + onnx-graphsurgeon for modelopt.onnx.quantization (ONNX-space Q/DQ insertion)
    echo '📦 Installing NVIDIA ModelOpt + ONNX tools...'
    pip install --no-cache-dir 'nvidia-modelopt[torch]' || \
    pip install --no-cache-dir 'nvidia-modelopt' || \
    echo '⚠️ nvidia-modelopt installation failed (optional - skip if not needed)'
    pip install --no-cache-dir 'nvidia-onnx-graphsurgeon' 'onnxconverter-common' || \
    { pip install --no-cache-dir 'onnx-graphsurgeon' --index-url https://pypi.ngc.nvidia.com && \
      pip install --no-cache-dir 'onnxconverter-common'; } || \
    echo '⚠️ onnx-graphsurgeon install failed (optional)'
    # lief + onnx>=1.19 for modelopt.onnx.quantization (no-deps to avoid protobuf upgrade)
    pip install --no-cache-dir 'lief' 'onnx>=1.19.0' --no-deps || \
    echo '⚠️ lief/onnx upgrade failed (optional for FP8 quantization)'
    python3 -c 'import modelopt; print(\"\u2705 nvidia-modelopt:\", modelopt.__version__)' 2>/dev/null || \
        echo 'ℹ️ nvidia-modelopt not available'
    python3 -c 'import onnx_graphsurgeon; print(\"\u2705 onnx-graphsurgeon:\", onnx_graphsurgeon.__version__)' 2>/dev/null || \
        echo 'ℹ️ onnx-graphsurgeon not available'

    # Python packages
    pip install --no-cache-dir 'cuda-python>=13.1.0,<14.0'
    pip install --no-cache-dir nvidia-ml-py flask screeninfo psutil matplotlib ninja pytest pycocotools
    pip uninstall -y pynvml || true
    
    exit 0")

    docker start -a "$container_id"
    echo "💾 Committing prep container to $IMAGE_NAME..."
    docker commit "$container_id" "$IMAGE_NAME" >/dev/null
    docker rm "$container_id" >/dev/null
}

action

echo "✅ Prepared image $IMAGE_NAME is ready."
echo "   Next: ./2_prepare_nvdec.sh (optional), ./3_prepare_models.sh (optional), then ./4_run_app.sh --profile ..."

#!/usr/bin/env bash
# Installs Python packages needed for ONNX model preparation inside the container.
# This script is mounted at /app/scripts/nvdec/install_python_packages.sh and executed
# by 2_prepare_nvdec.sh during the image build.
set -e

echo '📦 Installing ONNX tooling for model preparation...'

# onnxconverter-common : FP32→FP16 ONNX conversion (AutoCast fallback)
# lief                 : required by nvidia-modelopt for weight stripping
pip install --no-cache-dir 'onnxconverter-common' 'lief' || \
    echo '⚠️ onnxconverter-common/lief install failed (non-fatal)'

# onnx 1.19 : required by modelopt.onnx for FP8 Q/DQ + AutoCast (adds FLOAT4E2M1, etc.)
pip install --no-cache-dir 'onnx>=1.19.0,<1.20' --no-deps || \
    echo '⚠️ onnx>=1.19 install failed (non-fatal)'

# onnx-graphsurgeon : graph manipulation, used by modelopt internals
pip install --no-cache-dir 'onnx-graphsurgeon' || \
    echo '⚠️ onnx-graphsurgeon install failed (non-fatal)'

# Patch onnx-graphsurgeon 0.3.27 (PyPI) for onnx 1.19 compatibility.
# Problem: graphsurgeon 0.3.27 uses onnx.mapping.{TENSOR_TYPE_TO_NP_TYPE,NP_TYPE_TO_TENSOR_TYPE}
#          which were removed in onnx 1.19.
# Solution: replace with equivalent onnx.helper APIs, exactly as done in TRT's graphsurgeon 0.5.9.
# Reference: TensorRT/tools/onnx-graphsurgeon — uses onnx.helper.tensor_dtype_to_np_dtype() etc.
GS_DIR=$(python -c "import onnx_graphsurgeon, os; print(os.path.dirname(onnx_graphsurgeon.__file__))" 2>/dev/null || true)
if [[ -n "$GS_DIR" ]]; then
    echo "🔧 Patching onnx-graphsurgeon (in $GS_DIR) for onnx 1.19 compatibility..."
    # exporters/onnx_exporter.py: NP_TYPE_TO_TENSOR_TYPE → np_dtype_to_tensor_dtype
    sed -i 's/onnx\.mapping\.NP_TYPE_TO_TENSOR_TYPE\[np\.dtype(dtype)\]/onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))/g' \
        "$GS_DIR/exporters/onnx_exporter.py"
    # importers/onnx_importer.py: TENSOR_TYPE_TO_NP_TYPE membership check
    sed -i 's/onnx_type in onnx\.mapping\.TENSOR_TYPE_TO_NP_TYPE/onnx_type in onnx.helper.get_all_tensor_dtypes()/g' \
        "$GS_DIR/importers/onnx_importer.py"
    # importers/onnx_importer.py: TENSOR_TYPE_TO_NP_TYPE lookup
    sed -i 's/return onnx\.mapping\.TENSOR_TYPE_TO_NP_TYPE\[onnx_type\]/return onnx.helper.tensor_dtype_to_np_dtype(onnx_type)/g' \
        "$GS_DIR/importers/onnx_importer.py"
    # ir/graph.py: TENSOR_TYPE_TO_NP_TYPE lookup
    sed -i 's/onnx\.mapping\.TENSOR_TYPE_TO_NP_TYPE\[final_type\]/onnx.helper.tensor_dtype_to_np_dtype(final_type)/g' \
        "$GS_DIR/ir/graph.py"
    find "$GS_DIR" -name '*.pyc' -delete
    echo '✅ onnx-graphsurgeon patched'
    # Verify the patch worked
    python -c "import onnx_graphsurgeon; print('  graphsurgeon import OK')"
else
    echo '⚠️ onnx-graphsurgeon not found, skipping onnx.mapping patch'
fi

# polygraphy: required by modelopt.onnx.autocast ReferenceRunner
pip install --no-cache-dir 'polygraphy' --extra-index-url https://pypi.ngc.nvidia.com || \
    pip install --no-cache-dir 'polygraphy' || \
    echo '⚠️ polygraphy install failed (non-fatal)'

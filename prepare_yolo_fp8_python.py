#!/usr/bin/env python3
"""
FP8 TensorRT Engine Builder using Python API (no trtexec needed).

Builds engines with FP8+FP16 directly via Python TRT 10.14 API:
  - Same version as runtime → no magic-tag mismatch
  - No OBEY_PRECISION_CONSTRAINTS (would break on non-FP8-capable layers)
  - Dynamic optimization profile: batch 1..32 (works for global & tiles)

Usage (inside Docker with GPU):
  python3 prepare_yolo_fp8_python.py

Produces:
  models/tensorrt/yolo26n-seg-fp8-b32.engine  (dynamic batch 1..32)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import tensorrt as trt
    import onnx
except ImportError as e:
    print(f"Install dependencies: {e}")
    sys.exit(1)

print(f"TensorRT version: {trt.__version__}")

PT_PATH        = Path("models/pt/yolo26n-seg.pt")
ONNX_DYNAMIC   = Path("models/pt/yolo26n-seg-dynamic.onnx")
OUT_ENGINE     = Path("models/tensorrt/yolo26n-seg-fp8-b32.engine")
WORKSPACE_GB   = 16
MAX_BATCH      = 32
OPT_BATCH      = 16


def ensure_dynamic_onnx(pt_path: Path, onnx_path: Path) -> bool:
    """Export dynamic-batch ONNX from .pt, or verify the existing one is dynamic."""
    if onnx_path.exists():
        # Verify it's actually dynamic
        m = onnx.load(str(onnx_path))
        dim0 = m.graph.input[0].type.tensor_type.shape.dim[0]
        if dim0.dim_param:  # symbolic / dynamic
            print(f"Dynamic ONNX already exists: {onnx_path}  (batch axis: '{dim0.dim_param}')")
            return True
        else:
            print(f"Existing ONNX has STATIC batch={dim0.dim_value}, re-exporting as dynamic...")

    print(f"Exporting dynamic-batch ONNX from {pt_path}...")
    result = subprocess.run(
        [
            "python3", "-c",
            f"""
from ultralytics import YOLO
m = YOLO('{pt_path}')
m.export(format='onnx', imgsz=640, simplify=True, batch=1, dynamic=True, verbose=False)
print('Export done')
"""
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Export failed:")
        print(result.stderr[-3000:])
        return False

    # Ultralytics saves alongside .pt - move to ONNX_DYNAMIC path
    default_onnx = pt_path.with_suffix('.onnx')
    if default_onnx.exists() and default_onnx != onnx_path:
        default_onnx.rename(onnx_path)
        print(f"Moved {default_onnx} → {onnx_path}")

    if not onnx_path.exists():
        print(f"ERROR: Expected ONNX at {onnx_path} but not found")
        return False

    print(f"Dynamic ONNX ready: {onnx_path}")
    return True


def build_fp8_engine(onnx_path: Path, output_path: Path,
                     min_batch: int = 1, opt_batch: int = 16, max_batch: int = 32) -> bool:
    """Build FP8+FP16 TensorRT engine from ONNX using Python TRT API."""
    print(f"\n{'='*60}")
    print(f"Building FP8+FP16 engine  batch={min_batch}..{opt_batch}..{max_batch}")
    print(f"  ONNX  : {onnx_path}")
    print(f"  Output: {output_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config  = builder.create_builder_config()

    # Workspace: 16 GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * 1024 ** 3)

    # FP8 (Blackwell native) + FP16 fallback.
    # NOTE: Do NOT set OBEY_PRECISION_CONSTRAINTS – most ops don't have FP8 kernels
    # and the flag would force failures instead of graceful FP16 fallback.
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, 'FP8'):
        config.set_flag(trt.BuilderFlag.FP8)
        print("FP8 flag: enabled")
    else:
        print(f"WARNING: trt.BuilderFlag.FP8 not found in TRT {trt.__version__} – FP16 only")

    # Check hardware FP8 support (informational only)
    if hasattr(builder, 'platform_has_fast_fp8'):
        if builder.platform_has_fast_fp8:
            print("Hardware: FP8 tensor cores available (Blackwell/Ada)")
        else:
            print("Hardware: no native FP8 tensor cores – will use FP16 paths")

    # Parse ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)

    onnx_abs = onnx_path.resolve()
    old_cwd  = os.getcwd()
    os.chdir(onnx_abs.parent)  # needed for external weight files
    try:
        with open(onnx_abs, 'rb') as f:
            ok = parser.parse(f.read())
        if not ok:
            for i in range(parser.num_errors):
                print(f"ONNX error: {parser.get_error(i)}")
            return False
    finally:
        os.chdir(old_cwd)

    inp = network.get_input(0)
    print(f"Network input: {inp.name}  shape={inp.shape}")
    print(f"Network layers: {network.num_layers}")

    # Optimization profile: dynamic batch for flexibility
    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        min=(min_batch, 3, 640, 640),
        opt=(opt_batch, 3, 640, 640),
        max=(max_batch, 3, 640, 640),
    )
    config.add_optimization_profile(profile)

    print(f"Building serialized network...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: build_serialized_network returned None")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(serialized)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"Saved: {output_path}  ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    # 1. Make sure we have a dynamic-batch ONNX
    if not ensure_dynamic_onnx(PT_PATH, ONNX_DYNAMIC):
        sys.exit(1)

    # 2. Build a single FP8+FP16 engine with dynamic batch profile
    ok = build_fp8_engine(
        ONNX_DYNAMIC, OUT_ENGINE,
        min_batch=1, opt_batch=OPT_BATCH, max_batch=MAX_BATCH
    )

    print(f"\n{'='*60}")
    print(f"FP8 engine build: {'OK' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)

#!/usr/bin/env python3
"""
ModelOpt FP8 PTQ for any YOLO26*-seg model — ONNX-space Q/DQ insertion.

Uses modelopt.onnx.quantization to insert Q/DQ nodes directly in the
existing ONNX, then builds a TRT engine that fuses them to native FP8
tensor cores on Blackwell (sm_120).

Why ONNX-space (not PyTorch export):
  - torch 2.9 dynamo exporter can't handle tensorrt.quantize_op
  - Works directly on our existing validated dynamic ONNX
  - TRT 10.15 Q/DQ fusion → expected ~20-25% gain on larger models vs FP16

Usage inside Docker:
  docker run --rm --gpus all --shm-size=4g -v "$PWD:/app" -w /app \
    people-counter:gpu-final-nvdec python3 prepare_yolo_modelopt_fp8.py
  # or for a specific model:
  docker run ... python3 prepare_yolo_modelopt_fp8.py --model yolo26m-seg

Produces:
  models/pt/{MODEL}-fp8-qdq.onnx
  models/tensorrt/{MODEL}-fp8-qdq.engine  (dynamic batch 1..32)
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np

# ── Dependency checks ────────────────────────────────────────────────────────
# Compatibility patches for onnx 1.19+ with modelopt 0.41 + onnx-graphsurgeon 0.3.x
import onnx as _onnx_pre
import numpy as _np

# 1. FLOAT4E2M1=23: added in onnx spec after 1.17, needed by modelopt.onnx.quantization.int4
if not hasattr(_onnx_pre.TensorProto, "FLOAT4E2M1"):
    _onnx_pre.TensorProto.FLOAT4E2M1 = 23

# 2. onnx.mapping.TENSOR_TYPE_TO_NP_TYPE: removed in onnx 1.18+, used by onnx-graphsurgeon 0.3.x
if not hasattr(_onnx_pre, "mapping"):
    try:
        from onnx._mapping import TENSOR_TYPE_MAP as _ttm
        _t2np = {k: v.np_dtype for k, v in _ttm.items() if v.np_dtype is not None}
    except Exception:
        _t2np = {1: _np.float32, 2: _np.uint8, 3: _np.int8, 4: _np.uint16,
                 5: _np.int16, 6: _np.int32, 7: _np.int64, 9: _np.bool_,
                 10: _np.float16, 11: _np.float64, 12: _np.uint32, 13: _np.uint64}
    class _FakeMapping:
        TENSOR_TYPE_TO_NP_TYPE = _t2np
        NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in _t2np.items()}
    _onnx_pre.mapping = _FakeMapping()

try:
    import modelopt
    import modelopt.onnx.quantization as moq
    print(f"✅ ModelOpt:      {modelopt.__version__}")
except ImportError as e:
    print(f"❌ modelopt.onnx.quantization not available: {e}")
    print("   Run: pip install nvidia-modelopt nvidia-onnx-graphsurgeon")
    sys.exit(1)

try:
    import tensorrt as trt
    print(f"✅ TensorRT:      {trt.__version__}")
except ImportError as e:
    print(f"❌ TensorRT: {e}"); sys.exit(1)

try:
    import onnx
    print(f"✅ onnx:          {onnx.__version__}")
except ImportError as e:
    print(f"❌ onnx: {e}"); sys.exit(1)

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--model", default="yolo26n-seg",
                     help="Model base name, e.g. yolo26n-seg or yolo26m-seg (default: yolo26n-seg)")
_args = _parser.parse_args()
MODEL_NAME   = _args.model

# ── Config ────────────────────────────────────────────────────────────────────
PT_PATH      = Path(f"models/pt/{MODEL_NAME}.pt")
ONNX_FP16    = Path(f"models/pt/{MODEL_NAME}-dynamic.onnx")
ONNX_QDQ     = Path(f"models/pt/{MODEL_NAME}-fp8-qdq.onnx")
ENGINE_OUT   = Path(f"models/tensorrt/{MODEL_NAME}-fp8-qdq.engine")
ENGINE_FP16  = Path(f"models/tensorrt/{MODEL_NAME}.engine")
IMG_SIZE     = 640
CALIB_STEPS  = 50
WORKSPACE_GB = 16

# ── Calibration data ──────────────────────────────────────────────────────────
def get_calib_data() -> dict:
    from PIL import Image
    for coco_dir in [
        Path("datasets/coco/images/val2017"),
        Path("/root/.cache/ultralytics/datasets/coco/images/val2017"),
    ]:
        if coco_dir.exists():
            imgs = sorted(coco_dir.glob("*.jpg"))[:CALIB_STEPS]
            if imgs:
                arrays = []
                for p in imgs:
                    img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                    arrays.append(arr)
                print(f"✅ Calibration: {len(arrays)} COCO images from {coco_dir}")
                return {"images": np.stack(arrays, axis=0)}
    print("⚠️  No COCO images → random calibration data (less accurate)")
    return {"images": np.random.rand(CALIB_STEPS, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)}

# ── Auto-export dynamic ONNX from .pt if needed ───────────────────────────────
def ensure_dynamic_onnx() -> bool:
    if ONNX_FP16.exists():
        m = onnx.load(str(ONNX_FP16))
        dim0 = m.graph.input[0].type.tensor_type.shape.dim[0]
        if dim0.dim_param:
            print(f"✅ Dynamic ONNX exists: {ONNX_FP16}  (batch='{dim0.dim_param}')")
            return True
        print(f"⚠️  ONNX has static batch={dim0.dim_value}, re-exporting as dynamic...")

    if not PT_PATH.exists():
        print(f"❌ PT model not found: {PT_PATH}")
        return False

    print(f"Exporting dynamic ONNX from {PT_PATH}...")
    import subprocess
    result = subprocess.run(
        ["python3", "-c", f"""
from ultralytics import YOLO
m = YOLO('{PT_PATH}')
m.export(format='onnx', imgsz=640, simplify=True, batch=1, dynamic=True, verbose=False)
print('Export done')
"""],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Export stderr:", result.stderr[-2000:])
        return False

    # Ultralytics saves alongside .pt — move to ONNX_FP16 path
    default_onnx = PT_PATH.with_suffix('.onnx')
    if default_onnx.exists() and default_onnx != ONNX_FP16:
        default_onnx.rename(ONNX_FP16)
        print(f"Moved → {ONNX_FP16}")
    elif not ONNX_FP16.exists():
        print(f"❌ Expected ONNX at {ONNX_FP16} not found after export")
        return False

    print(f"✅ Dynamic ONNX ready: {ONNX_FP16}")
    return True


# ── Step 1-3: ONNX-space FP8 quantization ─────────────────────────────────────
def quantize_onnx() -> bool:
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Input ONNX: {ONNX_FP16}")

    if not ensure_dynamic_onnx():
        return False

    m = onnx.load(str(ONNX_FP16))
    print(f"  Nodes: {len(m.graph.node)}  |  opset: {m.opset_import[0].version}")

    print(f"\nCalibration data ({CALIB_STEPS} samples)...")
    calib_data = get_calib_data()
    print(f"  Shape: {calib_data['images'].shape}")

    print(f"\nModelOpt ONNX FP8 quantization → {ONNX_QDQ}")
    ONNX_QDQ.parent.mkdir(parents=True, exist_ok=True)

    moq.quantize(
        onnx_path=str(ONNX_FP16),
        quantize_mode="fp8",
        output_path=str(ONNX_QDQ),
        calibration_data=calib_data,
    )

    if not ONNX_QDQ.exists():
        print(f"❌ Output not created: {ONNX_QDQ}")
        return False

    m_out = onnx.load(str(ONNX_QDQ))
    qdq = sum(1 for n in m_out.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear"))
    size_mb = ONNX_QDQ.stat().st_size / 1e6
    print(f"✅ ONNX saved: {size_mb:.1f} MB  |  Q/DQ nodes: {qdq}", end="  ")
    if qdq > 0:
        print("→ TRT will fuse to FP8 tensor cores ✅")
        return True
    else:
        print("→ ⚠️  No Q/DQ — FP16 speed expected")
        return False

# ── Step 4: TRT engine ────────────────────────────────────────────────────────
try:
    from app_v2.infrastructure.timing_cache_manager import TimingCacheManager as _TCM_fp8
except ImportError:
    _TCM_fp8 = None  # type: ignore[assignment]


def build_trt_engine() -> bool:
    print(f"\n{'='*60}")
    print(f"Building TRT engine: {ONNX_QDQ} → {ENGINE_OUT}")

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config  = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * 1024 ** 3)

    # Timing cache — persists kernel tactic choices across rebuilds (deterministic on sm_120)
    _cache_path = ENGINE_OUT.parent / "timing_cache.bin"
    _tc = _TCM_fp8(str(_cache_path)) if _TCM_fp8 is not None else None
    if _tc is not None:
        _tc.load_into_config(config)
    # Precision flags only needed when NOT using STRONGLY_TYPED (set conditionally below)

    # STRONGLY_TYPED is required for FP8 Q/DQ nodes in TRT 10.x (not EXPLICIT_BATCH)
    # This allows TRT to correctly fuse QuantizeLinear/DequantizeLinear into FP8 kernels
    # IMPORTANT: STRONGLY_TYPED is mutually exclusive with FP8/FP16 BuilderFlags —
    # types come from the Q/DQ ONNX nodes themselves when STRONGLY_TYPED is active.
    strongly_typed_flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if strongly_typed_flag is not None:
        network = builder.create_network(1 << int(strongly_typed_flag))
        print("  Network: STRONGLY_TYPED (FP8 Q/DQ fusion, no builder precision flags needed)")
        # Do NOT set FP8/FP16 flags with STRONGLY_TYPED — types are in ONNX Q/DQ nodes
    else:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        print("  Network: EXPLICIT_BATCH (STRONGLY_TYPED not available, using FP8 flag)")
        config.set_flag(trt.BuilderFlag.FP16)
        if hasattr(trt.BuilderFlag, "FP8"):
            config.set_flag(trt.BuilderFlag.FP8)
            print("  FP8 builder flag: enabled")
    parser  = trt.OnnxParser(network, logger)
    with open(ONNX_QDQ, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return False

    inp = network.get_input(0)
    print(f"  Input: {inp.name}  shape={inp.shape}  layers={network.num_layers}")

    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name,
        min=(1, 3, IMG_SIZE, IMG_SIZE),
        opt=(16, 3, IMG_SIZE, IMG_SIZE),
        max=(32, 3, IMG_SIZE, IMG_SIZE))
    config.add_optimization_profile(profile)

    print("  Building (2-5 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("❌ build_serialized_network returned None"); return False

    ENGINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ENGINE_OUT, "wb") as f:
        f.write(serialized)

    # Persist timing cache after successful build
    if _tc is not None:
        _tc.save_from_config(config)

    size_mb = ENGINE_OUT.stat().st_size / 1e6
    print(f"✅ Engine: {ENGINE_OUT} ({size_mb:.1f} MB)")

    if ENGINE_FP16.exists():
        fp16_mb = ENGINE_FP16.stat().st_size / 1e6
        ratio = fp16_mb / size_mb
        print(f"   FP16: {fp16_mb:.1f} MB  |  FP8 Q/DQ: {size_mb:.1f} MB  |  ratio: {ratio:.1f}x")
        print("✅ Real FP8 weights!" if ratio >= 1.5 else "⚠️  Similar size to FP16 — Q/DQ may have been folded")
    return True

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    has_qdq = quantize_onnx()
    ok = build_trt_engine()
    if not ok:
        sys.exit(1)
    print(f"\n{'='*60}")
    print("ModelOpt FP8 PTQ: DONE")
    if has_qdq:
        print("✅ Q/DQ exported → TRT uses native FP8 Blackwell kernels (~20-25% gain expected)")
    else:
        print("⚠️  No Q/DQ → FP16 speed expected (check ModelOpt version/config)")
    print(f"\nTo test: edit app_v2/config/pipeline.yaml → yolo_tiles engine: {ENGINE_OUT}")

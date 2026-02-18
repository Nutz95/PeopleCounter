#!/usr/bin/env python3
"""Convert yolo26n-seg FP32 ONNX to FP32/FP16 mixed-precision ONNX,
then build a TensorRT STRONGLY_TYPED engine.

Two conversion strategies are supported; the first available is used:
  1. ModelOpt AutoCast  (``nvidia-modelopt[onnx]`` with lief + onnx_graphsurgeon)
     -- selective mixed-precision based on calibration stats
  2. onnxconverter-common float16  (always available in our build image)
     -- converts all ops to FP16, keeps I/O as FP32 via Cast nodes

The resulting engine is equivalent to the "FP16 STRONGLY_TYPED" archetype,
faster than FP32 and easier to build than FP8 (no calibration data needed).

Usage (inside people-counter:gpu-final Docker):
    python prepare_yolo_autocast_fp16.py [--onnx-in models/onnx/yolo26n-seg.onnx]
                                         [--engine-out models/tensorrt/yolo26n-seg-fp16-mixed.engine]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import onnx

# ---------------------------------------------------------------------------
# Strategy 1: ModelOpt AutoCast (selective mixed-precision)
# Requires polygraphy for the ReferenceRunner used in threshold calibration.
# ---------------------------------------------------------------------------
_USE_MODELOPT = False
convert_to_mixed_precision = None
try:
    import polygraphy  # noqa: F401  — required by modelopt.onnx.autocast internals
    from modelopt.onnx.autocast import convert_to_mixed_precision  # type: ignore[no-redef]
    _USE_MODELOPT = True
    print("[AutoCast] Using ModelOpt AutoCast (selective FP32/FP16)")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Strategy 2: onnxconverter-common (always-on fallback)
# ---------------------------------------------------------------------------
_oc_float16 = None
if not _USE_MODELOPT:
    try:
        from onnxconverter_common import float16 as _oc_float16
        print("[AutoCast] ModelOpt/polygraphy not available; using onnxconverter-common float16")
    except ImportError as exc:
        print(f"[AutoCast] ERROR: No FP16 converter available: {exc}")
        sys.exit(1)

try:
    import tensorrt as trt
except ImportError as exc:
    print(f"[AutoCast] ERROR: TensorRT Python bindings not available: {exc}")
    sys.exit(1)

try:
    from app_v2.infrastructure.timing_cache_manager import TimingCacheManager
except ImportError:
    TimingCacheManager = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_ONNX_IN = "models/onnx/yolo26n-seg.onnx"
DEFAULT_ENGINE_OUT = "models/tensorrt/yolo26n-seg-fp16-mixed.engine"
DEFAULT_TIMING_CACHE = "models/tensorrt/timing_cache.bin"


def convert_onnx_to_fp16(onnx_in: Path, onnx_out: Path) -> bool:
    """Apply FP16 conversion via ModelOpt AutoCast when possible.

    Returns True if *onnx_out* was written (STRONGLY_TYPED engine recommended).
    Returns False when AutoCast is unavailable or fails (caller should build a
    standard FP16-flag engine from the original FP32 ONNX instead).
    """
    print(f"[AutoCast] Converting {onnx_in} → {onnx_out}")

    if not _USE_MODELOPT:
        print("[AutoCast] ModelOpt not available; using TRT FP16 builder flag instead")
        return False

    try:
        converted = convert_to_mixed_precision(
            onnx_path=str(onnx_in),
            low_precision_type="fp16",
            keep_io_types=True,
            data_max=8.0,
            init_max=8.0,
        )
        onnx_out.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(converted, str(onnx_out))
        print(f"[AutoCast] FP16-mixed ONNX saved to {onnx_out} ({onnx_out.stat().st_size // 1024} KB)")
        return True
    except Exception as exc:
        print(f"[AutoCast] ModelOpt AutoCast failed: {exc}")
        print("[AutoCast] Falling back to TRT FP16 builder flag on original FP32 ONNX")
        return False


def build_trt_engine(
    onnx_path: Path,
    engine_out: Path,
    workspace_gb: int,
    timing_cache_path: Path | None,
    *,
    strongly_typed: bool = True,
) -> None:
    """Build TRT engine from ONNX.

    *strongly_typed=True*: sets NetworkDefinitionCreationFlag.STRONGLY_TYPED (for FP16-mixed ONNX).
    *strongly_typed=False*: standard network + FP16 builder flag (engine uses FP16 internally).
    """
    print(f"[AutoCast] Building TRT engine from {onnx_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if strongly_typed:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
    else:
        network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if not strongly_typed:
        config.set_flag(trt.BuilderFlag.FP16)

    # Timing cache
    timing_mgr = None
    if timing_cache_path and TimingCacheManager is not None:
        timing_mgr = TimingCacheManager(timing_cache_path)
        timing_mgr.load_into_config(config)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        data = f.read()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(f"  TRT parse error {i}: {parser.get_error(i)}")
        sys.exit(1)

    # Dynamic batch profile: 1..32 (matches existing FP8 engine)
    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, 3, 640, 640), (8, 3, 640, 640), (32, 3, 640, 640))
    config.add_optimization_profile(profile)

    print("[AutoCast] Building engine (dynamic batch 1..32)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[AutoCast] ERROR: build_serialized_network returned None")
        sys.exit(1)

    if timing_mgr is not None:
        timing_mgr.save_from_config(config)

    engine_out.parent.mkdir(parents=True, exist_ok=True)
    engine_out.write_bytes(bytes(serialized))
    size_mb = len(bytes(serialized)) / (1024 * 1024)
    print(f"[AutoCast] Engine saved to {engine_out} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build YOLO FP16-mixed TRT engine (AutoCast or onnxconverter-common)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-in", default=DEFAULT_ONNX_IN, help="Input FP32 ONNX model")
    parser.add_argument("--engine-out", default=DEFAULT_ENGINE_OUT, help="Output TRT engine path")
    parser.add_argument("--workspace-gb", type=int, default=4, help="TRT workspace (GB)")
    parser.add_argument(
        "--timing-cache",
        default=DEFAULT_TIMING_CACHE,
        help="Timing cache path (shared with FP8 builder)",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep intermediate FP16-mixed ONNX file",
    )
    args = parser.parse_args()

    onnx_in = Path(args.onnx_in)
    if not onnx_in.exists():
        print(f"[AutoCast] ERROR: Input ONNX not found: {onnx_in}")
        sys.exit(1)

    engine_out = Path(args.engine_out)
    timing_cache = Path(args.timing_cache) if args.timing_cache else None

    if args.keep_onnx:
        mixed_onnx = engine_out.with_name(engine_out.stem + "-fp16-mixed.onnx")
        strongly_typed = convert_onnx_to_fp16(onnx_in, mixed_onnx)
        src_onnx = mixed_onnx if strongly_typed else onnx_in
        build_trt_engine(src_onnx, engine_out, args.workspace_gb, timing_cache, strongly_typed=strongly_typed)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            mixed_onnx = Path(tmp) / "yolo26n-seg-fp16-mixed.onnx"
            strongly_typed = convert_onnx_to_fp16(onnx_in, mixed_onnx)
            src_onnx = mixed_onnx if strongly_typed else onnx_in
            build_trt_engine(src_onnx, engine_out, args.workspace_gb, timing_cache, strongly_typed=strongly_typed)

    print("[AutoCast] Done.")


if __name__ == "__main__":
    main()


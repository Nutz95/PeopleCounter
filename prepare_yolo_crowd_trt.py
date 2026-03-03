#!/usr/bin/env python3
"""Build a TensorRT FP16 engine for the YOLO-CROWD model.

Reads the dynamic-batch ONNX produced by export_yolocrowd_to_onnx.py and
builds a TRT engine that supports batch = 1..32 at 640×640.

The YOLO-CROWD model outputs in YOLOv5 format: [batch, 25200, 6] pre-decoded
anchors (cx_px, cy_px, w_px, h_px, obj_conf, cls_conf) — **no NMS baked in**.
NMS is applied client-side by YoloDecoder._decode_raw_yolov5().

Usage (inside people-counter:gpu-final-nvdec Docker):
    python3 prepare_yolo_crowd_trt.py
    python3 prepare_yolo_crowd_trt.py --onnx-in models/pt/yolo-crowd-dynamic.onnx \\
                                      --engine-out models/tensorrt/yolo-crowd-fp16.engine
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tensorrt as trt  # type: ignore[import]
except ImportError as exc:
    print(f"[CROWD-TRT] ERROR: TensorRT Python bindings not available: {exc}")
    sys.exit(1)

try:
    from app_v2.infrastructure.timing_cache_manager import TimingCacheManager
except ImportError:
    TimingCacheManager = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ONNX_IN   = "models/pt/yolo-crowd-dynamic.onnx"
DEFAULT_ENGINE_OUT = "models/tensorrt/yolo-crowd-fp16.engine"
DEFAULT_TIMING_CACHE = "models/tensorrt/timing_cache.bin"


def build_trt_engine(
    onnx_path: Path,
    engine_out: Path,
    workspace_gb: int,
    timing_cache_path: Path | None,
) -> None:
    """Build a FP16 TRT engine from an ONNX with dynamic batch."""
    print(f"[CROWD-TRT] Building TRT engine from {onnx_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
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

    # Dynamic batch profile: 1..16..32 (matches yolo26n-fp8-qdq.engine profile)
    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, 3, 640, 640), (16, 3, 640, 640), (32, 3, 640, 640))
    config.add_optimization_profile(profile)

    print("[CROWD-TRT] Building engine (FP16, dynamic batch 1→16→32)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[CROWD-TRT] ERROR: build_serialized_network returned None")
        sys.exit(1)

    if timing_mgr is not None:
        timing_mgr.save_from_config(config)

    engine_out.parent.mkdir(parents=True, exist_ok=True)
    engine_out.write_bytes(bytes(serialized))
    size_mb = len(bytes(serialized)) / (1024 * 1024)
    print(f"[CROWD-TRT] Engine saved: {engine_out} ({size_mb:.1f} MB)")
    print("[CROWD-TRT] Output: output0 = [batch, 25200, 6] — YOLOv5 decoded format")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TRT FP16 engine for YOLO-CROWD model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-in",     default=DEFAULT_ONNX_IN,    help="Input ONNX path")
    parser.add_argument("--engine-out",  default=DEFAULT_ENGINE_OUT,  help="Output TRT engine path")
    parser.add_argument("--workspace-gb", type=int, default=4,        help="TRT workspace (GB)")
    parser.add_argument("--timing-cache", default=DEFAULT_TIMING_CACHE, help="Timing cache path")
    args = parser.parse_args()

    onnx_in = Path(args.onnx_in)
    if not onnx_in.exists():
        print(f"[CROWD-TRT] ERROR: ONNX not found: {onnx_in}")
        sys.exit(1)

    engine_out   = Path(args.engine_out)
    timing_cache = Path(args.timing_cache) if args.timing_cache else None

    if engine_out.exists():
        print(f"[CROWD-TRT] Engine already exists, skipping: {engine_out}")
        return

    build_trt_engine(onnx_in, engine_out, args.workspace_gb, timing_cache)


if __name__ == "__main__":
    main()

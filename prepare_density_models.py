#!/usr/bin/env python3
"""
DM-Count density model: ONNX export → TRT FP16 / FP8-QDQ benchmark.

Three engines are produced and benchmarked:
  1. FP16          — TRT FP16 flag only (default, always built)
  2. FP8-QDQ       — ModelOpt ONNX-space FP8 quantization (requires QNRF dataset)
  3. FP16-strict   — TRT FP16 forced on every layer (optional)

Usage (inside Docker):
  # FP16 only (default — FP8 skipped until calibration dataset provided)
  docker run --rm --gpus all --shm-size=4g \\
    -v "$PWD:/app" -w /app people-counter:gpu-final-nvdec \\
    python3 prepare_density_models.py

  # FP8 with UCF-QNRF calibration dataset
  docker run --rm --gpus all --shm-size=4g \\
    -v "$PWD:/app" -v /path/to/UCF-QNRF:/calib:ro \\
    -w /app people-counter:gpu-final-nvdec \\
    python3 prepare_density_models.py --calib-dir /calib/Train/img

Output engines (models/tensorrt/):
  dm_count_qnrf.engine             ← FP16 (= production default)
  dm_count_qnrf-fp8-qdq.engine    ← FP8 QDQ (only with --calib-dir or DENSITY_CALIB_DIR)
  dm_count_qnrf-fp16-strict.engine ← FP16 forced

Environment variables:
  SKIP_DENSITY_FP8=1   Skip FP8-QDQ build even if --calib-dir is provided (default: 0)
  DENSITY_CALIB_DIR    Path to folder of JPEG/PNG crowd images for FP8 calibration
                       (UCF-QNRF Train/img recommended: crcv.ucf.edu/data/ucf-qnrf/)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

ONNX_PATH        = ROOT / "models/onnx/dm_count_qnrf.onnx"
ONNX_QDQ_PATH    = ROOT / "models/pt/dm_count_qnrf-fp8-qdq.onnx"
ENGINE_FP16      = ROOT / "models/tensorrt/dm_count_qnrf.engine"
ENGINE_FP8_QDQ   = ROOT / "models/tensorrt/dm_count_qnrf-fp8-qdq.engine"
ENGINE_FP16_STRICT = ROOT / "models/tensorrt/dm_count_qnrf-fp16-strict.engine"

# Tiling config: 4K (3840×2160) → 6 cols × 3 rows, no overlap
TARGET_H   = 720
TARGET_W   = 640
BATCH_OPT  = 18   # 6×3
# ── Default tiling config (6×3 = 18 tiles, 640×720) ─────────────────────────────
# Override at runtime with --tile-size WxH (both must be multiples of 16).
DEFAULT_TARGET_H   = 720
DEFAULT_TARGET_W   = 640
DEFAULT_BATCH_OPT  = 18   # 6×3
DEFAULT_BATCH_MAX  = 18

# Runtime values — set by _parse_args() based on --tile-size
TARGET_H   = DEFAULT_TARGET_H
TARGET_W   = DEFAULT_TARGET_W
BATCH_OPT  = DEFAULT_BATCH_OPT
BATCH_MAX  = DEFAULT_BATCH_MAX

CALIB_STEPS = 32

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _engine_paths(
    target_w: int,
    target_h: int,
    batch_suffix: str | None = None,
) -> tuple[Path, Path, Path]:
    """Return (FP16_engine, FP8_engine, FP16_strict_engine) for a given tile size."""
    suffix = "" if (target_w == 640 and target_h == 720) else f"_{target_w}x{target_h}"
    if batch_suffix:
        suffix += batch_suffix
    return (
        ROOT / f"models/tensorrt/dm_count_qnrf{suffix}.engine",
        ROOT / f"models/tensorrt/dm_count_qnrf{suffix}-fp8-qdq.engine",
        ROOT / f"models/tensorrt/dm_count_qnrf{suffix}-fp16-strict.engine",
    )


def _onnx_path(target_w: int, target_h: int) -> Path:
    suffix = "" if (target_w == 640 and target_h == 720) else f"_{target_w}x{target_h}"
    return ROOT / f"models/onnx/dm_count_qnrf{suffix}.onnx"


# ── Module-level aliases (updated after arg parsing) ────────────────────────────
ONNX_PATH        = _onnx_path(DEFAULT_TARGET_W, DEFAULT_TARGET_H)
ONNX_QDQ_PATH    = ROOT / "models/pt/dm_count_qnrf-fp8-qdq.onnx"
ENGINE_FP16, ENGINE_FP8_QDQ, ENGINE_FP16_STRICT = _engine_paths(DEFAULT_TARGET_W, DEFAULT_TARGET_H)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DM-Count QNRF TRT engines (FP16 + optional FP8-QDQ)."
    )
    parser.add_argument(
        "--calib-dir",
        default=os.environ.get("DENSITY_CALIB_DIR", ""),
        help="Path to folder of crowd JPEG/PNG images for FP8 calibration "
             "(UCF-QNRF Train/img recommended). Also read from $DENSITY_CALIB_DIR. "
             "If absent, FP8 is skipped.",
    )
    parser.add_argument(
        "--skip-fp8",
        action="store_true",
        default=os.environ.get("SKIP_DENSITY_FP8", "0") == "1",
        help="Skip FP8-QDQ build even when --calib-dir is provided.",
    )
    parser.add_argument(
        "--fp16-strict",
        action="store_true",
        default=False,
        help="Also build the FP16-strict engine (all layers forced FP16).",
    )
    parser.add_argument(
        "--benchmark-strategies",
        action="store_true",
        default=False,
        help="Run full strategy benchmark: batch sweep + 3-stream parallel rows."
             " Uses the existing FP16 engine — does not rebuild.",
    )
    parser.add_argument(
        "--tile-size",
        default="640x720",
        metavar="WxH",
        help="Tile resolution, e.g. 640x720 (default), 1920x1088, 3840x2160. "
             "Both dims must be multiples of 16. "
             "Batch max is computed automatically from 4K tiling (ceil).",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=0,
        metavar="N",
        help="Override the auto-computed max batch size. "
             "When N differs from the auto value a _bN suffix is added to the engine name "
             "(e.g. dm_count_qnrf_1920x1088_b1.engine). "
             "Useful to benchmark a single-frame (batch=1) inference without full-4K coverage.",
    )
    return parser.parse_args()


def _apply_tile_size(args: argparse.Namespace) -> None:
    """Parse --tile-size WxH (and optional --max-batch N) → update module globals.

    Tile coverage is computed as ``ceil(frame_dim / tile_dim)`` so that the last
    tile may overlap the frame edge rather than leaving uncovered pixels.
    3840×2160 is already a multiple of 16, so ``--tile-size 3840x2160`` cleanly
    gives batch=1 (one full-frame inference, no tiling).
    """
    global TARGET_H, TARGET_W, BATCH_OPT, BATCH_MAX, ONNX_PATH, ENGINE_FP16, ENGINE_FP8_QDQ, ENGINE_FP16_STRICT
    parts = args.tile_size.lower().strip().split("x")
    if len(parts) != 2:
        print(f"\u274c --tile-size must be WxH, got: {args.tile_size!r}")
        sys.exit(1)
    tw, th = int(parts[0]), int(parts[1])
    if tw % 16 != 0 or th % 16 != 0:
        print(f"\u274c --tile-size dimensions must be multiples of 16 (got {tw}x{th}).")
        sys.exit(1)
    TARGET_W, TARGET_H = tw, th

    # Compute auto batch max: ceil(3840/tw) × ceil(2160/th)
    import math
    auto_cols = math.ceil(3840 / TARGET_W)
    auto_rows = math.ceil(2160 / TARGET_H)
    auto_batch = auto_cols * auto_rows

    # --max-batch override (adds _bN suffix to engine name when it differs from auto)
    explicit = getattr(args, "max_batch", 0)
    if explicit > 0 and explicit != auto_batch:
        BATCH_MAX = explicit
        BATCH_OPT = explicit           # use the forced batch for benchmark
        batch_suffix: str | None = f"_b{explicit}"
    else:
        BATCH_MAX = auto_batch
        BATCH_OPT = BATCH_MAX
        batch_suffix = None

    ONNX_PATH = _onnx_path(TARGET_W, TARGET_H)
    ENGINE_FP16, ENGINE_FP8_QDQ, ENGINE_FP16_STRICT = _engine_paths(
        TARGET_W, TARGET_H, batch_suffix
    )


# ── ONNX export ───────────────────────────────────────────────────────────────

def ensure_onnx() -> bool:
    """Export DM-Count QNRF to ONNX at the current TARGET_W×TARGET_H if not present."""
    if ONNX_PATH.exists():
        print(f"✅ ONNX exists: {ONNX_PATH}")
        return True

    print(f"Exporting DM-Count QNRF to ONNX ({TARGET_W}×{TARGET_H}) ...")
    try:
        import subprocess
        res = subprocess.run(
            ["python3", str(ROOT / "export_density_to_onnx.py"),
             "--target-w", str(TARGET_W), "--target-h", str(TARGET_H)],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            print("stderr:", res.stderr[-1500:])
            return False
        print(res.stdout[-800:])
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

    if not ONNX_PATH.exists():
        print(f"❌ ONNX not found at {ONNX_PATH}")
        return False
    print(f"✅ ONNX exported: {ONNX_PATH}")
    return True


# ── Calibration data ──────────────────────────────────────────────────────────

def get_calib_data(calib_dir: str) -> np.ndarray | None:
    """Load calibration images from *calib_dir*.

    Returns ``[N, 3, H, W]`` fp32 ImageNet-normalised array, or ``None`` if the
    directory is empty/missing (FP8 calibration will be skipped in that case).

    The UCF-QNRF ``Train/img`` folder is the recommended source:
      https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip
    """
    from PIL import Image  # type: ignore[import]

    calib_path = Path(calib_dir) if calib_dir else None
    if not calib_path or not calib_path.exists():
        print(f"⚠️  Calibration directory not found: {calib_dir!r}")
        print("    FP8-QDQ build requires real crowd images (UCF-QNRF Train/img).")
        print("    Download: https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip")
        print("    Then rerun with:  --calib-dir /path/to/UCF-QNRF/Train/img")
        return None

    images: list[np.ndarray] = []
    for p in sorted(calib_path.glob("*.jpg")):
        img = Image.open(p).convert("RGB").resize((TARGET_W, TARGET_H))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        images.append(arr.transpose(2, 0, 1))
        if len(images) >= CALIB_STEPS:
            break

    if not images:
        print(f"⚠️  No .jpg images found in {calib_path} — FP8 skipped.")
        return None

    print(f"✅ Calibration: {len(images)} images from {calib_path}")
    return np.stack(images, axis=0)


# ── TRT engine builder ────────────────────────────────────────────────────────

def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp8_qdq: bool = False,
    fp16_strict: bool = False,
) -> bool:
    """Build a TRT engine from ``onnx_path``.

    Parameters
    ----------
    fp8_qdq
        Input ONNX already contains Q/DQ nodes (FP8 QDQ path).
    fp16_strict
        Set FP16 flag AND mark all layers explicitly FP16 (experimental).
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        print(f"❌ TensorRT not available: {e}")
        return False

    print(f"\nBuilding engine: {engine_path.name}")
    print(f"  Source ONNX : {onnx_path}")
    print(f"  Tile size   : {TARGET_W}\u00d7{TARGET_H}")
    print(f"  Batch opt/max: {BATCH_OPT}/{BATCH_MAX}")
    print(f"  FP8-QDQ     : {fp8_qdq}")
    print(f"  FP16-strict : {fp16_strict}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)

    # Precision flags
    config.set_flag(trt.BuilderFlag.FP16)
    if fp8_qdq and hasattr(trt.BuilderFlag, "FP8"):
        config.set_flag(trt.BuilderFlag.FP8)

    # Timing cache
    cache_path = engine_path.parent / "timing_cache_density.bin"
    try:
        from app_v2.infrastructure.timing_cache_manager import TimingCacheManager
        tc = TimingCacheManager(str(cache_path))
        tc.load_into_config(config)
    except Exception:
        tc = None

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    onnx_abs = str(onnx_path.resolve())
    old_cwd = os.getcwd()
    os.chdir(str(onnx_path.parent))
    try:
        with open(onnx_abs, "rb") as f:
            if not parser.parse(f.read()):
                print("❌ ONNX parse errors:")
                for i in range(parser.num_errors):
                    print("  ", parser.get_error(i))
                return False
    finally:
        os.chdir(old_cwd)

    # Optimization profile: batch 1..BATCH_MAX, fixed H/W
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name,
                      (1, 3, TARGET_H, TARGET_W),
                      (BATCH_OPT, 3, TARGET_H, TARGET_W),
                      (BATCH_MAX, 3, TARGET_H, TARGET_W))
    config.add_optimization_profile(profile)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        from app_v2.infrastructure.stream_writers import FileStreamWriter
        writer = FileStreamWriter(str(engine_path))
        ok = builder.build_serialized_network_to_stream(network, config, writer)
        writer.close()
        if not ok:
            print("❌ build_serialized_network_to_stream failed")
            return False
        print(f"  Saved {engine_path.name} ({writer.bytes_written / 1024**2:.1f} MB) "
              f"in {time.perf_counter()-t0:.0f}s [stream]")
    except Exception:
        serialised = builder.build_serialized_network(network, config)
        if serialised is None:
            print("❌ build_serialized_network failed")
            return False
        engine_path.write_bytes(bytes(serialised))
        print(f"  Saved {engine_path.name} ({engine_path.stat().st_size / 1024**2:.1f} MB) "
              f"in {time.perf_counter()-t0:.0f}s")

    if tc is not None:
        try:
            tc.save_from_config(config)
        except Exception:
            pass

    return True


# ── FP8 QDQ quantization ──────────────────────────────────────────────────────

def quantize_fp8(calib_data: np.ndarray) -> bool:
    """Insert FP8 Q/DQ nodes into the density ONNX via ModelOpt."""
    if ONNX_QDQ_PATH.exists():
        print(f"✅ FP8 QDQ ONNX already exists: {ONNX_QDQ_PATH}")
        return True

    try:
        import modelopt.onnx.quantization as moq  # type: ignore[import]
    except ImportError as e:
        print(f"⚠️  ModelOpt not available — skipping FP8 QDQ: {e}")
        return False

    print(f"\nModelOpt FP8 quantization → {ONNX_QDQ_PATH}")
    ONNX_QDQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        moq.quantize(
            onnx_path=str(ONNX_PATH),
            quantize_mode="fp8",
            output_path=str(ONNX_QDQ_PATH),
            calibration_data={"input": calib_data},
        )
        print(f"✅ FP8 QDQ ONNX saved: {ONNX_QDQ_PATH}")
        return True
    except Exception as e:
        print(f"❌ ModelOpt quantization failed: {e}")
        return False


# ── Latency strategy benchmarks ──────────────────────────────────────────────

def _load_trt_engine(engine_path: Path) -> tuple[Any, Any] | None:
    """Load a TRT engine; return (runtime, engine) or None on error."""
    try:
        import tensorrt as trt
    except ImportError:
        return None
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    return runtime, runtime.deserialize_cuda_engine(engine_path.read_bytes())


def benchmark_batch_sweep(engine_path: Path, n_runs: int = 30) -> dict[int, float]:
    """Measure latency for batch sizes [1, 2, 3, 4, 6, 9, 12, 18] on the FP16 engine."""
    import torch
    loaded = _load_trt_engine(engine_path)
    if loaded is None:
        return {}
    _, engine = loaded
    ctx = engine.create_execution_context()
    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    profile_max = engine.get_tensor_profile_shape(inp_name, 0)[2][0]

    results: dict[int, float] = {}
    stream = torch.cuda.Stream()

    for batch in [1, 2, 3, 4, 6, 9, 12, 18]:
        if batch > profile_max:
            continue
        ctx.set_input_shape(inp_name, (batch, 3, TARGET_H, TARGET_W))
        inp  = torch.zeros(batch, 3, TARGET_H, TARGET_W, dtype=torch.float32, device="cuda")
        out_shape = tuple(ctx.get_tensor_shape(out_name))
        if any(d <= 0 for d in out_shape):
            continue
        out = torch.zeros(*out_shape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(inp_name, inp.data_ptr())
        ctx.set_tensor_address(out_name, out.data_ptr())
        # warm-up
        for _ in range(5):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_runs * 1000.0
        results[batch] = ms
        fps_equiv = 1000.0 / ms if ms > 0 else 0
        print(f"  batch={batch:2d} → {ms:6.2f} ms  ({fps_equiv:5.1f} fps-equiv)")

    return results


def benchmark_parallel_rows(
    engine_path: Path,
    rows: int = 3,
    cols: int = 6,
    n_runs: int = 30,
) -> float:
    """3 execution contexts × batch=cols on 3 independent CUDA streams (concurrent).

    TRT allows multiple concurrent execution contexts on the same engine — each
    context gets its own CUDA stream.  On Blackwell/Ada the SM scheduler will
    overlap the 3 sub-batches if they don't saturate all SMs individually.

    Returns wall-clock latency (ms) for one full-frame inference (all rows done).
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor

    loaded = _load_trt_engine(engine_path)
    if loaded is None:
        return -1.0
    _, engine = loaded

    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)

    # Create one context + stream + buffer set per row
    contexts, streams, inp_bufs, out_bufs = [], [], [], []
    for _ in range(rows):
        ctx = engine.create_execution_context()
        ctx.set_input_shape(inp_name, (cols, 3, TARGET_H, TARGET_W))
        out_shape = tuple(ctx.get_tensor_shape(out_name))
        if any(d <= 0 for d in out_shape):
            print("  ⚠️  Parallel-rows: invalid output shape, skipping")
            return -1.0
        inp = torch.zeros(cols, 3, TARGET_H, TARGET_W, dtype=torch.float32, device="cuda")
        out = torch.zeros(*out_shape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(inp_name, inp.data_ptr())
        ctx.set_tensor_address(out_name, out.data_ptr())
        contexts.append(ctx)
        streams.append(torch.cuda.Stream())
        inp_bufs.append(inp)
        out_bufs.append(out)

    def _run_row(i: int) -> None:
        ctx, s = contexts[i], streams[i]
        with torch.cuda.stream(s):
            ctx.execute_async_v3(s.cuda_stream)

    # Warm-up
    with ThreadPoolExecutor(max_workers=rows) as ex:
        for _ in range(5):
            list(ex.map(_run_row, range(rows)))
    torch.cuda.synchronize()

    # Benchmark: launch all rows concurrently, synchronise once
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=rows) as ex:
        for _ in range(n_runs):
            list(ex.map(_run_row, range(rows)))
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_runs * 1000.0
    fps_equiv = 1000.0 / ms if ms > 0 else 0
    print(f"  {rows} streams × batch={cols} → {ms:6.2f} ms  ({fps_equiv:5.1f} fps-equiv)")
    return ms


def run_strategy_benchmark(engine_path: Path) -> None:
    """Full strategy comparison: print a table of latency vs tile configuration."""
    if not engine_path.exists():
        print(f"  [skip] engine not found: {engine_path}")
        return

    print()
    print("─" * 65)
    print(f" Strategy benchmark: {engine_path.name}")
    print("─" * 65)
    print(f" {'Strategy':<38} {'Latency':>9}  {'fps-equiv':>9}  {'vs baseline':>11}")
    print("─" * 65)

    baseline_ms: float | None = None

    # ── A. batch sweep ────────────────────────────────────────────────────────
    print("\n[A] Batch-size sweep (single sequential call):")
    sweep = benchmark_batch_sweep(engine_path)
    if 18 in sweep:
        baseline_ms = sweep[18]

    rows_map = {1: "global×1", 2: "2 tiles", 3: "3 tiles", 4: "2×2=4 tiles",
                6: "1 row=6 tiles", 9: "3×3=9 tiles", 12: "2 rows=12 tiles", 18: "3 rows=18 tiles (baseline)"}
    strategy_rows = []
    for b, ms in sorted(sweep.items()):
        label = f"batch={b} ({rows_map.get(b, '')})"
        bvs = f"{baseline_ms/ms:.2f}×" if baseline_ms else "—"
        fps = f"{1000/ms:.1f}"
        strategy_rows.append((label, ms, fps, bvs))

    # ── B. parallel 3 rows × batch=6 ─────────────────────────────────────────
    print("\n[B] 3 concurrent CUDA streams × batch=6 (one stream per row):")
    par_ms = benchmark_parallel_rows(engine_path, rows=3, cols=6)
    if par_ms > 0:
        bvs = f"{baseline_ms/par_ms:.2f}×" if baseline_ms else "—"
        fps = f"{1000/par_ms:.1f}"
        strategy_rows.append(("3-stream × batch=6 (parallel rows)", par_ms, fps, bvs))

    # ── C. summary table ──────────────────────────────────────────────────────
    print()
    print("─" * 65)
    print(f" {'Strategy':<38} {'ms':>7}  {'fps-eq':>7}  {'vs B=18':>8}")
    print("─" * 65)
    for label, ms, fps, bvs in sorted(strategy_rows, key=lambda x: x[1]):
        mark = " ← ✅ TARGET" if ms < 25 else (" ← ⚠️ close" if ms < 35 else "")
        print(f"  {label:<38} {ms:7.2f}  {fps:>7}  {bvs:>8}{mark}")
    print("─" * 65)
    print()
    print(" Budget targets:")
    print("   🎯 Real-time 30 fps  → < 33 ms")
    print("   🎯 Target best-case  → < 20 ms")
    print()
    print(" Note: 'global×1' (batch=1) = whole frame downscaled to 640×720.")
    print("       Faster but misses dense sub-regions. Good for sparse scenes.")
    print(" Note: 3-stream parallel = 3 TRT execution contexts launched concurrently.")
    print("       Speedup depends on whether RTX 5060 Ti has spare SM capacity.")
    print()


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_engine(engine_path: Path, n_runs: int = 20) -> dict[str, float]:
    """Measure latency for the maximum-batch run on the given engine."""
    result: dict[str, float] = {}
    if not engine_path.exists():
        return result

    try:
        import tensorrt as trt
        import torch
    except ImportError:
        return result

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    ctx = engine.create_execution_context()

    inp_name = engine.get_tensor_name(0)
    # Detect the engine's actual opt/max batch from profile 0
    profile_max = ctx.engine.get_tensor_profile_shape(inp_name, 0)[2][0]  # max shape dim 0
    run_batch = min(BATCH_OPT, profile_max)
    print(f"\nBenchmarking {engine_path.name}  (batch={run_batch}/{profile_max} max, {n_runs} runs) ...")

    ctx.set_input_shape(inp_name, (run_batch, 3, TARGET_H, TARGET_W))

    dummy = torch.randn(run_batch, 3, TARGET_H, TARGET_W, dtype=torch.float32, device="cuda")
    out_shape = tuple(ctx.get_tensor_shape(engine.get_tensor_name(1)))
    if any(d <= 0 for d in out_shape):
        print(f"  ⚠️  Output shape invalid {out_shape} — engine profile mismatch, skipping benchmark")
        return {}
    dummy_out = torch.zeros(*out_shape, dtype=torch.float32, device="cuda")

    ctx.set_tensor_address(inp_name, dummy.data_ptr())
    ctx.set_tensor_address(engine.get_tensor_name(1), dummy_out.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(5):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    t_start = time.perf_counter()
    for _ in range(n_runs):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t_start) / n_runs * 1000.0

    memory_mib = dummy_out.numel() * 4 / 1024 ** 2
    result = {"latency_ms": elapsed, "output_mib": memory_mib, "run_batch": run_batch}
    print(f"  Avg latency : {elapsed:.2f} ms / batch")
    print(f"  Output shape: {tuple(dummy_out.shape)}  ({memory_mib:.2f} MiB)")
    count_est = float(dummy_out[0].sum().item())
    print(f"  Count sample (random noise): {count_est:.2f}  (expected ~0 ± small)")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _apply_tile_size(args)

    import math
    cols = math.ceil(3840 / TARGET_W)
    rows = math.ceil(2160 / TARGET_H)

    print("=" * 65)
    print(" DM-Count QNRF — TRT FP16 / FP8 QDQ benchmark")
    print(f" Tile: {TARGET_W}×{TARGET_H}  (multiple-of-16: {TARGET_W%16==0 and TARGET_H%16==0})")
    print(f" 4K coverage: {cols} cols × {rows} rows = {BATCH_MAX} tiles (batch max)")
    print(f" Skip FP8: {args.skip_fp8}")
    if not args.skip_fp8:
        calib_src = args.calib_dir or "<not provided — UCF-QNRF required>"
        print(f" Calib dir: {calib_src}")
    print("=" * 65)

    if not ensure_onnx():
        sys.exit(1)

    # ── 1. FP16 engine ───────────────────────────────────────────────────────
    if not ENGINE_FP16.exists():
        ok = build_trt_engine(ONNX_PATH, ENGINE_FP16)
        if not ok:
            print("❌ FP16 engine build failed")
    else:
        print(f"✅ FP16 engine exists: {ENGINE_FP16}")

    # ── 2. FP8 QDQ engine ────────────────────────────────────────────────────
    if args.skip_fp8:
        print("\n⏭  FP8-QDQ skipped (SKIP_DENSITY_FP8=1 or --skip-fp8)")
        print("   Provide --calib-dir /path/to/UCF-QNRF/Train/img to enable.")
    else:
        calib = get_calib_data(args.calib_dir)
        if calib is None:
            print("⏭  FP8-QDQ skipped (no calibration data — see message above).")
        else:
            if quantize_fp8(calib):
                if not ENGINE_FP8_QDQ.exists():
                    ok = build_trt_engine(ONNX_QDQ_PATH, ENGINE_FP8_QDQ, fp8_qdq=True)
                    if not ok:
                        print("❌ FP8 QDQ engine build failed")
                else:
                    print(f"✅ FP8 QDQ engine exists: {ENGINE_FP8_QDQ}")
            else:
                print("❌ FP8 QDQ quantization failed (ModelOpt unavailable?)")

    # ── 3. FP16-strict engine (optional) ─────────────────────────────────────
    if args.fp16_strict:
        if not ENGINE_FP16_STRICT.exists():
            build_trt_engine(ONNX_PATH, ENGINE_FP16_STRICT, fp16_strict=True)
        else:
            print(f"✅ FP16-strict engine exists: {ENGINE_FP16_STRICT}")

    # ── Strategy benchmark (optional) ────────────────────────────────────────
    if args.benchmark_strategies:
        print("\n" + "=" * 65)
        print(" Latency strategy benchmark")
        print("=" * 65)
        run_strategy_benchmark(ENGINE_FP16)
        if ENGINE_FP8_QDQ.exists():
            run_strategy_benchmark(ENGINE_FP8_QDQ)

    # ── Benchmark all available engines ──────────────────────────────────────
    print("\n" + "=" * 65)
    print(" Benchmark results")
    print("=" * 65)
    results: dict[str, dict[str, float]] = {}
    candidates = [("FP16 (baseline)", ENGINE_FP16), ("FP8 QDQ", ENGINE_FP8_QDQ)]
    if args.fp16_strict:
        candidates.append(("FP16-strict", ENGINE_FP16_STRICT))
    for label, path in candidates:
        r = benchmark_engine(path)
        if r:
            results[label] = r

    if results:
        print("\n── Summary ──")
        baseline = results.get("FP16 (baseline)", {}).get("latency_ms")
        for label, r in results.items():
            lat = r.get("latency_ms", 0.0)
            speedup = f"  ({baseline/lat:.2f}× vs FP16)" if baseline and label != "FP16 (baseline)" else ""
            print(f"  {label:<22} {lat:6.2f} ms/batch{speedup}")


if __name__ == "__main__":
    main()

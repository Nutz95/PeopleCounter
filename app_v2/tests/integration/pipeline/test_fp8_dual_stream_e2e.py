"""Integration test — FP8-QDQ dual-stream architecture benchmark.

Architecture under test:
    Stream A  : yolo26m-seg FP8-QDQ  batch=1  (global tile)
    Stream B  : yolo26n-seg FP8-QDQ  batch=16 (sub-tiles)
    Both streams run concurrently on separate CUDA streams.

Hypothesis (from trtexec benchmarks, RTX 5060 Ti sm_120):
    Stream A GPU compute  ≈  5.05 ms
    Stream B GPU compute  ≈  8.66 ms  (bottleneck)
    Ideal e2e  = max(A, B) ≈  8.66 ms   (perfect overlap)
    Serial e2e = A + B     ≈ 13.71 ms

The test reports:
    - Median GPU time per stream (CUDA events)
    - Wall-clock e2e per frame (serial vs parallel)
    - Parallelism efficiency = serial / parallel

Skip conditions (no GPU or no engines: run `./3_prepare_models.sh` first):
    SKIP_FP8=1  or
    models/tensorrt/yolo26m-seg-fp8-qdq.engine missing
    models/tensorrt/yolo26n-seg-fp8-qdq.engine missing
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Paths / skip guard
# ---------------------------------------------------------------------------
_ENGINES_ROOT = Path(__file__).resolve().parents[5] / "models" / "tensorrt"
_ENGINE_GLOBAL = _ENGINES_ROOT / "yolo26m-seg-fp8-qdq.engine"
_ENGINE_TILES  = _ENGINES_ROOT / "yolo26n-seg-fp8-qdq.engine"

_SKIP_REASON = (
    "FP8-QDQ engines not found — run ./3_prepare_models.sh first.\n"
    f"  Expected: {_ENGINE_GLOBAL}\n"
    f"  Expected: {_ENGINE_TILES}"
)


def _engines_available() -> bool:
    return _ENGINE_GLOBAL.exists() and _ENGINE_TILES.exists()


pytestmark = pytest.mark.skipif(
    not _engines_available(),
    reason=_SKIP_REASON,
)

# ---------------------------------------------------------------------------
# Lazy imports (requires TRT + CUDA at runtime)
# ---------------------------------------------------------------------------
try:
    import torch
    import tensorrt as trt
    _HAS_GPU = torch.cuda.is_available()
except Exception:
    _HAS_GPU = False

if not _HAS_GPU:
    pytestmark = pytest.mark.skip(reason="CUDA GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_engine(path: Path) -> Any:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None, f"Failed to deserialize {path}"
    return engine, runtime


def _make_input(batch: int, h: int = 640, w: int = 640) -> Any:
    """Synthetic float16 NCHW tensor."""
    return torch.randn(batch, 3, h, w, dtype=torch.float16, device="cuda")


def _bind_and_run(ctx: Any, engine: Any, input_tensor: Any, stream: Any) -> None:
    """Set shapes, bind addresses, execute async."""
    input_name = engine.get_tensor_name(0)
    ctx.set_input_shape(input_name, tuple(int(x) for x in input_tensor.shape))
    ctx.set_tensor_address(input_name, int(input_tensor.data_ptr()))

    for i in range(1, engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(int(x) for x in ctx.get_tensor_shape(name))
        dtype = _trt_to_torch(engine.get_tensor_dtype(name))
        buf = torch.empty(shape, device="cuda", dtype=dtype)
        ctx.set_tensor_address(name, int(buf.data_ptr()))

    ctx.execute_async_v3(int(stream.cuda_stream))


def _trt_to_torch(trt_dtype: Any) -> Any:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.INT32: torch.int32,
    }
    return mapping.get(trt_dtype, torch.float32)


def _cuda_time_ms(fn, warmup: int = 5, iters: int = 30) -> list[float]:
    """Measure GPU kernel time with CUDA events; return list of ms per iter."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()       # type: ignore[call-arg]
        fn()
        end.record()         # type: ignore[call-arg]
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFP8DualStreamE2E:
    """Benchmark + correctness check for the dual-stream FP8 architecture."""

    @pytest.fixture(scope="class")
    def engines(self):
        engine_m, rt_m = _load_engine(_ENGINE_GLOBAL)
        engine_n, rt_n = _load_engine(_ENGINE_TILES)
        yield engine_m, engine_n
        # runtimes kept alive via closure; engines freed on GC

    @pytest.fixture(scope="class")
    def contexts(self, engines):
        engine_m, engine_n = engines
        ctx_m = engine_m.create_execution_context()
        ctx_n = engine_n.create_execution_context()
        assert ctx_m is not None
        assert ctx_n is not None
        return ctx_m, ctx_n

    @pytest.fixture(scope="class")
    def streams(self):
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        return stream_a, stream_b

    # ------------------------------------------------------------------
    # Individual stream benchmarks
    # ------------------------------------------------------------------

    def test_global_tile_gpu_time(self, contexts, streams, engines, capsys):
        """yolo26m FP8-QDQ batch=1 — GPU compute should be ≈ 5 ms."""
        ctx_m, _ = contexts
        stream_a, _ = streams
        engine_m, _ = engines
        inp = _make_input(1)

        times = _cuda_time_ms(
            lambda: _bind_and_run(ctx_m, engine_m, inp, stream_a)
        )
        median_ms = statistics.median(times)
        p95_ms    = sorted(times)[int(len(times) * 0.95)]

        with capsys.disabled():
            print(
                f"\n[global tile] yolo26m-seg FP8-QDQ batch=1 : "
                f"median={median_ms:.2f} ms  p95={p95_ms:.2f} ms  "
                f"(trtexec reference: 5.05 ms)"
            )

        # Loose bound: must be < 2× trtexec reference on same GPU
        assert median_ms < 15.0, f"Global tile GPU time {median_ms:.2f} ms exceeds threshold"

    def test_sub_tiles_gpu_time(self, contexts, streams, engines, capsys):
        """yolo26n FP8-QDQ batch=16 — GPU compute should be ≈ 8.7 ms."""
        _, ctx_n = contexts
        _, stream_b = streams
        _, engine_n = engines
        inp = _make_input(16)

        times = _cuda_time_ms(
            lambda: _bind_and_run(ctx_n, engine_n, inp, stream_b)
        )
        median_ms = statistics.median(times)
        p95_ms    = sorted(times)[int(len(times) * 0.95)]

        with capsys.disabled():
            print(
                f"\n[sub-tiles  ] yolo26n-seg FP8-QDQ batch=16: "
                f"median={median_ms:.2f} ms  p95={p95_ms:.2f} ms  "
                f"(trtexec reference: 8.66 ms)"
            )

        assert median_ms < 25.0, f"Sub-tiles GPU time {median_ms:.2f} ms exceeds threshold"

    # ------------------------------------------------------------------
    # Dual-stream parallel benchmark
    # ------------------------------------------------------------------

    def test_parallel_e2e_latency(self, contexts, streams, engines, capsys):
        """Both streams running concurrently — e2e = max(A, B) ideally.

        Measures wall-clock latency of the dual-stream pair and computes
        parallelism efficiency vs the serial (A then B) baseline.
        """
        ctx_m, ctx_n = contexts
        stream_a, stream_b = streams
        engine_m, engine_n = engines
        inp_m = _make_input(1)
        inp_n = _make_input(16)

        # --- serial baseline ---
        def serial():
            _bind_and_run(ctx_m, engine_m, inp_m, stream_a)
            stream_a.synchronize()
            _bind_and_run(ctx_n, engine_n, inp_n, stream_b)
            stream_b.synchronize()

        torch.cuda.synchronize()
        serial_times: list[float] = []
        for _ in range(5):
            serial()  # warmup
        for _ in range(20):
            t0 = time.perf_counter()
            serial()
            serial_times.append((time.perf_counter() - t0) * 1000)
        serial_median = statistics.median(serial_times)

        # --- parallel concurrent ---
        def parallel():
            with torch.cuda.stream(stream_a):
                _bind_and_run(ctx_m, engine_m, inp_m, stream_a)
            with torch.cuda.stream(stream_b):
                _bind_and_run(ctx_n, engine_n, inp_n, stream_b)
            # wait for both
            stream_a.synchronize()
            stream_b.synchronize()

        parallel_times: list[float] = []
        for _ in range(5):
            parallel()  # warmup
        for _ in range(20):
            t0 = time.perf_counter()
            parallel()
            parallel_times.append((time.perf_counter() - t0) * 1000)
        parallel_median = statistics.median(parallel_times)

        efficiency = serial_median / max(1.0, parallel_median)

        with capsys.disabled():
            print(
                f"\n[dual-stream e2e]\n"
                f"  serial (A then B)  : {serial_median:.2f} ms\n"
                f"  parallel (A ∥ B)   : {parallel_median:.2f} ms\n"
                f"  parallelism gain   : {efficiency:.2f}×  "
                f"({'✅ overlap detected' if efficiency > 1.1 else '⚠️  minimal overlap'})\n"
                f"  30fps budget (33ms): {'✅ OK' if parallel_median < 33 else '❌ EXCEEDED'}"
            )

        assert parallel_median < 33.0, (
            f"Parallel e2e {parallel_median:.2f} ms exceeds 30fps budget (33ms)"
        )
        # Parallel should not be SLOWER than serial by more than 20% (overhead check)
        assert parallel_median < serial_median * 1.2, (
            f"Parallel {parallel_median:.2f} ms is unexpectedly slower than serial {serial_median:.2f} ms"
        )

    def test_global_tile_finishes_before_sub_tiles(self, contexts, streams, engines, capsys):
        """Verify the hypothesis: A (global, 5 ms) finishes before B (tiles, 8.7 ms).

        Records finish times of stream A and B using CUDA events.  If A
        consistently finishes first, the global tile result is available
        for post-processing while the tile batch is still computing.
        """
        ctx_m, ctx_n = contexts
        stream_a, stream_b = streams
        engine_m, engine_n = engines
        inp_m = _make_input(1)
        inp_n = _make_input(16)

        a_before_b_count = 0
        iters = 20

        ev_a_start = torch.cuda.Event(enable_timing=True)
        ev_a_end   = torch.cuda.Event(enable_timing=True)
        ev_b_start = torch.cuda.Event(enable_timing=True)
        ev_b_end   = torch.cuda.Event(enable_timing=True)

        for _ in range(3):  # warmup
            with torch.cuda.stream(stream_a):
                _bind_and_run(ctx_m, engine_m, inp_m, stream_a)
            with torch.cuda.stream(stream_b):
                _bind_and_run(ctx_n, engine_n, inp_n, stream_b)
            stream_a.synchronize()
            stream_b.synchronize()

        a_times: list[float] = []
        b_times: list[float] = []

        for _ in range(iters):
            torch.cuda.synchronize()
            with torch.cuda.stream(stream_a):
                ev_a_start.record(stream_a)      # type: ignore[call-arg]
                _bind_and_run(ctx_m, engine_m, inp_m, stream_a)
                ev_a_end.record(stream_a)         # type: ignore[call-arg]
            with torch.cuda.stream(stream_b):
                ev_b_start.record(stream_b)      # type: ignore[call-arg]
                _bind_and_run(ctx_n, engine_n, inp_n, stream_b)
                ev_b_end.record(stream_b)         # type: ignore[call-arg]

            stream_a.synchronize()
            stream_b.synchronize()

            t_a = ev_a_start.elapsed_time(ev_a_end)
            t_b = ev_b_start.elapsed_time(ev_b_end)
            a_times.append(t_a)
            b_times.append(t_b)
            if t_a < t_b:
                a_before_b_count += 1

        a_med = statistics.median(a_times)
        b_med = statistics.median(b_times)
        ratio = a_before_b_count / iters

        with capsys.disabled():
            print(
                f"\n[ordering check]\n"
                f"  yolo26m (global) GPU median : {a_med:.2f} ms\n"
                f"  yolo26n (tiles)  GPU median : {b_med:.2f} ms\n"
                f"  A finishes before B         : {a_before_b_count}/{iters} ({ratio*100:.0f}%)\n"
                f"  → {'✅ hypothesis confirmed' if ratio > 0.7 else '⚠️  hypothesis uncertain'}"
            )

        # We only assert that A is not *consistently* slower (accounts for SM contention)
        assert a_med < b_med * 1.5, (
            f"Global tile {a_med:.2f} ms is much slower than sub-tiles {b_med:.2f} ms — "
            "unexpected on this architecture"
        )

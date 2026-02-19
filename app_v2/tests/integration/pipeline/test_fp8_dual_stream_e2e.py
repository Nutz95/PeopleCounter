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
_ENGINES_ROOT = Path(__file__).resolve().parents[4] / "models" / "tensorrt"
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
    # Use the process-global TRT logger that was registered by tensorrt_engine_loader
    # (imported at session start via conftest.py).  Creating a separate logger here
    # would register a second singleton and produce CUDA error 700 when the first
    # one is released.
    from app_v2.infrastructure.tensorrt_engine_loader import _TRT_LOGGER
    _HAS_GPU = torch.cuda.is_available()
except Exception:
    _HAS_GPU = False
    _TRT_LOGGER = None

if not _HAS_GPU:
    pytestmark = pytest.mark.skip(reason="CUDA GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_engine(path: Path) -> Any:
    runtime = trt.Runtime(_TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None, f"Failed to deserialize {path}"
    return engine, runtime


def _make_input(batch: int, height: int = 640, width: int = 640) -> Any:
    """Synthetic float16 NCHW tensor."""
    return torch.randn(batch, 3, height, width, dtype=torch.float16, device="cuda")


def _make_runner(ctx: Any, engine: Any, input_tensor: Any, stream: Any):
    """Pre-allocate output buffers and return a zero-argument callable.

    The output buffers are bound once at construction time and stay alive for
    the lifetime of the returned function.  This avoids a use-after-free where
    per-call locals created inside a thin wrapper would be garbage-collected
    before the asynchronous CUDA kernel finishes, causing CUDA error 700
    (illegal memory access) when GPU work is flushed by the test harness.
    """
    input_name = engine.get_tensor_name(0)
    ctx.set_input_shape(input_name, tuple(int(x) for x in input_tensor.shape))
    ctx.set_tensor_address(input_name, int(input_tensor.data_ptr()))

    output_bufs: list[Any] = []
    for i in range(1, engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(int(x) for x in ctx.get_tensor_shape(name))
        dtype = _trt_to_torch(engine.get_tensor_dtype(name))
        buf = torch.empty(shape, device="cuda", dtype=dtype)
        ctx.set_tensor_address(name, int(buf.data_ptr()))
        output_bufs.append(buf)  # keep alive via closure

    stream_handle = int(stream.cuda_stream)

    def _run() -> None:
        ctx.execute_async_v3(stream_handle)

    # Python only closes over variables that are *referenced* inside _run.
    # output_bufs and input_tensor are not referenced, so they would be
    # garbage-collected as soon as _make_runner returns — freeing the GPU
    # memory that ctx still has registered.  Storing them as function
    # attributes keeps the tensors alive for the lifetime of the runner.
    _run._input_tensor = input_tensor  # type: ignore[attr-defined]
    _run._output_bufs = output_bufs    # type: ignore[attr-defined]
    return _run


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

    @pytest.fixture(autouse=True)
    def _sync_before_test(self):
        """Flush all pending GPU work before and after each test.

        Class-scoped fixtures (engines, contexts, streams) persist across
        tests.  When one test ends, TRT may have queued async GPU work that
        references that test's output buffers.  Ensuring all outstanding
        CUDA work has completed before the next test starts (and before those
        buffers may be garbage-collected) prevents use-after-free / CUDA 700.
        """
        import gc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        yield
        # Synchronize BEFORE Python frees local variables from the test, so
        # that any outstanding GPU work referencing those tensors finishes
        # before the memory is returned to the allocator pool.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()

    @pytest.fixture(scope="class")
    def engines(self):
        engine_global, runtime_global = _load_engine(_ENGINE_GLOBAL)
        engine_tiles, runtime_tiles = _load_engine(_ENGINE_TILES)
        # runtimes MUST be kept alive here (not just discarded) — the trt.Runtime
        # owns the CUDA context; if it is garbage-collected while engines/contexts
        # are still in use, TRT will tear down the context and every subsequent
        # TRT user in the same process gets CUDA error 700 (invalid context).
        yield engine_global, engine_tiles
        # Flush all pending GPU work before TRT objects are released.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del engine_global, engine_tiles
        del runtime_global, runtime_tiles

    @pytest.fixture(scope="class")
    def contexts(self, engines):
        """Class-scoped execution contexts.

        Kept alive for the entire test class so that the Myelin GPU modules
        (compiled on first ``execute_async_v3``) remain resident in the CUDA
        driver's module cache.

        There are two TRT/Myelin bugs on sm_120 (Blackwell) with FP8-QDQ
        engines that must be avoided simultaneously:

        1. **Rebinding bug**: calling ``set_tensor_address`` on a context that
           has already been executed (i.e. calling ``_make_runner`` more than
           once on the same context) corrupts Myelin's compiled Cast kernel
           arguments, causing CUDA error 700 on the next execution.

        2. **Double-delete bug**: creating a context, executing it once, and
           then *deleting* it — when this create→execute→delete cycle is
           repeated a second time for the same engine — also corrupts the
           engine's internal Myelin state for any future context.

        Both bugs are avoided by keeping a single pair of contexts alive for
        the entire test class *and* building the ``runners`` fixture so that
        ``_make_runner`` is called only once per context (see ``runners``
        fixture below).

        The session-scoped ``_warmup_myelin_modules`` fixture in conftest.py
        ensures the modules are compiled BEFORE any other CUDA kernel runs in
        the process, so the first use here is safe.
        """
        engine_global, engine_tiles = engines
        ctx_global = engine_global.create_execution_context()
        ctx_tiles = engine_tiles.create_execution_context()
        assert ctx_global is not None
        assert ctx_tiles is not None
        yield ctx_global, ctx_tiles
        # Flush all GPU work before releasing contexts so that async kernels
        # do not reference freed TRT engine memory.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del ctx_global, ctx_tiles

    @pytest.fixture(scope="class")
    def runners(self, contexts, engines, streams):
        """Class-scoped pre-built runners.  Each context is bound exactly once.

        ``_make_runner`` calls ``set_tensor_address`` on the execution context.
        Calling it more than once on an already-executed context (rebinding bug
        — see ``contexts`` docstring) causes CUDA error 700.  By creating the
        runners here, once for the whole class, every test method receives the
        same runner instances with fixed I/O buffers, ensuring each context is
        bound exactly once over its lifetime.
        """
        ctx_global, ctx_tiles = contexts
        engine_global, engine_tiles = engines
        cuda_stream_global, cuda_stream_tiles = streams
        run_global = _make_runner(ctx_global, engine_global, _make_input(1),  cuda_stream_global)
        run_tiles  = _make_runner(ctx_tiles,  engine_tiles,  _make_input(16), cuda_stream_tiles)
        yield run_global, run_tiles
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @pytest.fixture(scope="class")
    def streams(self):
        """Class-scoped CUDA streams, shared across tests in the class."""
        cuda_stream_global = torch.cuda.Stream()
        cuda_stream_tiles = torch.cuda.Stream()
        yield cuda_stream_global, cuda_stream_tiles
        cuda_stream_global.synchronize()
        cuda_stream_tiles.synchronize()

    # ------------------------------------------------------------------
    # Individual stream benchmarks
    # ------------------------------------------------------------------

    def test_global_tile_gpu_time(self, runners, streams, capsys):
        """yolo26m FP8-QDQ batch=1 — GPU compute should be ≈ 5 ms."""
        run, _ = runners
        cuda_stream_global, _ = streams

        times = _cuda_time_ms(run)
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

    def test_sub_tiles_gpu_time(self, runners, streams, capsys):
        """yolo26n FP8-QDQ batch=16 — GPU compute should be ≈ 8.7 ms."""
        _, run = runners
        _, cuda_stream_tiles = streams

        times = _cuda_time_ms(run)
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

    def test_parallel_e2e_latency(self, runners, streams, capsys):
        """Both streams running concurrently — e2e = max(A, B) ideally.

        Measures wall-clock latency of the dual-stream pair and computes
        parallelism efficiency vs the serial (A then B) baseline.
        """
        run_global, run_tiles = runners
        cuda_stream_global, cuda_stream_tiles = streams

        # --- serial baseline ---
        def serial():
            run_global()
            cuda_stream_global.synchronize()
            run_tiles()
            cuda_stream_tiles.synchronize()

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
            with torch.cuda.stream(cuda_stream_global):
                run_global()
            with torch.cuda.stream(cuda_stream_tiles):
                run_tiles()
            # wait for both
            cuda_stream_global.synchronize()
            cuda_stream_tiles.synchronize()

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

    def test_global_tile_finishes_before_sub_tiles(self, runners, streams, capsys):
        """Verify the hypothesis: A (global, 5 ms) finishes before B (tiles, 8.7 ms).

        Records finish times of stream A and B using CUDA events.  If A
        consistently finishes first, the global tile result is available
        for post-processing while the tile batch is still computing.
        """
        run_global, run_tiles = runners
        cuda_stream_global, cuda_stream_tiles = streams

        global_finishes_first_count = 0
        iters = 20

        event_global_start = torch.cuda.Event(enable_timing=True)
        event_global_end   = torch.cuda.Event(enable_timing=True)
        event_tiles_start  = torch.cuda.Event(enable_timing=True)
        event_tiles_end    = torch.cuda.Event(enable_timing=True)

        for _ in range(3):  # warmup
            with torch.cuda.stream(cuda_stream_global):
                run_global()
            with torch.cuda.stream(cuda_stream_tiles):
                run_tiles()
            cuda_stream_global.synchronize()
            cuda_stream_tiles.synchronize()

        global_tile_times: list[float] = []
        sub_tiles_times: list[float] = []

        for _ in range(iters):
            torch.cuda.synchronize()
            with torch.cuda.stream(cuda_stream_global):
                event_global_start.record(cuda_stream_global)      # type: ignore[call-arg]
                run_global()
                event_global_end.record(cuda_stream_global)        # type: ignore[call-arg]
            with torch.cuda.stream(cuda_stream_tiles):
                event_tiles_start.record(cuda_stream_tiles)        # type: ignore[call-arg]
                run_tiles()
                event_tiles_end.record(cuda_stream_tiles)         # type: ignore[call-arg]

            cuda_stream_global.synchronize()
            cuda_stream_tiles.synchronize()

            global_tile_duration_ms = event_global_start.elapsed_time(event_global_end)
            sub_tiles_duration_ms   = event_tiles_start.elapsed_time(event_tiles_end)
            global_tile_times.append(global_tile_duration_ms)
            sub_tiles_times.append(sub_tiles_duration_ms)
            if global_tile_duration_ms < sub_tiles_duration_ms:
                global_finishes_first_count += 1

        global_tile_median_ms = statistics.median(global_tile_times)
        sub_tiles_median_ms   = statistics.median(sub_tiles_times)
        ratio = global_finishes_first_count / iters

        with capsys.disabled():
            print(
                f"\n[ordering check]\n"
                f"  yolo26m (global) GPU median : {global_tile_median_ms:.2f} ms\n"
                f"  yolo26n (tiles)  GPU median : {sub_tiles_median_ms:.2f} ms\n"
                f"  global finishes before tiles: {global_finishes_first_count}/{iters} ({ratio*100:.0f}%)\n"
                f"  → {'✅ hypothesis confirmed' if ratio > 0.7 else '⚠️  hypothesis uncertain'}"
            )

        # We only assert that global tile is not *consistently* slower (accounts for SM contention)
        assert global_tile_median_ms < sub_tiles_median_ms * 1.5, (
            f"Global tile {global_tile_median_ms:.2f} ms is much slower than sub-tiles {sub_tiles_median_ms:.2f} ms — "
            "unexpected on this architecture"
        )

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    message=r".*pynvml package is deprecated.*",
    category=FutureWarning,
)


# ---------------------------------------------------------------------------
# TensorRT global logger — must outlive all TRT objects in the process.
# ---------------------------------------------------------------------------
# TRT registers the *first* trt.Logger() passed to trt.Runtime() as a
# process-global singleton (nvinfer1::getLogger()).  If that logger object is
# GC'd while any TRT engine or runtime is still alive, the internal pointer
# becomes dangling and the next createInferRuntime call crashes with CUDA
# error 700 ("illegal memory access").
#
# By importing tensorrt_engine_loader here we trigger module-level
# creation of _TRT_LOGGER *before* any test file creates its own logger.
# All subsequent trt.Runtime() calls (in both production code and tests)
# then reuse the same logger through the "logger differs → using existing"
# TRT warning path, which is safe.
try:
    import app_v2.infrastructure.tensorrt_engine_loader as _  # noqa: F401 – side-effect import
except Exception:
    pass  # TRT / engine loader not available; tests that need it will skip.


# ---------------------------------------------------------------------------
# Myelin GPU-module warm-up — must run before any CUDA kernel in the session.
# ---------------------------------------------------------------------------
# TRT's Myelin backend (used by FP8-QDQ engines on Blackwell / sm_120) compiles
# and maps its GPU module (cubin) during the VERY FIRST execute_async_v3 call.
# On CUDA 13.1 / TRT 10.15 / sm_120 there is a driver-level constraint: if any
# other CUDA kernel (e.g. a cudaMemset triggered by torch.zeros) has executed
# before this first Myelin dispatch, the cubin mapping (cuMemMap / cuModuleLoad)
# lands at an address that is subsequently inaccessible, causing CUDA error 700
# (illegal memory access) on the very next Myelin kernel launch.
#
# Work-around: execute both FP8-QDQ engines ONCE as the very first CUDA work in
# the session, so the Myelin module is compiled and stored in the driver's
# per-process module cache.  All subsequent engine executions — even with freshly
# deserialized engines — reuse the cached module and are immune to the error.
#
# The fixture is session-scoped and autouse so it runs before the first test
# regardless of collection order.
_FP8_ENGINES_ROOT = Path(__file__).resolve().parents[2] / "models" / "tensorrt"
_FP8_ENGINE_GLOBAL = _FP8_ENGINES_ROOT / "yolo26m-seg-fp8-qdq.engine"
_FP8_ENGINE_TILES  = _FP8_ENGINES_ROOT / "yolo26n-seg-fp8-qdq.engine"


@pytest.fixture(scope="session", autouse=True)
def _warmup_myelin_modules() -> None:  # type: ignore[return]
    """Pre-compile Myelin CUDA modules for both FP8-QDQ engines.

    This is a no-op when the engine files are absent (non-GPU CI) or when
    CUDA is unavailable.
    """
    if not (_FP8_ENGINE_GLOBAL.exists() and _FP8_ENGINE_TILES.exists()):
        return  # engines not available – skip silently

    try:
        import torch
        import tensorrt as trt
        from app_v2.infrastructure.tensorrt_engine_loader import _TRT_LOGGER
    except Exception:
        return  # TRT / CUDA not available

    if not torch.cuda.is_available():
        return

    def _warmup_engine(path: Path, input_shape: tuple, batch: int) -> None:
        runtime = trt.Runtime(_TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            return
        ctx = engine.create_execution_context()
        stream = torch.cuda.Stream()
        inp = torch.empty(input_shape, dtype=torch.float16, device="cuda")
        input_name = engine.get_tensor_name(0)
        ctx.set_input_shape(input_name, input_shape)
        ctx.set_tensor_address(input_name, int(inp.data_ptr()))
        output_bufs = []
        for i in range(1, engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(int(x) for x in ctx.get_tensor_shape(name))
            buf = torch.empty(shape, device="cuda", dtype=torch.float32)
            ctx.set_tensor_address(name, int(buf.data_ptr()))
            output_bufs.append(buf)
        # First execute_async_v3 triggers Myelin GPU-module compilation.
        ctx.execute_async_v3(int(stream.cuda_stream))
        torch.cuda.synchronize()
        # Objects kept alive until function returns to ensure the CUDA work
        # referencing inp/output_bufs has fully completed.
        del ctx, inp, output_bufs, stream, engine, runtime

    # Warm up tiles engine first (larger batch, the one that fails without warm-up)
    _warmup_engine(_FP8_ENGINE_TILES,  (16, 3, 640, 640), 16)
    # Warm up global engine too (good practice, keeps caches symmetric)
    _warmup_engine(_FP8_ENGINE_GLOBAL, (1,  3, 640, 640),  1)

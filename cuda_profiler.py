"""Helpers for enabling CUDA profiler API hooks without crashing when CUDA is unavailable."""
import os
import time
import traceback

try:
    from cuda.bindings import runtime as cudart
except Exception:  # pragma: no cover - best-effort import
    cudart = None


def _format_cuda_error(action: str, code: int) -> str:
    message = f"[CUDA PROFILER] {action} returned {code}"
    if not cudart:
        return message
    try:
        desc = cudart.cudaGetErrorString(code)
        if isinstance(desc, bytes):
            desc = desc.decode("utf-8", "ignore")
        message += f" ({desc})" if desc else ""
    except Exception:
        pass
    return message


def cuda_profiler_supported() -> bool:
    """Return True when the CUDA runtime provides profiler start/stop symbols."""
    return cudart is not None


def cuda_profiler_enabled() -> bool:
    """Respect the environment flag that gates profiling globally."""
    return cuda_profiler_supported() and os.environ.get("ENABLE_CUDA_PROFILING", "0") == "1"


def cuda_profiler_start() -> bool:
    """Call the CUDA profiler start API and return whether it succeeded."""
    if not cuda_profiler_supported():
        return False
    try:
        raw = cudart.cudaProfilerStart()
        result = raw[0] if isinstance(raw, tuple) else raw
        result = int(result)
        if result != 0:
            print(_format_cuda_error("cudaProfilerStart", result))
            return False
        return True
    except Exception as exc:
        print("[CUDA PROFILER] cudaProfilerStart threw an exception:", exc)
        traceback.print_exc()
        return False


def cuda_profiler_stop() -> bool:
    """Call the CUDA profiler stop API and return whether it succeeded."""
    if not cuda_profiler_supported():
        return False
    try:
        raw = cudart.cudaProfilerStop()
        result = raw[0] if isinstance(raw, tuple) else raw
        result = int(result)
        if result != 0:
            print(_format_cuda_error("cudaProfilerStop", result))
            return False
        return True
    except Exception as exc:
        print("[CUDA PROFILER] cudaProfilerStop threw an exception:", exc)
        traceback.print_exc()
        return False

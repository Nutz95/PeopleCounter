from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict


if TYPE_CHECKING:
    from app_v2.infrastructure.tensorrt_engine_loader import TensorRTEngineLoader


class TensorRTExecutionContext:
    """Wraps TensorRT execution contexts and their associated CUDA streams."""

    def __init__(self, engine_loader: "TensorRTEngineLoader", stream_pool: Any) -> None:
        self.engine_loader = engine_loader
        self.stream_pool = stream_pool
        self.bound_tensors: Dict[str, int] = {}

    def bind_stream(self, stream_key: str) -> None:
        """Bind the context to the CUDA stream mapped to stream_key."""
        stream_handle = self.stream_pool.acquire(stream_key)
        self.bound_tensors[stream_key] = stream_handle

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the TensorRT inference and return raw outputs."""
        return {"status": "stub", "inputs": inputs}

    def release_stream(self, stream_key: str) -> None:
        """Return the stream handle to the pool."""
        self.stream_pool.release(stream_key)
        self.bound_tensors.pop(stream_key, None)

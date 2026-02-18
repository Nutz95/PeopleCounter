from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import tensorrt as trt
except Exception:  # pragma: no cover
    trt = None  # type: ignore[assignment]


@dataclass
class _GraphEntry:
    """One captured CUDA graph entry for a specific batch size."""
    graph: Any                              # torch.cuda.CUDAGraph
    input_buf: Any                          # pre-allocated input tensor (CUDA, FP16)
    output_bufs: Dict[str, Any]            # pre-allocated output tensors (CUDA)
    graph_exec: Any = None                 # not used with torch API (graph.replay())
    warmup_count: int = 0


class CudaGraphCache:
    """Per-batch-size CUDA graph cache for TRT execution contexts.

    Workflow per batch size:
      1. First call  → warmup N times → capture graph with pre-allocated buffers
      2. Next calls  → memcpy input → replay graph → read pre-allocated outputs
    
    Constraints:
      - Input/output tensor shapes must be fixed for a given batch_size.
      - Falls back gracefully to direct execute_async_v3 if capture fails.
      - Thread-safe only when each batch_size is accessed from a single thread
        (which matches YoloTilingParallelTRT group isolation).
    """

    WARMUP_ITERS = 3

    def __init__(
        self,
        trt_context: Any,
        stream: Any,
        engine: Any,
        warmup_iters: int = WARMUP_ITERS,
    ) -> None:
        self._ctx = trt_context          # trt.IExecutionContext
        self._stream = stream            # torch.cuda.Stream
        self._engine = engine            # trt.ICudaEngine
        self._warmup_iters = warmup_iters
        self._entries: Dict[int, _GraphEntry] = {}
        self._capture_failed: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_captured(self, batch_size: int) -> bool:
        return batch_size in self._entries

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of capture state for logging/monitoring.

        Example::

            {
                "captured_batches": [8, 16],
                "failed_batches": [],
                "using_graph": True,
            }
        """
        captured = sorted(self._entries.keys())
        failed = sorted(self._capture_failed)
        return {
            "captured_batches": captured,
            "failed_batches": failed,
            "using_graph": len(captured) > 0,
        }

    def execute(
        self,
        input_tensor: Any,
        output_names: List[str],
        input_name: str,
    ) -> Dict[str, Any]:
        """Run inference via captured graph (or direct TRT if not yet captured).

        Returns dict of {output_name: tensor}.
        """
        if torch is None or trt is None:
            return {}

        batch_size = int(input_tensor.shape[0])

        if batch_size in self._capture_failed:
            return self._direct_execute(input_tensor, output_names, input_name)

        if batch_size not in self._entries:
            self._capture_graph(input_tensor, output_names, input_name, batch_size)
            if batch_size not in self._entries:
                return self._direct_execute(input_tensor, output_names, input_name)

        return self._replay_graph(input_tensor, batch_size)

    # ------------------------------------------------------------------
    # Internal: graph capture
    # ------------------------------------------------------------------

    def _capture_graph(
        self,
        sample_input: Any,
        output_names: List[str],
        input_name: str,
        batch_size: int,
    ) -> None:
        try:
            # 1. Allocate permanent buffers
            input_buf = torch.empty_like(sample_input)
            input_buf.copy_(sample_input)

            output_bufs: Dict[str, Any] = {}
            self._ctx.set_input_shape(input_name, tuple(int(x) for x in input_buf.shape))
            self._ctx.set_tensor_address(input_name, int(input_buf.data_ptr()))

            for name in output_names:
                shape = tuple(int(x) for x in self._ctx.get_tensor_shape(name))
                dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
                buf = torch.empty(shape, device="cuda", dtype=dtype)
                self._ctx.set_tensor_address(name, int(buf.data_ptr()))
                output_bufs[name] = buf

            stream_handle = int(self._stream.cuda_stream)

            # 2. Warmup (outside graph capture, allows cuDNN/TRT to initialize)
            for _ in range(self._warmup_iters):
                self._ctx.execute_async_v3(stream_handle)
            torch.cuda.synchronize()

            # 3. Capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=self._stream):
                self._ctx.execute_async_v3(stream_handle)

            self._entries[batch_size] = _GraphEntry(
                graph=g,
                input_buf=input_buf,
                output_bufs=output_bufs,
            )
        except Exception as exc:
            # Any failure → mark as non-capturable and fall back to direct execution
            print(f"[CudaGraphCache] Capture failed for batch={batch_size}: {exc}")
            self._capture_failed.add(batch_size)

    # ------------------------------------------------------------------
    # Internal: replay
    # ------------------------------------------------------------------

    def _replay_graph(self, input_tensor: Any, batch_size: int) -> Dict[str, Any]:
        entry = self._entries[batch_size]
        # Copy new input data into the pre-allocated buffer (same address = graph replay works)
        entry.input_buf.copy_(input_tensor, non_blocking=True)
        entry.graph.replay()
        # Return references to the pre-allocated output buffers
        # IMPORTANT: caller must NOT hold these across the next replay
        return {name: buf for name, buf in entry.output_bufs.items()}

    def _direct_execute(
        self,
        input_tensor: Any,
        output_names: List[str],
        input_name: str,
    ) -> Dict[str, Any]:
        """Fallback: direct TRT execution without graph."""
        stream_handle = int(self._stream.cuda_stream)
        self._ctx.set_input_shape(input_name, tuple(int(x) for x in input_tensor.shape))
        self._ctx.set_tensor_address(input_name, int(input_tensor.data_ptr()))
        outputs: Dict[str, Any] = {}
        for name in output_names:
            shape = tuple(int(x) for x in self._ctx.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
            buf = torch.empty(shape, device="cuda", dtype=dtype)
            self._ctx.set_tensor_address(name, int(buf.data_ptr()))
            outputs[name] = buf
        self._ctx.execute_async_v3(stream_handle)
        return outputs

    def release(self) -> None:
        """Free all captured graphs and buffers."""
        self._entries.clear()
        self._capture_failed.clear()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _trt_dtype_to_torch(trt_dtype: Any) -> Any:
    if trt is None or torch is None:
        return None
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
    }
    return mapping.get(trt_dtype, torch.float32)

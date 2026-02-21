from __future__ import annotations

from contextlib import nullcontext
import os
import time
from typing import TYPE_CHECKING, Any, Dict

try:
    import tensorrt as trt
except Exception:  # pragma: no cover
    trt = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from app_v2.infrastructure.tensorrt_engine_loader import TensorRTEngineLoader
    from app_v2.infrastructure.cuda_graph_cache import CudaGraphCache


class TensorRTExecutionContext:
    """Wraps TensorRT execution contexts and their associated CUDA streams."""

    def __init__(self, engine_loader: "TensorRTEngineLoader", stream_pool: Any, options: Dict[str, Any] | None = None) -> None:
        self.engine_loader = engine_loader
        self.stream_pool = stream_pool
        self.options = dict(options or {})
        self.bound_tensors: Dict[str, int] = {}
        self._bound_streams: Dict[str, Any] = {}
        self._metadata = self.engine_loader.load()
        self._context: Any | None = None
        env_enabled = os.environ.get("PEOPLE_COUNTER_ENABLE_TRT_EXECUTION")
        cfg_enabled = bool(self.options.get("enabled", False))
        self._trt_enabled = (env_enabled.strip() == "1") if isinstance(env_enabled, str) else cfg_enabled
        self._strict_shape_check = bool(self.options.get("strict_shape_check", True))
        self._cuda_graphs_enabled = bool(self.options.get("cuda_graphs", False))
        self._cuda_graph_cache: "CudaGraphCache | None" = None
        engine = self.engine_loader.engine
        if self._trt_enabled and engine is not None:
            self._context = engine.create_execution_context()

    def bind_stream(self, stream_key: str) -> None:
        """Bind the context to the CUDA stream mapped to stream_key."""
        stream_handle = self.stream_pool.acquire(stream_key)
        self._bound_streams[stream_key] = stream_handle
        if hasattr(stream_handle, "cuda_stream"):
            self.bound_tensors[stream_key] = int(stream_handle.cuda_stream)
        else:
            self.bound_tensors[stream_key] = 0

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run TensorRT inference and return raw outputs.

        The heavy lifting is split across focused private helpers:
        1. :meth:`_check_preconditions` — guard against disabled / missing TRT
        2. :meth:`_prepare_batch` — GPU tensor batch assembly + timing
        3. :meth:`_bind_io_tensors` — shape/address binding for input + outputs
        4. :meth:`_enqueue_and_sync` — async kernel launch + stream sync
        """
        precondition_error = self._check_preconditions(inputs)
        if precondition_error is not None:
            return precondition_error

        stream = next(iter(self._bound_streams.values()), None)
        stream_ctx = torch.cuda.stream(stream) if hasattr(stream, "cuda_stream") else nullcontext()
        with stream_ctx:
            # Establish GPU-side ordering: wait for all preprocess streams to
            # finish before touching their output tensors.  This is a non-blocking
            # GPU-side dependency (no CPU stall) — the host returns immediately.
            for event in inputs.get("preprocess_events", []):
                if hasattr(stream, "wait_event"):
                    stream.wait_event(event)
            prepare_batch_result = self._prepare_batch(inputs)
            if "error" in prepare_batch_result:
                return {"status": "gpu_unavailable", "inputs": inputs, "reason": prepare_batch_result["error"]}

            input_tensor: Any = prepare_batch_result["input_tensor"]
            prepare_batch_ms: float = prepare_batch_result["prepare_batch_ms"]

            bind_result = self._bind_io_tensors(inputs, input_tensor)
            if "error" in bind_result:
                return {"status": "gpu_unavailable", "inputs": inputs, "reason": bind_result["error"]}

            input_name: str = bind_result["input_name"]
            output_names: list[str] = bind_result["output_names"]
            outputs: dict[str, Any] = bind_result["outputs"]
            output_tensors: list[Any] = bind_result["output_tensors"]

            if self._cuda_graphs_enabled and hasattr(stream, "cuda_stream"):
                graph_result = self._execute_via_graph(
                    input_tensor=input_tensor,
                    input_name=input_name,
                    output_names=output_names,
                    stream=stream,
                )
                if graph_result is not None:
                    graph_result["prepare_batch_ms"] = prepare_batch_ms
                    return graph_result

            enqueue_result = self._enqueue_and_sync(stream)
            if "error" in enqueue_result:
                return {"status": "gpu_unavailable", "inputs": inputs, "reason": enqueue_result["error"]}

            return {
                "status": "ok",
                "outputs": outputs,
                "output_tensors": output_tensors,
                "input_shape": tuple(int(x) for x in input_tensor.shape),
                "engine_path": self._metadata.get("path"),
                "prepare_batch_ms": prepare_batch_ms,
                "enqueue_ms": enqueue_result["enqueue_ms"],
                "stream_sync_ms": enqueue_result["stream_sync_ms"],
            }

    # ------------------------------------------------------------------
    # execute() sub-steps
    # ------------------------------------------------------------------

    def _check_preconditions(self, inputs: Dict[str, Any]) -> Dict[str, Any] | None:
        """Return an error dict if TRT execution cannot proceed, else None."""
        if not self._trt_enabled:
            return {"status": "gpu_unavailable", "inputs": inputs, "reason": "trt execution disabled by config"}
        if self._context is None or trt is None or torch is None or not torch.cuda.is_available():
            return {"status": "gpu_unavailable", "inputs": inputs, "reason": self._metadata.get("reason", "no context")}
        return None

    def _prepare_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble input tensors into a contiguous GPU batch.

        Returns a dict with ``input_tensor`` + ``prepare_batch_ms`` on success,
        or ``error`` (str) on failure.
        """
        prepare_start_ns = self._perf_counter_ns()
        prepared_inputs = self._prepare_input_tensors(inputs)
        prepare_batch_ms = (self._perf_counter_ns() - prepare_start_ns) / 1_000_000.0
        if not prepared_inputs:
            return {"error": "no input tensors"}
        return {"input_tensor": prepared_inputs[0], "prepare_batch_ms": prepare_batch_ms}

    def _bind_io_tensors(
        self, inputs: Dict[str, Any], input_tensor: Any
    ) -> Dict[str, Any]:
        """Validate shapes and bind I/O tensor addresses on the TRT context.

        Returns a dict with ``input_name``, ``output_names``, ``outputs``,
        ``output_tensors`` on success, or ``error`` (str) on failure.
        """
        input_names = self._tensor_names(mode=trt.TensorIOMode.INPUT)
        output_names = self._tensor_names(mode=trt.TensorIOMode.OUTPUT)
        if not input_names or not output_names:
            return {"error": "engine missing io tensors"}

        input_name = input_names[0]
        if self._strict_shape_check and not self._is_input_shape_compatible(input_name, input_tensor):
            return {
                "error": f"input shape incompatible for {input_name}: "
                         f"{tuple(int(x) for x in input_tensor.shape)}"
            }

        self._context.set_input_shape(input_name, tuple(int(x) for x in input_tensor.shape))
        self._context.set_tensor_address(input_name, int(input_tensor.data_ptr()))

        outputs: dict[str, Any] = {}
        output_tensors: list[Any] = []
        for output_name in output_names:
            output_shape = tuple(int(x) for x in self._context.get_tensor_shape(output_name))
            if not output_shape or any(dim <= 0 for dim in output_shape):
                return {"error": f"unresolved output shape for {output_name}: {output_shape}"}
            dtype = self._torch_dtype(self.engine_loader.engine.get_tensor_dtype(output_name))
            output_tensor = torch.empty(output_shape, device="cuda", dtype=dtype)
            self._context.set_tensor_address(output_name, int(output_tensor.data_ptr()))
            outputs[output_name] = output_tensor
            output_tensors.append(output_tensor)

        return {
            "input_name": input_name,
            "output_names": output_names,
            "outputs": outputs,
            "output_tensors": output_tensors,
        }

    def _enqueue_and_sync(self, stream: Any) -> Dict[str, Any]:
        """Launch ``execute_async_v3`` and synchronise the stream.

        Returns a dict with ``enqueue_ms`` + ``stream_sync_ms`` on success,
        or ``error`` (str) on failure.
        """
        stream_handle = int(getattr(stream, "cuda_stream", 0))
        enqueue_start_ns = self._perf_counter_ns()
        ok = bool(self._context.execute_async_v3(stream_handle))
        enqueue_ms = (self._perf_counter_ns() - enqueue_start_ns) / 1_000_000.0
        if not ok:
            return {"error": "execute_async_v3 failed"}
        sync_start_ns = self._perf_counter_ns()
        if hasattr(stream, "synchronize"):
            stream.synchronize()
        stream_sync_ms = (self._perf_counter_ns() - sync_start_ns) / 1_000_000.0
        return {"enqueue_ms": enqueue_ms, "stream_sync_ms": stream_sync_ms}

    def _execute_via_graph(
        self,
        input_tensor: Any,
        input_name: str,
        output_names: list[str],
        stream: Any,
    ) -> Dict[str, Any] | None:
        """Try graph-replay path. Returns result dict on success, None to fall back."""
        from app_v2.infrastructure.cuda_graph_cache import CudaGraphCache

        if self._cuda_graph_cache is None:
            self._cuda_graph_cache = CudaGraphCache(
                trt_context=self._context,
                stream=stream,
                engine=self.engine_loader.engine,
            )

        enqueue_start_ns = self._perf_counter_ns()
        outputs = self._cuda_graph_cache.execute(input_tensor, output_names, input_name)
        enqueue_ms = (self._perf_counter_ns() - enqueue_start_ns) / 1_000_000.0

        if not outputs:
            return None

        sync_start_ns = self._perf_counter_ns()
        if hasattr(stream, "synchronize"):
            stream.synchronize()
        stream_sync_ms = (self._perf_counter_ns() - sync_start_ns) / 1_000_000.0

        return {
            "status": "ok",
            "outputs": outputs,
            "output_tensors": list(outputs.values()),
            "input_shape": tuple(int(x) for x in input_tensor.shape),
            "engine_path": self._metadata.get("path"),
            "prepare_batch_ms": 0.0,  # filled by caller
            "enqueue_ms": float(enqueue_ms),
            "stream_sync_ms": float(stream_sync_ms),
            "cuda_graph": True,
        }

    def _is_input_shape_compatible(self, input_name: str, tensor: Any) -> bool:
        engine = self.engine_loader.engine
        if engine is None:
            return False
        try:
            profile_shapes = engine.get_tensor_profile_shape(input_name, 0)
        except Exception:
            return True
        if not isinstance(profile_shapes, tuple) or len(profile_shapes) != 3:
            return True
        min_shape, _opt_shape, max_shape = profile_shapes
        if not min_shape or not max_shape:
            return True
        dims = tuple(int(x) for x in tensor.shape)
        if len(dims) != len(min_shape):
            return False
        for value, lo, hi in zip(dims, min_shape, max_shape):
            low = int(lo)
            high = int(hi)
            if low > 0 and value < low:
                return False
            if high > 0 and value > high:
                return False
        return True

    def _prepare_input_tensors(self, payload: Dict[str, Any]) -> list[Any]:
        raw_inputs = payload.get("inputs", [])
        prepared: list[Any] = []
        if not isinstance(raw_inputs, list):
            return prepared

        # Determine the dtype the engine expects for its first input tensor.
        engine_input_dtype = torch.float32  # safe default (all current engines use fp32)
        if trt is not None and self.engine_loader.engine is not None:
            input_names = self._tensor_names(mode=trt.TensorIOMode.INPUT)
            if input_names:
                trt_dtype = self.engine_loader.engine.get_tensor_dtype(input_names[0])
                engine_input_dtype = self._torch_dtype(trt_dtype) or torch.float32

        for item in raw_inputs:
            tensor = getattr(item, "tensor_ref", None)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != "cuda":
                continue
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.dtype != engine_input_dtype:
                tensor = tensor.to(dtype=engine_input_dtype)
            prepared.append(tensor.contiguous())
        if not prepared:
            return prepared
        if len(prepared) == 1:
            return [prepared[0]]
        batch = torch.cat(prepared, dim=0)
        return [batch]

    def _tensor_names(self, mode: Any) -> list[str]:
        engine = self.engine_loader.engine
        if engine is None:
            return []
        names: list[str] = []
        for index in range(int(engine.num_io_tensors)):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == mode:
                names.append(name)
        return names

    @staticmethod
    def _torch_dtype(trt_dtype: Any) -> Any:
        if trt is None or torch is None:
            return None
        if trt_dtype == trt.DataType.FLOAT:
            return torch.float32
        if trt_dtype == trt.DataType.HALF:
            return torch.float16
        if trt_dtype == trt.DataType.INT8:
            return torch.int8
        if trt_dtype == trt.DataType.INT32:
            return torch.int32
        return torch.float32

    def release_graphs(self) -> None:
        """Free all captured CUDA graphs (call before engine unload)."""
        if self._cuda_graph_cache is not None:
            self._cuda_graph_cache.release()
            self._cuda_graph_cache = None

    def release_stream(self, stream_key: str) -> None:
        """Return the stream handle to the pool."""
        self.stream_pool.release(stream_key)
        self.bound_tensors.pop(stream_key, None)
        self._bound_streams.pop(stream_key, None)

    @staticmethod
    def _perf_counter_ns() -> int:
        """Return nanoseconds timestamp for high-precision profiling."""
        return time.perf_counter_ns()

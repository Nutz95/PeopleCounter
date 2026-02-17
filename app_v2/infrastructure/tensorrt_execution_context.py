from __future__ import annotations

from contextlib import nullcontext
import os
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


class TensorRTExecutionContext:
    """Wraps TensorRT execution contexts and their associated CUDA streams."""

    def __init__(self, engine_loader: "TensorRTEngineLoader", stream_pool: Any) -> None:
        self.engine_loader = engine_loader
        self.stream_pool = stream_pool
        self.bound_tensors: Dict[str, int] = {}
        self._bound_streams: Dict[str, Any] = {}
        self._metadata = self.engine_loader.load()
        self._context: Any | None = None
        self._trt_enabled = os.environ.get("PEOPLE_COUNTER_ENABLE_TRT_EXECUTION", "0").strip() == "1"
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
        """Run the TensorRT inference and return raw outputs."""
        if not self._trt_enabled:
            return {"status": "fallback", "inputs": inputs, "reason": "trt execution disabled (set PEOPLE_COUNTER_ENABLE_TRT_EXECUTION=1)"}
        if self._context is None or trt is None or torch is None or not torch.cuda.is_available():
            return {"status": "fallback", "inputs": inputs, "reason": self._metadata.get("reason", "no context")}

        stream = next(iter(self._bound_streams.values()), None)
        stream_ctx = torch.cuda.stream(stream) if hasattr(stream, "cuda_stream") else nullcontext()
        with stream_ctx:
            prepared_inputs = self._prepare_input_tensors(inputs)
            if not prepared_inputs:
                return {"status": "fallback", "inputs": inputs, "reason": "no input tensors"}

            input_names = [name for name in self._tensor_names(mode=trt.TensorIOMode.INPUT)]
            output_names = [name for name in self._tensor_names(mode=trt.TensorIOMode.OUTPUT)]
            if not input_names or not output_names:
                return {"status": "fallback", "inputs": inputs, "reason": "engine missing io tensors"}

            input_tensor = prepared_inputs[0]
            input_name = input_names[0]
            self._context.set_input_shape(input_name, tuple(int(x) for x in input_tensor.shape))
            self._context.set_tensor_address(input_name, int(input_tensor.data_ptr()))

            outputs: dict[str, Any] = {}
            output_tensors: list[Any] = []
            for output_name in output_names:
                shape = tuple(int(x) for x in self._context.get_tensor_shape(output_name))
                if not shape or any(dim <= 0 for dim in shape):
                    return {
                        "status": "fallback",
                        "inputs": inputs,
                        "reason": f"unresolved output shape for {output_name}: {shape}",
                    }
                safe_shape = tuple(shape)
                dtype = self._torch_dtype(self.engine_loader.engine.get_tensor_dtype(output_name))
                output_tensor = torch.empty(safe_shape, device="cuda", dtype=dtype)
                self._context.set_tensor_address(output_name, int(output_tensor.data_ptr()))
                outputs[output_name] = output_tensor
                output_tensors.append(output_tensor)

            stream_handle = int(getattr(stream, "cuda_stream", 0))
            ok = bool(self._context.execute_async_v3(stream_handle))
            if not ok:
                return {"status": "fallback", "inputs": inputs, "reason": "execute_async_v3 failed"}
            if hasattr(stream, "synchronize"):
                stream.synchronize()

            return {
                "status": "ok",
                "outputs": outputs,
                "output_tensors": output_tensors,
                "input_shape": tuple(int(x) for x in input_tensor.shape),
                "engine_path": self._metadata.get("path"),
            }

    def _prepare_input_tensors(self, payload: Dict[str, Any]) -> list[Any]:
        raw_inputs = payload.get("inputs", [])
        prepared: list[Any] = []
        if not isinstance(raw_inputs, list):
            return prepared
        for item in raw_inputs:
            tensor = getattr(item, "tensor_ref", None)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type != "cuda":
                continue
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.dtype != torch.float16:
                tensor = tensor.to(dtype=torch.float16)
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

    def release_stream(self, stream_key: str) -> None:
        """Return the stream handle to the pool."""
        self.stream_pool.release(stream_key)
        self.bound_tensors.pop(stream_key, None)
        self._bound_streams.pop(stream_key, None)

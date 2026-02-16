from __future__ import annotations

from typing import Any

from app_v2.core.preprocessor_types import GpuTensor, PreprocessTask

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _allocate_buffer(requested_bytes: int) -> int | None:
    if requested_bytes <= 0:
        raise ValueError("requested_bytes must be positive")
    return None


def _allocate_gpu_tensor(task: PreprocessTask) -> tuple[int | None, Any | None]:
    if torch is not None and torch.cuda.is_available():
        tensor = torch.zeros((task.target_height, task.target_width, 3), dtype=torch.uint8, device="cuda")
        return int(tensor.data_ptr()), tensor
    return None, None


def _build_tensor(task: PreprocessTask, buffer_ptr: int | None, buffer_size: int, tensor_ref: Any | None) -> GpuTensor:
    return GpuTensor(
        model_name=task.model_name,
        task_index=task.task_index,
        width=task.target_width,
        height=task.target_height,
        buffer_size=buffer_size,
        device_ptr=int(buffer_ptr) if buffer_ptr is not None else None,
        tensor_ref=tensor_ref,
    )


def run_letterbox_kernel(frame: Any, task: PreprocessTask, stream: int = 0) -> GpuTensor:
    """Allocate a GPU tensor for letterbox output."""
    del stream
    buffer_size = task.target_width * task.target_height * 3
    device_ptr, tensor_ref = _allocate_gpu_tensor(task)
    if device_ptr is None:
        device_ptr = getattr(frame, "device_ptr_y", None)
    return _build_tensor(task, device_ptr, buffer_size, tensor_ref)


def run_tiling_kernel(frame: Any, task: PreprocessTask, stream: int = 0) -> GpuTensor:
    """Allocate a GPU tensor for tiling output."""
    del stream
    buffer_size = task.target_width * task.target_height * 3
    device_ptr, tensor_ref = _allocate_gpu_tensor(task)
    if device_ptr is None:
        device_ptr = getattr(frame, "device_ptr_y", None)
    return _build_tensor(task, device_ptr, buffer_size, tensor_ref)

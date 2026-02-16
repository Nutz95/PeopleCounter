from __future__ import annotations

from typing import Any

from app_v2.core.preprocessor_types import GpuTensor, PreprocessTask
from app_v2.infrastructure.gpu_tensor_pool import GpuTensorLease, GpuTensorPool

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


_DEFAULT_POOL = GpuTensorPool()


def _require_cuda() -> None:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("GPU-only preprocess requires CUDA-enabled PyTorch")


def _source_tensor_from_frame(frame: Any) -> Any:
    _require_cuda()
    candidate = getattr(frame, "tensor_ref", None) or getattr(frame, "rgb_tensor", None)
    if candidate is not None and torch is not None and isinstance(candidate, torch.Tensor):
        if candidate.device.type != "cuda":
            raise RuntimeError("Preprocess source tensor must be on CUDA")
        if candidate.ndim == 3 and candidate.shape[-1] == 3:
            return candidate
        if candidate.ndim == 3 and candidate.shape[0] == 3:
            return candidate.permute(1, 2, 0).contiguous()
    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    return torch.zeros((height, width, 3), dtype=torch.uint8, device="cuda")


def _acquire_output_tensor(task: PreprocessTask, stream: int, pool: GpuTensorPool | None) -> GpuTensorLease:
    active_pool = pool or _DEFAULT_POOL
    assert torch is not None
    return active_pool.acquire(
        shape=(3, task.target_height, task.target_width),
        dtype=torch.float16,
        memory_format="RGB_NCHW_FP16",
        stream=stream,
        device="cuda",
        timeout_s=1.0,
    )


def _as_nchw_fp32(image_hwc: Any) -> Any:
    assert torch is not None
    if image_hwc.dtype == torch.uint8:
        tensor = image_hwc.to(dtype=torch.float32) / 255.0
    else:
        tensor = image_hwc.to(dtype=torch.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


def _to_target_fp16(resized_nchw: Any, lease: GpuTensorLease) -> Any:
    out = lease.tensor
    out.zero_()
    out.copy_(resized_nchw.squeeze(0).to(dtype=out.dtype))
    return out


def _letterbox_resize(src_hwc: Any, target_h: int, target_w: int) -> Any:
    assert torch is not None
    assert F is not None
    src_h = int(src_hwc.shape[0])
    src_w = int(src_hwc.shape[1])
    scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
    resize_w = max(1, int(round(src_w * scale)))
    resize_h = max(1, int(round(src_h * scale)))

    src_nchw = _as_nchw_fp32(src_hwc)
    resized = F.interpolate(src_nchw, size=(resize_h, resize_w), mode="bilinear", align_corners=False)

    canvas = torch.zeros((1, 3, target_h, target_w), dtype=torch.float32, device=src_hwc.device)
    off_y = (target_h - resize_h) // 2
    off_x = (target_w - resize_w) // 2
    canvas[:, :, off_y : off_y + resize_h, off_x : off_x + resize_w] = resized
    return canvas


def _tile_resize(src_hwc: Any, task: PreprocessTask) -> Any:
    assert F is not None
    x0 = max(0, int(task.source_x))
    y0 = max(0, int(task.source_y))
    x1 = min(int(src_hwc.shape[1]), x0 + max(1, int(task.source_width)))
    y1 = min(int(src_hwc.shape[0]), y0 + max(1, int(task.source_height)))
    crop = src_hwc[y0:y1, x0:x1, :]
    if crop.numel() == 0:
        crop = src_hwc[:1, :1, :]
    crop_nchw = _as_nchw_fp32(crop)
    return F.interpolate(crop_nchw, size=(task.target_height, task.target_width), mode="bilinear", align_corners=False)


def _build_tensor(task: PreprocessTask, tensor_ref: Any, lease_ref: GpuTensorLease) -> GpuTensor:
    buffer_ptr = int(tensor_ref.data_ptr())
    buffer_size = int(tensor_ref.element_size() * tensor_ref.nelement())
    return GpuTensor(
        model_name=task.model_name,
        task_index=task.task_index,
        width=task.target_width,
        height=task.target_height,
        buffer_size=buffer_size,
        device_ptr=buffer_ptr,
        memory_format="RGB_NCHW_FP16",
        tensor_ref=tensor_ref,
        lease_ref=lease_ref,
    )


def run_letterbox_kernel(frame: Any, task: PreprocessTask, stream: int = 0, pool: GpuTensorPool | None = None) -> GpuTensor:
    """Run a GPU letterbox transform into pooled RGB NCHW FP16 output."""
    src_hwc = _source_tensor_from_frame(frame)
    lease = _acquire_output_tensor(task, stream, pool)
    resized = _letterbox_resize(src_hwc, task.target_height, task.target_width)
    output = _to_target_fp16(resized, lease)
    return _build_tensor(task, output, lease)


def run_tiling_kernel(frame: Any, task: PreprocessTask, stream: int = 0, pool: GpuTensorPool | None = None) -> GpuTensor:
    """Run a GPU tiling transform into pooled RGB NCHW FP16 output."""
    src_hwc = _source_tensor_from_frame(frame)
    lease = _acquire_output_tensor(task, stream, pool)
    resized = _tile_resize(src_hwc, task)
    output = _to_target_fp16(resized, lease)
    return _build_tensor(task, output, lease)

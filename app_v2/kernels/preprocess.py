from __future__ import annotations

from typing import Any

from app_v2.core.preprocessor_types import GpuTensor, PreprocessTask, TensorMemoryFormat
from app_v2.infrastructure.gpu_tensor_pool import GpuTensorLease, GpuTensorPool
from app_v2.kernels.nv12_cuda_bridge import (
    nv12_crop_to_rgb_nchw_fp16,
    nv12_frame_to_rgb_hwc_cuda,
    nv12_tiles_to_rgb_nchw_fp16_batch,
    nv12_to_rgb_nchw_fp16_letterbox,
)

try:
    import torch
    import torch.nn.functional as torch_functional
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]


_DEFAULT_POOL = GpuTensorPool()


def _require_cuda() -> None:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("GPU-only preprocess requires CUDA-enabled PyTorch")


def _source_tensor_from_frame(frame: Any, source_tensor: Any | None = None) -> Any:
    _require_cuda()
    if source_tensor is not None:
        return source_tensor
    candidate = getattr(frame, "tensor_ref", None) or getattr(frame, "rgb_tensor", None)
    if candidate is not None and torch is not None and isinstance(candidate, torch.Tensor):
        if candidate.device.type != "cuda":
            raise RuntimeError("Preprocess source tensor must be on CUDA")
        if candidate.ndim == 3 and candidate.shape[-1] == 3:
            return candidate
        if candidate.ndim == 3 and candidate.shape[0] == 3:
            return candidate.permute(1, 2, 0).contiguous()

    has_nv12_pointers = getattr(frame, "device_ptr_y", None) is not None
    is_nv12 = bool(getattr(frame, "is_nv12", lambda: False)())
    if has_nv12_pointers or is_nv12:
        telemetry = getattr(frame, "telemetry", None)
        if telemetry:
            telemetry.mark_stage_start("preprocess_nv12_bridge")
        rgb_hwc = nv12_frame_to_rgb_hwc_cuda(frame)
        if telemetry:
            telemetry.mark_stage_end("preprocess_nv12_bridge")
        return rgb_hwc

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    return torch.zeros((height, width, 3), dtype=torch.uint8, device="cuda")


def resolve_frame_source_tensor(frame: Any) -> Any:
    """Resolve frame source tensor once so callers can reuse it across many tasks."""
    return _source_tensor_from_frame(frame)


def _acquire_output_tensor(task: PreprocessTask, stream: int, pool: GpuTensorPool | None) -> GpuTensorLease:
    active_pool = pool or _DEFAULT_POOL
    assert torch is not None
    return active_pool.acquire(
        shape=(3, task.target_height, task.target_width),
        dtype=torch.float16,
        memory_format=TensorMemoryFormat.RGB_NCHW_FP16,
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
    out.copy_(resized_nchw.squeeze(0).to(dtype=out.dtype))
    return out


def _letterbox_resize(src_hwc: Any, target_h: int, target_w: int) -> Any:
    assert torch is not None
    assert torch_functional is not None
    src_h = int(src_hwc.shape[0])
    src_w = int(src_hwc.shape[1])
    scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
    resize_w = max(1, int(round(src_w * scale)))
    resize_h = max(1, int(round(src_h * scale)))

    src_nchw = _as_nchw_fp32(src_hwc)
    resized = torch_functional.interpolate(src_nchw, size=(resize_h, resize_w), mode="bilinear", align_corners=False)

    canvas = torch.zeros((1, 3, target_h, target_w), dtype=torch.float32, device=src_hwc.device)
    off_y = (target_h - resize_h) // 2
    off_x = (target_w - resize_w) // 2
    canvas[:, :, off_y : off_y + resize_h, off_x : off_x + resize_w] = resized
    return canvas


def _tile_resize(src_hwc: Any, task: PreprocessTask) -> Any:
    assert torch_functional is not None
    x0 = max(0, int(task.source_x))
    y0 = max(0, int(task.source_y))
    x1 = min(int(src_hwc.shape[1]), x0 + max(1, int(task.source_width)))
    y1 = min(int(src_hwc.shape[0]), y0 + max(1, int(task.source_height)))
    crop = src_hwc[y0:y1, x0:x1, :]
    if crop.numel() == 0:
        crop = src_hwc[:1, :1, :]
    crop_nchw = _as_nchw_fp32(crop)
    return torch_functional.interpolate(
        crop_nchw,
        size=(task.target_height, task.target_width),
        mode="bilinear",
        align_corners=False,
    )


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
        memory_format=TensorMemoryFormat.RGB_NCHW_FP16,
        tensor_ref=tensor_ref,
        lease_ref=lease_ref,
    )


def run_letterbox_kernel(
    frame: Any,
    task: PreprocessTask,
    stream: int = 0,
    pool: GpuTensorPool | None = None,
    source_tensor: Any | None = None,
) -> GpuTensor:
    """Run a GPU letterbox transform into pooled RGB NCHW FP16 output."""
    src_hwc = _source_tensor_from_frame(frame, source_tensor=source_tensor)
    lease = _acquire_output_tensor(task, stream, pool)
    resized = _letterbox_resize(src_hwc, task.target_height, task.target_width)
    output = _to_target_fp16(resized, lease)
    return _build_tensor(task, output, lease)


def run_tiling_kernel(
    frame: Any,
    task: PreprocessTask,
    stream: int = 0,
    pool: GpuTensorPool | None = None,
    source_tensor: Any | None = None,
) -> GpuTensor:
    """Run a GPU tiling transform into pooled RGB NCHW FP16 output."""
    src_hwc = _source_tensor_from_frame(frame, source_tensor=source_tensor)
    lease = _acquire_output_tensor(task, stream, pool)
    resized = _tile_resize(src_hwc, task)
    output = _to_target_fp16(resized, lease)
    return _build_tensor(task, output, lease)


def run_tiling_kernel_fused(
    frame: Any,
    task: PreprocessTask,
    stream: int = 0,
    pool: GpuTensorPool | None = None,
    source_tensor: Any | None = None,
) -> GpuTensor:
    """Fused NV12 crop + resize kernel — no full-frame RGB intermediate.

    For NV12 frames (when ``source_tensor is None``) this copies only the
    ``task.source_height × task.source_width`` region from the Y and UV planes
    directly, avoiding the ``(frame_H × frame_W × 3)`` peak-memory allocation
    that the standard :func:`run_tiling_kernel` incurs.

    For non-NV12 frames (pre-decoded RGB tensors, or when *source_tensor* is
    already provided) the function falls back transparently to the standard
    two-step crop-and-resize path.
    """
    is_nv12 = (
        source_tensor is None
        and getattr(frame, "device_ptr_y", None) is not None
    )

    lease = _acquire_output_tensor(task, stream, pool)

    if is_nv12:
        telemetry = getattr(frame, "telemetry", None)
        if telemetry:
            telemetry.mark_stage_start("preprocess_nv12_crop_fused")

        rgb_nchw_fp16 = nv12_crop_to_rgb_nchw_fp16(
            frame,
            crop_x=int(task.source_x),
            crop_y=int(task.source_y),
            crop_w=int(task.source_width),
            crop_h=int(task.source_height),
            target_h=task.target_height,
            target_w=task.target_width,
        )

        if telemetry:
            telemetry.mark_stage_end("preprocess_nv12_crop_fused")

        out = lease.tensor
        out.copy_(rgb_nchw_fp16)
    else:
        src_hwc = _source_tensor_from_frame(frame, source_tensor=source_tensor)
        resized = _tile_resize(src_hwc, task)
        out = _to_target_fp16(resized, lease)

    return _build_tensor(task, out, lease)


def run_tiling_kernel_fused_batch(
    frame: Any,
    tasks: list[PreprocessTask],
    stream: int = 0,
    pool: GpuTensorPool | None = None,
) -> list[GpuTensor]:
    """Batch-fused NV12 tiling: one full-frame copy, single grid_sample for all tiles.

    All tasks must share the same ``target_height`` and ``target_width``.  When
    that condition is not met, or when the frame is not an NV12 source, the
    function falls back to calling :func:`run_tiling_kernel_fused` sequentially.

    Compared to calling :func:`run_tiling_kernel_fused` in a loop, for N=16 tiles
    this reduces Python→CUDA kernel dispatches from ~500 to ~25, cutting the
    Python-side dispatch overhead from ~40 ms to < 1 ms.
    """
    if not tasks:
        return []

    is_nv12 = getattr(frame, "device_ptr_y", None) is not None
    if not is_nv12:
        return [
            run_tiling_kernel_fused(frame, task, stream=stream, pool=pool)
            for task in tasks
        ]

    # All tiles must share the same target resolution for a single batched grid_sample.
    target_h = tasks[0].target_height
    target_w = tasks[0].target_width
    if not all(task.target_height == target_h and task.target_width == target_w for task in tasks[1:]):
        return [
            run_tiling_kernel_fused(frame, task, stream=stream, pool=pool)
            for task in tasks
        ]

    tile_crops = [
        (int(task.source_x), int(task.source_y), int(task.source_width), int(task.source_height))
        for task in tasks
    ]

    rgb_batch = nv12_tiles_to_rgb_nchw_fp16_batch(frame, tile_crops, target_h, target_w)
    # rgb_batch: [num_tiles, 3, target_h, target_w] fp16

    result: list[GpuTensor] = []
    for tile_index, task in enumerate(tasks):
        lease = _acquire_output_tensor(task, stream, pool)
        lease.tensor.copy_(rgb_batch[tile_index])  # [3, th, tw] slice → pool tensor
        result.append(_build_tensor(task, lease.tensor, lease))
    return result


def run_letterbox_kernel_fused(
    frame: Any,
    task: PreprocessTask,
    stream: int = 0,
    pool: GpuTensorPool | None = None,
    source_tensor: Any | None = None,
) -> GpuTensor:
    """Fused NV12 + letterbox kernel — skips the full-resolution RGB intermediate.

    For NV12 frames this is significantly faster than :func:`run_letterbox_kernel`
    because it samples the Y/UV planes *directly at output resolution* and folds
    the YUV→RGB conversion into the same pass, eliminating a temporary
    ``(H × W × 3)`` tensor allocation and the subsequent resize pass.

    For non-NV12 frames (pre-decoded RGB tensors) the function falls back
    transparently to the standard two-step letterbox path.
    """
    # Fused path only for NV12 frames without a pre-supplied source tensor
    is_nv12 = (
        source_tensor is None
        and getattr(frame, "device_ptr_y", None) is not None
        and (
            bool(getattr(frame, "is_nv12", lambda: False)())
            or getattr(frame, "device_ptr_y", None) is not None
        )
    )

    lease = _acquire_output_tensor(task, stream, pool)

    if is_nv12:
        telemetry = getattr(frame, "telemetry", None)
        if telemetry:
            telemetry.mark_stage_start("preprocess_nv12_fused")

        rgb_nchw_fp16 = nv12_to_rgb_nchw_fp16_letterbox(
            frame, task.target_height, task.target_width
        )

        if telemetry:
            telemetry.mark_stage_end("preprocess_nv12_fused")

        out = lease.tensor
        out.copy_(rgb_nchw_fp16)
    else:
        # Non-NV12: regular two-step
        src_hwc = _source_tensor_from_frame(frame, source_tensor=source_tensor)
        resized = _letterbox_resize(src_hwc, task.target_height, task.target_width)
        out = _to_target_fp16(resized, lease)

    return _build_tensor(task, out, lease)

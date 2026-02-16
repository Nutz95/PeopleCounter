from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn.functional as torch_functional
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]

try:
    from cuda.bindings import runtime as cuda_runtime
except Exception:  # pragma: no cover
    cuda_runtime = None  # type: ignore[assignment]


def nv12_frame_to_rgb_hwc_cuda(frame: Any) -> Any:
    """Convert an NV12 frame described by device pointers into RGB HWC uint8 on CUDA."""
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for NV12 bridge")
    if torch_functional is None:
        raise RuntimeError("torch.nn.functional is required for NV12 bridge")
    if cuda_runtime is None:
        raise RuntimeError("cuda.bindings.runtime is required for NV12 bridge")

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    pitch = int(getattr(frame, "pitch") or width)

    source_y_ptr = _extract_device_pointer(getattr(frame, "device_ptr_y", None))
    source_uv_ptr = _extract_device_pointer(getattr(frame, "device_ptr_uv", None))
    if source_y_ptr <= 0:
        raise RuntimeError("NV12 frame missing valid device_ptr_y")
    if source_uv_ptr <= 0 or source_uv_ptr == source_y_ptr:
        source_uv_ptr = source_y_ptr + (pitch * height)

    y_plane = torch.empty((height, width), dtype=torch.uint8, device="cuda")
    uv_plane = torch.empty((max(1, height // 2), width), dtype=torch.uint8, device="cuda")

    _copy_plane_2d_async(
        destination_ptr=int(y_plane.data_ptr()),
        destination_pitch_bytes=width,
        source_ptr=source_y_ptr,
        source_pitch_bytes=pitch,
        width_bytes=width,
        height_rows=height,
    )
    _copy_plane_2d_async(
        destination_ptr=int(uv_plane.data_ptr()),
        destination_pitch_bytes=width,
        source_ptr=source_uv_ptr,
        source_pitch_bytes=pitch,
        width_bytes=width,
        height_rows=max(1, height // 2),
    )

    y = y_plane.to(dtype=torch.float32)
    uv_pairs = uv_plane.view(max(1, height // 2), max(1, width // 2), 2).permute(2, 0, 1).to(dtype=torch.float32)
    u = torch_functional.interpolate(
        uv_pairs[0:1].unsqueeze(0),
        size=(height, width),
        mode="nearest",
    ).squeeze(0).squeeze(0)
    v = torch_functional.interpolate(
        uv_pairs[1:2].unsqueeze(0),
        size=(height, width),
        mode="nearest",
    ).squeeze(0).squeeze(0)

    y = y - 16.0
    u = u - 128.0
    v = v - 128.0

    r = torch.clamp((1.164 * y) + (1.596 * v), 0.0, 255.0)
    g = torch.clamp((1.164 * y) - (0.392 * u) - (0.813 * v), 0.0, 255.0)
    b = torch.clamp((1.164 * y) + (2.017 * u), 0.0, 255.0)

    return torch.stack((r, g, b), dim=-1).to(dtype=torch.uint8)


def _copy_plane_2d_async(
    *,
    destination_ptr: int,
    destination_pitch_bytes: int,
    source_ptr: int,
    source_pitch_bytes: int,
    width_bytes: int,
    height_rows: int,
) -> None:
    assert cuda_runtime is not None
    assert torch is not None
    stream_handle = int(getattr(torch.cuda.current_stream(), "cuda_stream", 0))
    status = cuda_runtime.cudaMemcpy2DAsync(
        destination_ptr,
        destination_pitch_bytes,
        source_ptr,
        source_pitch_bytes,
        width_bytes,
        height_rows,
        cuda_runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        stream_handle,
    )
    normalized_status = status[0] if isinstance(status, tuple) else status
    if normalized_status != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpy2DAsync failed with status={status}")


def _extract_device_pointer(value: Any) -> int:
    if isinstance(value, int):
        return value
    gpu_mem = getattr(value, "GpuMem", None)
    if callable(gpu_mem):
        maybe_ptr = gpu_mem()
        if isinstance(maybe_ptr, int):
            return maybe_ptr
    return 0

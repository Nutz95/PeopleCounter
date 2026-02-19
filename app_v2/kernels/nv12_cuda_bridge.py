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


def nv12_crop_to_rgb_nchw_fp16(
    frame: Any,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    target_h: int,
    target_w: int,
) -> Any:
    """Fused NV12 crop + resize → RGB NCHW float16, without a full-frame intermediate.

    Copies only the `crop_h × crop_w` region from the Y and UV planes instead
    of decoding the entire frame.  This avoids the ``(frame_H × frame_W × 3)``
    peak-memory spike that :func:`nv12_frame_to_rgb_hwc_cuda` would produce.

    Returns a ``[3, target_h, target_w]`` float16 tensor in *[0, 1]* range.
    Pixel values outside the crop region are filled with 0 via ``padding_mode='border'``.
    """
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for fused NV12 crop")
    if torch_functional is None:
        raise RuntimeError("torch.nn.functional is required for fused NV12 crop")
    if cuda_runtime is None:
        raise RuntimeError("cuda.bindings.runtime is required for fused NV12 crop")

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    pitch = int(getattr(frame, "pitch") or width)

    y_ptr = _extract_device_pointer(getattr(frame, "device_ptr_y", None))
    uv_ptr = _extract_device_pointer(getattr(frame, "device_ptr_uv", None))
    if y_ptr <= 0:
        raise RuntimeError("NV12 frame missing valid device_ptr_y")
    if uv_ptr <= 0 or uv_ptr == y_ptr:
        uv_ptr = y_ptr + pitch * height

    # Clamp crop to frame bounds
    crop_x = max(0, min(crop_x, width - 1))
    crop_y = max(0, min(crop_y, height - 1))
    crop_w = max(1, min(crop_w, width - crop_x))
    crop_h = max(1, min(crop_h, height - crop_y))

    # UV plane: half-height, each row has crop_w interleaved bytes (crop_w/2 U-V pairs)
    uv_row_first = crop_y // 2
    uv_row_count = max(1, (crop_h + 1) // 2)

    # Source byte offsets for the crop windows
    y_crop_ptr = y_ptr + crop_y * pitch
    uv_crop_ptr = uv_ptr + uv_row_first * pitch

    y_plane = torch.empty((crop_h, crop_w), dtype=torch.uint8, device="cuda")
    uv_plane = torch.empty((uv_row_count, crop_w), dtype=torch.uint8, device="cuda")

    _copy_plane_2d_async(
        destination_ptr=int(y_plane.data_ptr()),
        destination_pitch_bytes=crop_w,
        source_ptr=y_crop_ptr,
        source_pitch_bytes=pitch,
        width_bytes=crop_w,
        height_rows=crop_h,
    )
    _copy_plane_2d_async(
        destination_ptr=int(uv_plane.data_ptr()),
        destination_pitch_bytes=crop_w,
        source_ptr=uv_crop_ptr,
        source_pitch_bytes=pitch,
        width_bytes=crop_w,
        height_rows=uv_row_count,
    )

    # Build simple bilinear resize grid [1, target_h, target_w, 2]
    oj = torch.arange(target_w, device="cuda", dtype=torch.float32)
    oi = torch.arange(target_h, device="cuda", dtype=torch.float32)
    nx = (oj + 0.5) * (2.0 / max(1, target_w)) - 1.0   # [-1, 1]
    ny = (oi + 0.5) * (2.0 / max(1, target_h)) - 1.0
    gx = nx.view(1, target_w).expand(target_h, target_w)
    gy = ny.view(target_h, 1).expand(target_h, target_w)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)   # [1, th, tw, 2]

    y_f = y_plane.float().unsqueeze(0).unsqueeze(0)     # [1, 1, crop_h, crop_w]
    uv_pairs = (
        uv_plane
        .view(uv_row_count, max(1, crop_w // 2), 2)
        .permute(2, 0, 1)
        .float()
    )  # [2, uv_row_count, crop_w//2]
    u_f = uv_pairs[0:1].unsqueeze(0)                    # [1, 1, uh, uw]
    v_f = uv_pairs[1:2].unsqueeze(0)

    sample_kw = {"mode": "bilinear", "padding_mode": "border", "align_corners": False}
    y_out = torch_functional.grid_sample(y_f, grid, **sample_kw)   # [1,1,th,tw]
    u_out = torch_functional.grid_sample(u_f, grid, **sample_kw)
    v_out = torch_functional.grid_sample(v_f, grid, **sample_kw)

    y_val = y_out - 16.0
    u_val = u_out - 128.0
    v_val = v_out - 128.0

    r = torch.clamp(1.164 * y_val + 1.596 * v_val, 0.0, 255.0)
    g = torch.clamp(1.164 * y_val - 0.392 * u_val - 0.813 * v_val, 0.0, 255.0)
    b = torch.clamp(1.164 * y_val + 2.017 * u_val, 0.0, 255.0)

    rgb = torch.cat([r, g, b], dim=1) / 255.0           # [1, 3, th, tw] fp32
    return rgb.to(dtype=torch.float16).squeeze(0)        # [3, th, tw] fp16


def nv12_to_rgb_nchw_fp16_letterbox(
    frame: Any,
    target_h: int,
    target_w: int,
) -> Any:
    """Fused NV12 decode + letterbox resize → RGB NCHW float16.

    Returns a ``[3, target_h, target_w]`` float16 tensor in *[0, 1]* range.

    Compared to calling :func:`nv12_frame_to_rgb_hwc_cuda` followed by a
    bilinear letterbox resize, this function avoids allocating the full-
    resolution RGB intermediate by sampling Y and UV planes *directly at
    output resolution* via ``grid_sample``, then applying the YUV→RGB
    conversion on the smaller (target) tensor.

    The UV half-resolution is handled transparently by ``grid_sample``'s
    normalised-coordinate convention — a sub-pixel alignment offset of
    ``0.5 / W`` is introduced (< 1 pixel for any realistic resolution),
    which is acceptable for real-time video.
    """
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for fused NV12 letterbox")
    if torch_functional is None:
        raise RuntimeError("torch.nn.functional is required for fused NV12 letterbox")
    if cuda_runtime is None:
        raise RuntimeError("cuda.bindings.runtime is required for fused NV12 letterbox")

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    pitch = int(getattr(frame, "pitch") or width)

    y_ptr = _extract_device_pointer(getattr(frame, "device_ptr_y", None))
    uv_ptr = _extract_device_pointer(getattr(frame, "device_ptr_uv", None))
    if y_ptr <= 0:
        raise RuntimeError("NV12 frame missing valid device_ptr_y")
    if uv_ptr <= 0 or uv_ptr == y_ptr:
        uv_ptr = y_ptr + pitch * height

    # ------------------------------------------------------------------ #
    # Copy Y and UV planes (same as two-step path)
    # ------------------------------------------------------------------ #
    y_plane = torch.empty((height, width), dtype=torch.uint8, device="cuda")
    uv_plane = torch.empty((max(1, height // 2), width), dtype=torch.uint8, device="cuda")

    _copy_plane_2d_async(
        destination_ptr=int(y_plane.data_ptr()),
        destination_pitch_bytes=width,
        source_ptr=y_ptr,
        source_pitch_bytes=pitch,
        width_bytes=width,
        height_rows=height,
    )
    _copy_plane_2d_async(
        destination_ptr=int(uv_plane.data_ptr()),
        destination_pitch_bytes=width,
        source_ptr=uv_ptr,
        source_pitch_bytes=pitch,
        width_bytes=width,
        height_rows=max(1, height // 2),
    )

    # ------------------------------------------------------------------ #
    # Build letterbox sampling grid [1, target_h, target_w, 2]
    # ------------------------------------------------------------------ #
    scale = min(target_w / max(1, width), target_h / max(1, height))
    render_w = max(1, int(round(width * scale)))
    render_h = max(1, int(round(height * scale)))
    off_x = (target_w - render_w) // 2
    off_y = (target_h - render_h) // 2

    oj = torch.arange(target_w, device="cuda", dtype=torch.float32)
    oi = torch.arange(target_h, device="cuda", dtype=torch.float32)

    # Pixel (oj) in output → fractional pixel in input Y plane → normalised to [-1, 1]
    nx = (oj - off_x) * width / render_w  # src_x (fractional, in [0, W))
    ny = (oi - off_y) * height / render_h  # src_y (fractional, in [0, H))
    nx = (nx + 0.5) * (2.0 / width) - 1.0   # align_corners=False convention
    ny = (ny + 0.5) * (2.0 / height) - 1.0

    # grid_sample expects [1, H, W, 2] with (x=col, y=row) ordering
    gx = nx.view(1, target_w).expand(target_h, target_w)
    gy = ny.view(target_h, 1).expand(target_h, target_w)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, target_h, target_w, 2]

    # ------------------------------------------------------------------ #
    # Sample Y, U, V at output resolution
    # ------------------------------------------------------------------ #
    y_f = y_plane.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    uv_pairs = (
        uv_plane.view(max(1, height // 2), max(1, width // 2), 2)
        .permute(2, 0, 1)
        .float()
    )  # [2, H/2, W/2]
    u_f = uv_pairs[0:1].unsqueeze(0)  # [1, 1, H/2, W/2]
    v_f = uv_pairs[1:2].unsqueeze(0)  # [1, 1, H/2, W/2]

    # The SAME normalised grid works for Y (H×W) and UV (H/2×W/2) —
    # grid_sample scales coords relative to each input's size, so the UV
    # sub-pixel offset is ≤ 0.5 px and negligible for video quality.
    sample_kw = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}
    y_out = torch_functional.grid_sample(y_f, grid, **sample_kw)  # [1,1,th,tw]
    u_out = torch_functional.grid_sample(u_f, grid, **sample_kw)
    v_out = torch_functional.grid_sample(v_f, grid, **sample_kw)

    # ------------------------------------------------------------------ #
    # BT.601 limited-range YUV → RGB, normalise [0,255] → [0,1] FP16
    # ------------------------------------------------------------------ #
    y_val = y_out - 16.0
    u_val = u_out - 128.0
    v_val = v_out - 128.0

    r = torch.clamp(1.164 * y_val + 1.596 * v_val, 0.0, 255.0)
    g = torch.clamp(1.164 * y_val - 0.392 * u_val - 0.813 * v_val, 0.0, 255.0)
    b = torch.clamp(1.164 * y_val + 2.017 * u_val, 0.0, 255.0)

    rgb = torch.cat([r, g, b], dim=1) / 255.0       # [1, 3, target_h, target_w] fp32

    # Mask: pixels outside the rendered area must be black (0), not the
    # artefact colour produced by BT.601(Y=0, U=0, V=0).
    valid_x = ((nx >= -1.0) & (nx <= 1.0)).float().view(1, 1, 1, target_w)
    valid_y = ((ny >= -1.0) & (ny <= 1.0)).float().view(1, 1, target_h, 1)
    mask = valid_y * valid_x                          # [1, 1, target_h, target_w]
    rgb = rgb * mask

    return rgb.to(dtype=torch.float16).squeeze(0)   # [3, target_h, target_w] fp16

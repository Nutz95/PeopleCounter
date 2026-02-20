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

# ---------------------------------------------------------------------------
# Per-session CUDA tensor caches — avoids reallocating large tensors every frame.
#
# _PLANE_BUFFER_CACHE: reusable [H, W] uint8 + fp32 buffers for Y and UV planes.
#   Key: (stream_id, height, width) — keyed by stream_id so parallel threads
#   operating on different CUDA streams never share the same memory.
#   Content is overwritten each frame by cudaMemcpy2DAsync on that stream.
#
# _BATCH_GRID_CACHE: the [N, target_h, target_w, 2] float32 sampling grids for
#   the tiling path.  Key: (height, width, target_h, target_w, tile_crops_tuple).
#   Grids are purely geometric (no frame content) so they never change for a
#   fixed camera + config.  For a 4K source / 16 tiles at 640×640 this tensor
#   is ~50 MB — caching it saves that allocation + grid-building ops every frame.
#
# _LETTERBOX_GRID_CACHE: same idea for the single-tile letterbox grid [1,th,tw,2].
#   Key: (height, width, target_h, target_w).
# ---------------------------------------------------------------------------
# (stream_id, height, width) -> (y_uint8, uv_uint8, y_fp32, u_fp32, v_fp32)
# y_fp32: [1, 1, H, W]   u_fp32/v_fp32: [1, 1, H//2, W//2]
_PLANE_BUFFER_CACHE: dict[tuple[int, int, int], tuple[Any, Any, Any, Any, Any]] = {}
_BATCH_GRID_CACHE: dict[tuple, Any] = {}
_LETTERBOX_GRID_CACHE: dict[tuple[int, int, int, int], Any] = {}

# BT.601 limited-range YUV→RGB matrix (row = output channel, col = input channel).
# Applied as: rgb[n,o,h,w] = clamp(sum_i M[o,i] * (yuv[n,i,h,w] - offset[i]), 0, 255) / 255
_YUV_TO_RGB_MATRIX: Any = None       # lazily initialised 3×3 fp32 on CUDA
_YUV_OFFSETS: Any = None             # lazily initialised [3] fp32 on CUDA


def _get_yuv_matrix() -> tuple[Any, Any]:
    """Return (matrix [3,3], offsets [1,3,1,1]) cached on CUDA, fp32."""
    global _YUV_TO_RGB_MATRIX, _YUV_OFFSETS
    if _YUV_TO_RGB_MATRIX is None and torch is not None and torch.cuda.is_available():
        _YUV_TO_RGB_MATRIX = torch.tensor(
            [[1.164,  0.000,  1.596],
             [1.164, -0.392, -0.813],
             [1.164,  2.017,  0.000]],
            dtype=torch.float32, device="cuda",
        )
        _YUV_OFFSETS = torch.tensor(
            [16.0, 128.0, 128.0], dtype=torch.float32, device="cuda"
        ).view(1, 3, 1, 1)
    return _YUV_TO_RGB_MATRIX, _YUV_OFFSETS


def _get_plane_buffers(
    height: int, width: int, stream_id: int = 0
) -> tuple[Any, Any, Any, Any, Any]:
    """Return (y_u8, uv_u8, y_f32[1,1,H,W], u_f32[1,1,H//2,W//2], v_f32[1,1,H//2,W//2]).

    All tensors are pre-allocated once and reused across frames.
    Keyed by (stream_id, height, width) so concurrent threads on different
    CUDA streams never share the same buffers (avoids read-while-write races).
    The pre-allocated fp32 views are filled in-place with copy_() to avoid
    the per-frame .float() allocation (~33 MB per call).
    """
    key = (stream_id, height, width)
    if key not in _PLANE_BUFFER_CACHE:
        assert torch is not None
        half_h = max(1, height // 2)
        half_w = max(1, width // 2)
        y_u8 = torch.empty((height, width), dtype=torch.uint8, device="cuda")
        uv_u8 = torch.empty((half_h, width), dtype=torch.uint8, device="cuda")
        # Pre-allocated fp32 views used in-place each frame via copy_()
        y_f32 = torch.empty((1, 1, height, width), dtype=torch.float32, device="cuda")
        u_f32 = torch.empty((1, 1, half_h, half_w), dtype=torch.float32, device="cuda")
        v_f32 = torch.empty((1, 1, half_h, half_w), dtype=torch.float32, device="cuda")
        _PLANE_BUFFER_CACHE[key] = (y_u8, uv_u8, y_f32, u_f32, v_f32)
    return _PLANE_BUFFER_CACHE[key]


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


def nv12_to_rgb_hwc_resized_cuda(
    frame: Any,
    target_h: int,
    target_w: int,
    stream_id: int = 0,
) -> Any:
    """NV12 → bilinear resize in YUV space → RGB HWC uint8.

    More efficient than ``nv12_frame_to_rgb_hwc_cuda`` + ``F.interpolate`` on
    the resulting RGB tensor because:

    - NV12 is 1.5 bytes/pixel; RGB is 3 bytes/pixel.  Resizing in YUV space
      means the bilinear kernels process 2× less data.
    - The full-resolution RGB intermediate (~25 MB for 4K) is *never* allocated;
      the YUV→RGB matrix multiply runs on the smaller (target-resolution) planes.
    - Plane buffers are reused across frames via ``_get_plane_buffers``, so the
      common case (same frame dimensions) incurs zero allocation per call.

    Args:
        frame:     GpuFrame with ``device_ptr_y``, ``device_ptr_uv``,
                   ``pitch``, ``width``, ``height``.
        target_h:  Output height in pixels.
        target_w:  Output width in pixels.
        stream_id: Passed to :func:`_get_plane_buffers` to partition the buffer
                   cache so concurrent callers on different CUDA streams never
                   share the same memory.  Use a value that does not overlap with
                   any preprocess stream ID (0–5) — e.g. 99 for the video encoder.

    Returns:
        ``[target_h, target_w, 3]`` uint8 CUDA tensor in RGB order.
    """
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for NV12 resize bridge")
    if torch_functional is None:
        raise RuntimeError("torch.nn.functional is required for NV12 resize bridge")
    if cuda_runtime is None:
        raise RuntimeError("cuda.bindings.runtime is required for NV12 resize bridge")

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    pitch = int(getattr(frame, "pitch") or width)
    half_h = max(1, height // 2)
    half_w = max(1, width // 2)

    y_ptr = _extract_device_pointer(getattr(frame, "device_ptr_y", None))
    uv_ptr = _extract_device_pointer(getattr(frame, "device_ptr_uv", None))
    if y_ptr <= 0:
        raise RuntimeError("NV12 frame missing valid device_ptr_y")
    if uv_ptr <= 0 or uv_ptr == y_ptr:
        uv_ptr = y_ptr + pitch * height

    # ── Copy NV12 planes using per-stream cached fp32 views ────────────────
    y_plane, uv_plane, y_f, u_f, v_f = _get_plane_buffers(height, width, stream_id)
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
        height_rows=half_h,
    )

    # Fill pre-allocated fp32 views in-place (no per-frame allocation).
    y_f.copy_(y_plane.unsqueeze(0).unsqueeze(0))              # [1, 1, H, W]
    uv_hw2 = uv_plane.view(half_h, half_w, 2)
    u_f.copy_(uv_hw2[:, :, 0].unsqueeze(0).unsqueeze(0))     # [1, 1, H//2, W//2]
    v_f.copy_(uv_hw2[:, :, 1].unsqueeze(0).unsqueeze(0))

    # ── Bilinear resize in YUV space ────────────────────────────────────────
    # Y:  [1,1,H,W]     → [1,1,target_h,target_w]  (bilinear)
    # UV: [1,1,H//2,W//2] → [1,1,target_h,target_w] (bilinear; handles both
    #     downscale and the implicit 2× chroma upsample in one pass)
    interp_kw = {"mode": "bilinear", "align_corners": False}
    y_s = torch_functional.interpolate(y_f, size=(target_h, target_w), **interp_kw)
    u_s = torch_functional.interpolate(u_f, size=(target_h, target_w), **interp_kw)
    v_s = torch_functional.interpolate(v_f, size=(target_h, target_w), **interp_kw)

    # ── BT.601 YUV→RGB via matrix multiply (1 matmul+clamp vs ~18 ops) ─────
    M, offsets = _get_yuv_matrix()
    yuv = torch.cat([y_s, u_s, v_s], dim=1) - offsets        # [1, 3, th, tw]
    rgb = (yuv.permute(0, 2, 3, 1) @ M.T).permute(0, 3, 1, 2)  # [1, 3, th, tw]
    rgb = rgb.clamp_(0.0, 255.0)

    # [th, tw, 3] uint8 — HWC layout expected by torchvision.io.encode_jpeg
    return rgb.squeeze(0).permute(1, 2, 0).to(dtype=torch.uint8)


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

    # Source byte offsets for the crop windows — include the horizontal column offset.
    # Each row in the Y plane is `pitch` bytes wide; crop_x bytes skips to the first
    # column of the crop region within that row.
    y_crop_ptr = y_ptr + crop_y * pitch + crop_x
    uv_crop_ptr = uv_ptr + uv_row_first * pitch + crop_x

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


def nv12_tiles_to_rgb_nchw_fp16_batch(
    frame: Any,
    tile_crops: list[tuple[int, int, int, int]],
    target_h: int,
    target_w: int,
    stream_id: int = 0,
) -> Any:
    """Batch NV12 tiling: one full-frame plane copy, all tiles in a single grid_sample pass.

    Unlike calling :func:`nv12_crop_to_rgb_nchw_fp16` in a loop (which launches
    ~30 CUDA ops × N tiles from Python), this function:

    - Copies the full Y and UV planes exactly **once** (2 CUDA memcpy calls total).
    - Builds all per-tile sampling grids with **vectorised** tensor ops — no Python
      loop over tiles.
    - Runs a **single** batched ``grid_sample`` for Y, U, and V — one kernel launch
      each regardless of how many tiles there are.

    This reduces Python→CUDA dispatch overhead from O(N × ops_per_tile) to O(ops),
    cutting preprocess dispatch latency for 16 tiles from ~40 ms to < 1 ms.

    Args:
        tile_crops: List of ``(crop_x, crop_y, crop_w, crop_h)`` for each tile,
            expressed in full-frame pixel coordinates.
        stream_id: CUDA stream identifier — used to partition the plane-buffer
            cache so concurrent threads never share the same buffers.

    Returns:
        ``[num_tiles, 3, target_h, target_w]`` float16 tensor in *[0, 1]* range.
    """
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for batch NV12 tiling")
    if torch_functional is None:
        raise RuntimeError("torch.nn.functional is required for batch NV12 tiling")
    if cuda_runtime is None:
        raise RuntimeError("cuda.bindings.runtime is required for batch NV12 tiling")

    num_tiles = len(tile_crops)
    if num_tiles == 0:
        return torch.empty((0, 3, target_h, target_w), dtype=torch.float16, device="cuda")

    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    pitch = int(getattr(frame, "pitch") or width)

    y_ptr = _extract_device_pointer(getattr(frame, "device_ptr_y", None))
    uv_ptr = _extract_device_pointer(getattr(frame, "device_ptr_uv", None))
    if y_ptr <= 0:
        raise RuntimeError("NV12 frame missing valid device_ptr_y")
    if uv_ptr <= 0 or uv_ptr == y_ptr:
        uv_ptr = y_ptr + pitch * height

    # --- Plane buffers: per-stream cache to avoid cross-thread aliasing ---
    y_plane, uv_plane, y_f, u_f, v_f = _get_plane_buffers(height, width, stream_id)
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

    # --- Fill pre-allocated fp32 views in-place (no new allocation per frame) ---
    # y_f: [1, 1, H, W]  u_f/v_f: [1, 1, H//2, W//2]
    y_f.copy_(y_plane.unsqueeze(0).unsqueeze(0))
    half_h = max(1, height // 2)
    half_w = max(1, width // 2)
    uv_hw2 = uv_plane.view(half_h, half_w, 2)  # stride-trick, no copy
    u_f.copy_(uv_hw2[:, :, 0].unsqueeze(0).unsqueeze(0))
    v_f.copy_(uv_hw2[:, :, 1].unsqueeze(0).unsqueeze(0))

    # --- Sampling grid: cached across frames (purely geometric, no pixel data) ---
    # [N, target_h, target_w, 2]. For a 4K source with 16 tiles at 640×640 this
    # is ~50 MB of fp32 — caching avoids recomputing + reallocating it every frame.
    grid_key = (height, width, target_h, target_w, tuple(tile_crops))
    if grid_key not in _BATCH_GRID_CACHE:
        crops_tensor = torch.tensor(tile_crops, device="cuda", dtype=torch.float32)
        crops_x = crops_tensor[:, 0]
        crops_y = crops_tensor[:, 1]
        crops_w = crops_tensor[:, 2]
        crops_h = crops_tensor[:, 3]

        pixel_cols = torch.arange(target_w, device="cuda", dtype=torch.float32).unsqueeze(0)
        pixel_rows = torch.arange(target_h, device="cuda", dtype=torch.float32).unsqueeze(0)
        src_x = crops_x.unsqueeze(1) + (pixel_cols + 0.5) * crops_w.unsqueeze(1) / target_w
        src_y = crops_y.unsqueeze(1) + (pixel_rows + 0.5) * crops_h.unsqueeze(1) / target_h

        norm_x = (src_x + 0.5) * (2.0 / width)  - 1.0
        norm_y = (src_y + 0.5) * (2.0 / height) - 1.0

        grid_x = norm_x.unsqueeze(1).expand(num_tiles, target_h, target_w)
        grid_y = norm_y.unsqueeze(2).expand(num_tiles, target_h, target_w)
        _BATCH_GRID_CACHE[grid_key] = torch.stack([grid_x, grid_y], dim=-1).contiguous()

    batch_grid = _BATCH_GRID_CACHE[grid_key]

    # expand() is a stride-0 view — no data copy.
    y_batch = y_f.expand(num_tiles, 1, height, width)
    u_batch = u_f.expand(num_tiles, 1, max(1, height // 2), max(1, width // 2))
    v_batch = v_f.expand(num_tiles, 1, max(1, height // 2), max(1, width // 2))

    sample_kw = {"mode": "bilinear", "padding_mode": "border", "align_corners": False}
    y_out = torch_functional.grid_sample(y_batch, batch_grid, **sample_kw)  # [N, 1, th, tw]
    u_out = torch_functional.grid_sample(u_batch, batch_grid, **sample_kw)
    v_out = torch_functional.grid_sample(v_batch, batch_grid, **sample_kw)

    # --- BT.601 YUV→RGB via matrix multiply: 1 matmul+clamp+div vs ~18 separate ops ---
    # yuv: [N, 3, th, tw]; subtract offsets [16, 128, 128] then apply M [3×3]
    M, offsets = _get_yuv_matrix()
    yuv = torch.cat([y_out, u_out, v_out], dim=1) - offsets  # [N, 3, th, tw]
    # permute to [N, th, tw, 3] for @ then back
    rgb = (yuv.permute(0, 2, 3, 1) @ M.T).permute(0, 3, 1, 2)  # [N, 3, th, tw]
    rgb = rgb.clamp_(0.0, 255.0).div_(255.0)
    return rgb.to(dtype=torch.float16)            # [N, 3, th, tw] fp16


def nv12_to_rgb_nchw_fp16_letterbox(
    frame: Any,
    target_h: int,
    target_w: int,
    stream_id: int = 0,
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
    # Copy Y and UV planes — per-stream cached buffers (thread-safe, no race)
    # ------------------------------------------------------------------ #
    y_plane, uv_plane, y_f, u_f, v_f = _get_plane_buffers(height, width, stream_id)

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

    # Fill pre-allocated fp32 views in-place (no new tensor allocation per frame)
    y_f.copy_(y_plane.unsqueeze(0).unsqueeze(0))
    half_h = max(1, height // 2)
    half_w = max(1, width // 2)
    uv_hw2 = uv_plane.view(half_h, half_w, 2)
    u_f.copy_(uv_hw2[:, :, 0].unsqueeze(0).unsqueeze(0))
    v_f.copy_(uv_hw2[:, :, 1].unsqueeze(0).unsqueeze(0))

    # ------------------------------------------------------------------ #
    # Letterbox sampling grid [1, target_h, target_w, 2] — cached (purely geometric)
    # ------------------------------------------------------------------ #
    grid_key = (height, width, target_h, target_w)
    if grid_key not in _LETTERBOX_GRID_CACHE:
        scale = min(target_w / max(1, width), target_h / max(1, height))
        render_w = max(1, int(round(width * scale)))
        render_h = max(1, int(round(height * scale)))
        off_x = (target_w - render_w) // 2
        off_y = (target_h - render_h) // 2

        oj = torch.arange(target_w, device="cuda", dtype=torch.float32)
        oi = torch.arange(target_h, device="cuda", dtype=torch.float32)

        nx = (oj - off_x) * width / render_w
        ny = (oi - off_y) * height / render_h
        nx = (nx + 0.5) * (2.0 / width) - 1.0
        ny = (ny + 0.5) * (2.0 / height) - 1.0

        gx = nx.view(1, target_w).expand(target_h, target_w)
        gy = ny.view(target_h, 1).expand(target_h, target_w)
        cached_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).contiguous()

        # Precompute the letterbox mask (pixels outside the render area → black)
        valid_x = ((nx >= -1.0) & (nx <= 1.0)).float().view(1, 1, 1, target_w)
        valid_y = ((ny >= -1.0) & (ny <= 1.0)).float().view(1, 1, target_h, 1)
        cached_mask = (valid_y * valid_x).contiguous()  # [1, 1, target_h, target_w]
        _LETTERBOX_GRID_CACHE[grid_key] = (cached_grid, cached_mask)

    grid, valid_mask = _LETTERBOX_GRID_CACHE[grid_key]

    # ------------------------------------------------------------------ #
    # Sample Y, U, V at output resolution — using pre-allocated fp32 views
    # ------------------------------------------------------------------ #
    # The SAME normalised grid works for Y (H×W) and UV (H/2×W/2).
    sample_kw = {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}
    y_out = torch_functional.grid_sample(y_f, grid, **sample_kw)  # [1,1,th,tw]
    u_out = torch_functional.grid_sample(u_f, grid, **sample_kw)
    v_out = torch_functional.grid_sample(v_f, grid, **sample_kw)

    # ------------------------------------------------------------------ #
    # BT.601 YUV→RGB via matrix multiply (1 matmul+clamp vs ~18 separate ops)
    # Apply letterbox black mask on the output.
    # ------------------------------------------------------------------ #
    M, offsets = _get_yuv_matrix()
    yuv = torch.cat([y_out, u_out, v_out], dim=1) - offsets  # [1, 3, th, tw]
    rgb = (yuv.permute(0, 2, 3, 1) @ M.T).permute(0, 3, 1, 2)  # [1, 3, th, tw]
    rgb = rgb.clamp_(0.0, 255.0).div_(255.0)
    rgb = rgb * valid_mask

    return rgb.to(dtype=torch.float16).squeeze(0)  # [3, target_h, target_w] fp16

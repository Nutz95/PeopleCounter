from __future__ import annotations

from dataclasses import dataclass

import pytest

from app_v2.kernels.nv12_cuda_bridge import (
    nv12_frame_to_rgb_hwc_cuda,
    nv12_to_rgb_nchw_fp16_letterbox,
)


torch = pytest.importorskip("torch")

try:
    from cuda.bindings import runtime as _cuda_runtime  # noqa: F401
except Exception:
    _cuda_runtime = None


@dataclass
class _DummyNv12Frame:
    width: int
    height: int
    pitch: int
    device_ptr_y: int
    device_ptr_uv: int

    def is_nv12(self) -> bool:
        return True


_SKIP_NO_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for NV12 bridge test")
_SKIP_NO_CUDABIND = pytest.mark.skipif(_cuda_runtime is None, reason="cuda.bindings.runtime is required for NV12 bridge test")


@_SKIP_NO_CUDA
@_SKIP_NO_CUDABIND
def test_nv12_cuda_bridge_produces_rgb_tensor_on_gpu() -> None:
    width = 64
    height = 32

    y_plane = torch.full((height, width), 96, dtype=torch.uint8, device="cuda")
    uv_plane = torch.full((height // 2, width), 128, dtype=torch.uint8, device="cuda")

    frame = _DummyNv12Frame(
        width=width,
        height=height,
        pitch=width,
        device_ptr_y=int(y_plane.data_ptr()),
        device_ptr_uv=int(uv_plane.data_ptr()),
    )

    rgb = nv12_frame_to_rgb_hwc_cuda(frame)

    assert rgb.is_cuda
    assert rgb.dtype == torch.uint8
    assert tuple(rgb.shape) == (height, width, 3)


@_SKIP_NO_CUDA
@_SKIP_NO_CUDABIND
def test_fused_nv12_letterbox_shape_and_dtype() -> None:
    """Fused function returns [3, target_h, target_w] fp16 on CUDA."""
    width, height = 128, 64
    target_h, target_w = 32, 32

    y_plane = torch.full((height, width), 96, dtype=torch.uint8, device="cuda")
    uv_plane = torch.full((height // 2, width), 128, dtype=torch.uint8, device="cuda")

    frame = _DummyNv12Frame(
        width=width,
        height=height,
        pitch=width,
        device_ptr_y=int(y_plane.data_ptr()),
        device_ptr_uv=int(uv_plane.data_ptr()),
    )

    out = nv12_to_rgb_nchw_fp16_letterbox(frame, target_h, target_w)

    assert out.is_cuda
    assert out.dtype == torch.float16
    assert tuple(out.shape) == (3, target_h, target_w)


@_SKIP_NO_CUDA
@_SKIP_NO_CUDABIND
def test_fused_nv12_letterbox_values_in_01() -> None:
    """All output values must lie in [0, 1] (normalised)."""
    width, height = 64, 32
    y_plane = torch.randint(16, 235, (height, width), dtype=torch.uint8, device="cuda")
    uv_plane = torch.randint(16, 240, (height // 2, width), dtype=torch.uint8, device="cuda")

    frame = _DummyNv12Frame(
        width=width,
        height=height,
        pitch=width,
        device_ptr_y=int(y_plane.data_ptr()),
        device_ptr_uv=int(uv_plane.data_ptr()),
    )

    out = nv12_to_rgb_nchw_fp16_letterbox(frame, 32, 32)
    vals = out.float()
    assert float(vals.min()) >= -0.01, "values should be >= 0"
    assert float(vals.max()) <= 1.01, "values should be <= 1"


@_SKIP_NO_CUDA
@_SKIP_NO_CUDABIND
def test_fused_nv12_letterbox_padding_region_is_zero() -> None:
    """Wide letterbox should have zero-padded rows at top and bottom."""
    # 4:1 wide image → lots of vertical padding
    width, height = 128, 16
    target_h, target_w = 32, 32   # square target → vertical bars

    y_plane = torch.full((height, width), 120, dtype=torch.uint8, device="cuda")
    uv_plane = torch.full((height // 2, width), 128, dtype=torch.uint8, device="cuda")

    frame = _DummyNv12Frame(
        width=width,
        height=height,
        pitch=width,
        device_ptr_y=int(y_plane.data_ptr()),
        device_ptr_uv=int(uv_plane.data_ptr()),
    )

    out = nv12_to_rgb_nchw_fp16_letterbox(frame, target_h, target_w)
    # Top padding row should be all zeros
    top_row = out[:, 0, :]
    assert float(top_row.abs().max()) < 0.01, "top letterbox padding should be zero"


from __future__ import annotations

from dataclasses import dataclass

import pytest

from app_v2.kernels.nv12_cuda_bridge import nv12_frame_to_rgb_hwc_cuda


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for NV12 bridge test")
@pytest.mark.skipif(_cuda_runtime is None, reason="cuda.bindings.runtime is required for NV12 bridge test")
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

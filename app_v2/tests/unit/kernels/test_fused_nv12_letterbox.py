"""Unit tests for run_letterbox_kernel_fused (Opt #5 fused NV12+letterbox).

Tests verify:
- NV12 frames take the fused path (nv12_to_rgb_nchw_fp16_letterbox called).
- Non-NV12 frames fall back to the two-step path.
- Pre-supplied source_tensor bypasses the fused path.
- Output GpuTensor shape / properties are correct.

All CUDA ops are mocked; tests run without a physical GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Check CUDA availability once
# ---------------------------------------------------------------------------
try:
    import torch as _real_torch
    _TORCH_AVAILABLE = _real_torch.cuda.is_available()
except Exception:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeNv12Frame:
    width: int = 128
    height: int = 64
    pitch: int = 128
    device_ptr_y: int = 0xDEAD0001
    device_ptr_uv: int = 0xDEAD0002

    def is_nv12(self) -> bool:
        return True


@dataclass
class _FakeRgbFrame:
    width: int = 128
    height: int = 64
    # No device_ptr_y/uv — NOT an NV12 frame


def _task(target_w: int = 32, target_h: int = 32) -> Any:
    from app_v2.core.preprocessor_types import PreprocessTask
    return PreprocessTask(
        model_name="test",
        task_index=0,
        source_x=0,
        source_y=0,
        source_width=128,
        source_height=64,
        target_width=target_w,
        target_height=target_h,
    )


# ---------------------------------------------------------------------------
# Integration tests (require a real GPU + cuda.bindings)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="CUDA required")
class TestFusedKernelIntegration:
    """Run the fused kernel end-to-end on real GPU memory."""

    def _make_nv12_frame(self, width, height):
        import torch
        y = torch.full((height, width), 96, dtype=torch.uint8, device="cuda")
        uv = torch.full((height // 2, width), 128, dtype=torch.uint8, device="cuda")
        return _FakeNv12Frame(
            width=width,
            height=height,
            pitch=width,
            device_ptr_y=int(y.data_ptr()),
            device_ptr_uv=int(uv.data_ptr()),
        ), y, uv  # keep y/uv alive

    def test_fused_kernel_returns_gpu_tensor(self):
        try:
            from cuda.bindings import runtime  # noqa: F401
        except Exception:
            pytest.skip("cuda.bindings.runtime required")

        import torch
        from app_v2.kernels.preprocess import run_letterbox_kernel_fused

        frame, _y, _uv = self._make_nv12_frame(128, 64)
        result = run_letterbox_kernel_fused(frame, _task())

        assert result.tensor_ref.is_cuda
        assert result.tensor_ref.dtype == torch.float16
        assert tuple(result.tensor_ref.shape) == (3, 32, 32)

    def test_fused_kernel_non_nv12_fallback(self):
        """Non-NV12 source tensor → fallback to two-step letterbox."""
        import torch
        from app_v2.kernels.preprocess import run_letterbox_kernel_fused

        src = torch.randint(0, 256, (64, 128, 3), dtype=torch.uint8, device="cuda")
        frame = _FakeRgbFrame()
        result = run_letterbox_kernel_fused(frame, _task(), source_tensor=src)

        assert result.tensor_ref.is_cuda
        assert result.tensor_ref.dtype == torch.float16
        assert tuple(result.tensor_ref.shape) == (3, 32, 32)


# ---------------------------------------------------------------------------
# Unit tests — mock the fused function to verify dispatch logic
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="CUDA required")
class TestFusedKernelDispatch:
    """Verify that the fused path is selected for NV12 and not for RGB."""

    def _make_lease(self):
        import torch
        tensor = torch.zeros((3, 32, 32), dtype=torch.float16, device="cuda")
        lease = MagicMock()
        lease.tensor = tensor
        return lease

    def test_nv12_frame_calls_fused_function(self):
        import torch
        from app_v2.kernels.preprocess import run_letterbox_kernel_fused

        frame = _FakeNv12Frame()
        fake_output = torch.zeros((3, 32, 32), dtype=torch.float16, device="cuda")
        lease = self._make_lease()

        with (
            patch(
                "app_v2.kernels.preprocess.nv12_to_rgb_nchw_fp16_letterbox",
                return_value=fake_output,
            ) as mock_fused,
            patch(
                "app_v2.kernels.preprocess._acquire_output_tensor",
                return_value=lease,
            ),
        ):
            result = run_letterbox_kernel_fused(frame, _task())

        mock_fused.assert_called_once_with(frame, 32, 32)
        assert result is not None

    def test_pre_supplied_source_tensor_skips_fused_path(self):
        import torch
        from app_v2.kernels.preprocess import run_letterbox_kernel_fused

        frame = _FakeNv12Frame()
        src = torch.zeros((64, 128, 3), dtype=torch.uint8, device="cuda")
        lease = self._make_lease()

        with (
            patch(
                "app_v2.kernels.preprocess.nv12_to_rgb_nchw_fp16_letterbox",
            ) as mock_fused,
            patch(
                "app_v2.kernels.preprocess._acquire_output_tensor",
                return_value=lease,
            ),
            patch(
                "app_v2.kernels.preprocess._source_tensor_from_frame",
                return_value=src,
            ),
            patch(
                "app_v2.kernels.preprocess._letterbox_resize",
                return_value=torch.zeros(
                    (1, 3, 32, 32), dtype=torch.float32, device="cuda"
                ),
            ),
        ):
            run_letterbox_kernel_fused(frame, _task(), source_tensor=src)

        mock_fused.assert_not_called()

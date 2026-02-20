"""Unit tests for GPU NVJPEG JPEG encoding path.

These tests verify that:
  1. torchvision.io.encode_jpeg is available in the Docker image.
  2. The NVJPEG path (CUDA input tensor) produces valid JPEG bytes that can be
     decoded back to the original image shape.
  3. The NVJPEG encode throughput is non-trivially fast (<50 ms for 720p).

Run inside Docker:
    pytest app_v2/tests/unit/infrastructure/test_nvjpeg_encode.py -v
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


# ─── helpers ──────────────────────────────────────────────────────────────────


def _is_cuda_available() -> bool:
    return torch.cuda.is_available()


def _has_torchvision_io() -> bool:
    try:
        import torchvision.io  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_cuda = pytest.mark.skipif(not _is_cuda_available(), reason="CUDA device required")
skip_no_tvio = pytest.mark.skipif(not _has_torchvision_io(), reason="torchvision not installed")


# ─── tests ────────────────────────────────────────────────────────────────────


def test_torchvision_io_import() -> None:
    """torchvision.io must be importable (checks requirements.cuda.txt presence)."""
    import torchvision.io  # noqa: F401

    assert hasattr(torchvision.io, "encode_jpeg"), (
        "torchvision.io.encode_jpeg missing — torchvision >= 0.12 required"
    )


@skip_no_cuda
@skip_no_tvio
def test_nvjpeg_encode_returns_non_empty_bytes() -> None:
    """encode_jpeg on a CUDA uint8 tensor must return a non-empty 1-D uint8 tensor."""
    import torchvision.io as tvio

    chw = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8, device="cuda")
    buf = tvio.encode_jpeg(chw, quality=75)

    assert isinstance(buf, torch.Tensor), "encode_jpeg must return a Tensor"
    assert buf.dtype == torch.uint8, "output dtype should be uint8"
    assert buf.ndim == 1, "output must be a 1-D byte stream"
    assert buf.numel() > 256, "JPEG output suspiciously small"


@skip_no_cuda
@skip_no_tvio
def test_nvjpeg_encode_decode_roundtrip() -> None:
    """Encode a synthetic CUDA image and decode back — shape must be preserved."""
    import torchvision.io as tvio

    original = torch.zeros(3, 480, 640, dtype=torch.uint8, device="cuda")
    original[0] = 200   # red channel
    original[1] = 100   # green channel
    original[2] = 50    # blue channel

    buf = tvio.encode_jpeg(original, quality=90)
    # encode_jpeg with CUDA input returns a CUDA tensor — move to CPU for decode_image.
    decoded = tvio.decode_image(buf.cpu())
    assert decoded.shape == original.shape, (
        f"Decoded shape {decoded.shape} != original {original.shape}"
    )
    assert decoded.dtype == torch.uint8


@skip_no_cuda
@skip_no_tvio
def test_nvjpeg_encode_720p_latency() -> None:
    """NVJPEG encode of a 1280×720 image should complete within 50 ms."""
    import torchvision.io as tvio

    chw = torch.randint(0, 255, (3, 720, 1280), dtype=torch.uint8, device="cuda")

    # Warm up: the first call incurs driver/library init overhead.
    tvio.encode_jpeg(chw, quality=75)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    buf = tvio.encode_jpeg(chw, quality=75)
    # encode_jpeg with CUDA input returns a CUDA tensor; .cpu() is a PCIe DMA copy.
    jpeg_bytes = bytes(buf.cpu().numpy())
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    assert len(jpeg_bytes) > 0, "Empty JPEG output"
    assert elapsed_ms < 50.0, f"NVJPEG 720p encode took {elapsed_ms:.1f} ms — too slow"


@skip_no_cuda
@skip_no_tvio
def test_nvjpeg_encode_cuda_input_uses_gpu() -> None:
    """Verify that the CUDA tensor does not need to be moved to CPU before encode."""
    import torchvision.io as tvio

    chw_cuda = torch.randint(0, 255, (3, 360, 640), dtype=torch.uint8, device="cuda")
    # Should not raise when given a CUDA tensor directly.
    buf = tvio.encode_jpeg(chw_cuda, quality=75)
    assert buf.numel() > 0


# ─── Benchmark: NV12→YUV-resize→RGB + NVJPEG at multiple resolutions ──────────

class _FakeGpuFrame:
    """Synthetic 4K NV12 frame backed by a real CUDA allocation.

    All ops in ``nv12_to_rgb_hwc_resized_cuda`` and ``nv12_frame_to_rgb_hwc_cuda``
    only touch ``width``, ``height``, ``pitch``, ``device_ptr_y``,
    ``device_ptr_uv`` — so this minimal stand-in is sufficient for benchmarking.
    """

    def __init__(self, h: int, w: int) -> None:
        half_h = max(1, h // 2)
        # Contiguous NV12 layout: [Y plane: H×W] followed by [UV plane: (H/2)×W]
        buf = torch.randint(16, 220, (h * w + half_h * w,), dtype=torch.uint8, device="cuda")
        self._buf = buf
        self.height = h
        self.width = w
        self.pitch = w
        self.device_ptr_y = int(buf.data_ptr())
        self.device_ptr_uv = int(buf.data_ptr() + h * w)


@skip_no_cuda
@skip_no_tvio
def test_nvjpeg_4k_multi_resolution_benchmark() -> None:
    """Benchmark NV12→NVJPEG at multiple output resolutions from a 4K source.

    Compares two conversion paths:
      A. NV12 → full-res RGB(4K) → bilinear resize → NVJPEG  [old / naive path]
      B. NV12 → bilinear resize in YUV space → RGB(target) → NVJPEG  [new path]

    Run with ``pytest -s`` to see the printed table.

    ┌────────────┬──────────────┬──────────────┬─────────┬──────────────┐
    │ Resolution │ A: Old  (ms) │ B: New  (ms) │ Speedup │   JPEG size  │
    └────────────┴──────────────┴──────────────┴─────────┴──────────────┘
    """
    import torch.nn.functional as F
    import torchvision.io as tvio
    from app_v2.kernels.nv12_cuda_bridge import (
        nv12_frame_to_rgb_hwc_cuda,
        nv12_to_rgb_hwc_resized_cuda,
    )

    SOURCE_H, SOURCE_W = 2160, 3840  # 4K UHD source
    frame = _FakeGpuFrame(SOURCE_H, SOURCE_W)

    targets = [
        ("4K  (native)", SOURCE_H, SOURCE_W),
        ("1440p       ", 1440, 2560),
        ("1080p       ", 1080, 1920),
        ("720p        ", 720, 1280),
    ]

    WARMUP = 3
    ITERS = 7

    def _time_path(fn: "Callable[[], None]") -> float:
        """Return median wall-clock ms over ITERS runs (GPU-synchronised)."""
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1_000)
        times.sort()
        return times[ITERS // 2]  # median

    results: list[tuple[str, float, float, float]] = []

    for label, target_h, target_w in targets:
        # ── Path A: NV12→full-res RGB→resize→NVJPEG ──────────────────────
        jpeg_size_kb = 0.0

        def _path_a() -> None:
            nonlocal jpeg_size_kb
            rgb = nv12_frame_to_rgb_hwc_cuda(frame)            # [H, W, 3] uint8
            if target_h < SOURCE_H:
                nchw = rgb.permute(2, 0, 1).unsqueeze(0).float()
                nchw = F.interpolate(nchw, (target_h, target_w), mode="bilinear", align_corners=False)
                chw = nchw.squeeze(0).byte()
            else:
                chw = rgb.permute(2, 0, 1).contiguous()
            buf = tvio.encode_jpeg(chw, quality=75)
            jpeg_size_kb = buf.numel() / 1024

        t_old = _time_path(_path_a)

        # ── Path B: NV12→YUV-resize→RGB(target)→NVJPEG ───────────────────
        def _path_b() -> None:
            nonlocal jpeg_size_kb
            rgb = nv12_to_rgb_hwc_resized_cuda(frame, target_h, target_w, stream_id=99)
            chw = rgb.permute(2, 0, 1).contiguous()
            buf = tvio.encode_jpeg(chw, quality=75)
            jpeg_size_kb = buf.numel() / 1024

        t_new = _time_path(_path_b)

        results.append((label, t_old, t_new, jpeg_size_kb))

    # ── Print results table ───────────────────────────────────────────────
    hdr = (
        f"\n{'':2}{'Resolution':<14}"
        f"{'A: old NV12→RGB→resize (ms)':>30}"
        f"{'B: NV12→YUV-resize→RGB (ms)':>30}"
        f"{'Speedup':>10}"
        f"{'JPEG size':>12}"
    )
    sep = "─" * len(hdr)
    print(hdr)
    print(sep)
    for label, t_a, t_b, sz_kb in results:
        speedup = t_a / max(t_b, 0.001)
        flag = "✅" if speedup >= 1.0 else "⚠️ "
        print(
            f"  {label:<14}"
            f"{t_a:>30.2f}"
            f"{t_b:>30.2f}"
            f"{speedup:>9.2f}×"
            f"{sz_kb:>10.0f} KB"
            f"  {flag}"
        )
    print(sep)

    # Sanity: all encodes produced non-empty output.
    assert all(sz > 0 for _, _, _, sz in results), "One or more encodes returned empty JPEG"
    # New path must not be catastrophically slower than old path (3× tolerance).
    for label, t_old, t_new, _ in results:
        assert t_new < t_old * 3.0, (
            f"{label}: new path ({t_new:.1f} ms) is >3× slower than old ({t_old:.1f} ms)"
        )

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from app_v2.config.test_config import load_test_config
import importlib

from app_v2.infrastructure.gpu_ring_buffer import GpuPixelFormat
from app_v2.infrastructure.nvdec_decoder import NvdecDecodeConfig, NvdecDecoder
from logger.filtered_logger import warning as log_warning, LogChannel


CONFIG = load_test_config()


def _normalize_stream_url(value: Any) -> str | None:
    url = value or ""
    url = str(url).strip()
    return url if url else None


NVDEC_TEST_STREAM_URL = (
    _normalize_stream_url(os.environ.get("NVDEC_TEST_STREAM_URL"))
    or _normalize_stream_url(CONFIG.get("nvdec_test_stream_url"))
)


@pytest.mark.skipif(
    NVDEC_TEST_STREAM_URL is None,
    reason="Set NVDEC_TEST_STREAM_URL to exercise the real decoder",
)
def test_nvdec_decodes_windows_stream() -> None:
    try:
        importlib.import_module("PyNvCodec")
    except ModuleNotFoundError:
        pytest.skip("PyNvCodec must be installed to exercise the NVDEC decoder")
    try:
        decoder = NvdecDecoder(NVDEC_TEST_STREAM_URL, NvdecDecodeConfig(ring_capacity=4))
    except RuntimeError as exc:
        message = str(exc).lower()
        if "libnvcuvid" in message or "cuda codec" in message or "pyNvdecoder" in message:
            pytest.skip(f"NVDEC decoder unavailable: {exc}")
        raise
    decoder.start()
    try:
        decoder.decode_next_into_ring(frame_id=1, timestamp_ns=int(time.time_ns()))
        popped = decoder.ring.pop_ready(timeout_s=5)
        assert popped is not None
        slot, frame = popped
        decoder.ring.release(slot)
        assert frame.pixel_format == GpuPixelFormat.NV12
        assert frame.device_ptr_y is not None
        assert frame.device_ptr_uv is not None
    except Exception as exc:
        log_warning(LogChannel.NVDEC, f"NVDEC test stream raised {exc}")
        raise
    finally:
        decoder.stop()

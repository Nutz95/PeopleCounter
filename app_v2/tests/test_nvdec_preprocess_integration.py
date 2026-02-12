from __future__ import annotations

import importlib
import os
import time
from typing import Any

import pytest

from app_v2.config.test_config import load_test_config
from app_v2.infrastructure.gpu_preprocessor import GpuPreprocessor
from app_v2.infrastructure.nvdec_decoder import NvdecDecodeConfig, NvdecDecoder


CONFIG = load_test_config()


def _normalize_stream_url(value: Any) -> str | None:
    url = value or ""
    url = str(url).strip()
    return url if url else None


def _resolve_stream_url() -> str | None:
    return (
        _normalize_stream_url(os.environ.get("NVDEC_TEST_STREAM_URL"))
        or _normalize_stream_url(CONFIG.get("nvdec_test_stream_url"))
    )


def _pipeline_like_config() -> dict[str, Any]:
    return {
        "models": {
            "yolo_global": {"enabled": True},
            "yolo_tiles": {"enabled": True},
        },
        "preprocess": {
            "yolo_global": {
                "target_width": 640,
                "target_height": 640,
                "mode": "global",
                "overlap": 0.0,
            },
            "yolo_tiles": {
                "target_width": 640,
                "target_height": 640,
                "mode": "tiles",
                "overlap": 0.2,
            },
        },
    }


def test_nvdec_to_gpu_preprocess_global_and_tiles() -> None:
    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to exercise NVDEC->preprocess integration")
    try:
        importlib.import_module("PyNvCodec")
    except ModuleNotFoundError:
        pytest.skip("PyNvCodec must be installed to exercise NVDEC decoder integration")

    try:
        decoder = NvdecDecoder(stream_url, NvdecDecodeConfig(ring_capacity=4))
    except RuntimeError as exc:
        message = str(exc).lower()
        if "libnvcuvid" in message or "cuda codec" in message or "pynvdecoder" in message:
            pytest.skip(f"NVDEC decoder unavailable: {exc}")
        raise

    preprocessor = GpuPreprocessor()
    preprocessor.configure(_pipeline_like_config())

    decoder.start()
    try:
        decoder.decode_next_into_ring(frame_id=1, timestamp_ns=int(time.time_ns()))
        popped = decoder.ring.pop_ready(timeout_s=5)
        assert popped is not None
        slot, frame = popped

        output = preprocessor.build_output(frame_id=1, frame=frame)

        assert "yolo_global" in output.plans
        assert "yolo_tiles" in output.plans
        assert len(output.model_inputs["yolo_global"]) == 1
        assert len(output.model_inputs["yolo_tiles"]) >= 1

        flattened_global = output.flatten_inputs("yolo_global")
        flattened_tiles = output.flatten_inputs("yolo_tiles")
        flattened_all = output.flatten_inputs()
        assert len(flattened_global) == 1
        assert len(flattened_tiles) >= 1
        assert len(flattened_all) == len(flattened_global) + len(flattened_tiles)

        compat_inputs = preprocessor.process(frame_id=1, frame=frame)
        assert len(compat_inputs) == len(flattened_all)

        decoder.ring.release(slot)
    finally:
        decoder.stop()

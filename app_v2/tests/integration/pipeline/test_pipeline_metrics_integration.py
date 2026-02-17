from __future__ import annotations

import importlib
import os
import time
from typing import Any

import pytest

from app_v2.config.test_config import load_test_config
from app_v2.infrastructure.gpu_preprocessor import GpuPreprocessor
from app_v2.infrastructure.nvdec_decoder import NvdecDecodeConfig, NvdecDecoder
from app_v2.tests.integration.pipeline.perf_budget import evaluate_perf_budget

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
        "fusion_strategy": "ASYNC_OVERLAY",
        "models": {
            "yolo_global": {"enabled": True},
            "yolo_tiles": {"enabled": True},
        },
        "streams": {
            "transfer": 0,
            "yolo": 1,
            "density": 2,
            "yolo_global_preprocess": 3,
            "yolo_tiles_preprocess": 4,
            "density_preprocess": 5,
        },
        "preprocess_branches": {
            "yolo_global_preprocess": True,
            "yolo_tiles_preprocess": True,
            "density_preprocess": True,
        },
        "tensor_pool": {
            "max_per_key": 64,
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


def test_pipeline_metrics_snapshot_includes_stage_timings_and_pool_stats() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch must be installed to exercise pipeline metrics integration")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise pipeline metrics integration")

    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to exercise pipeline metrics integration")

    try:
        importlib.import_module("PyNvCodec")
    except ModuleNotFoundError:
        pytest.skip("PyNvCodec must be installed to exercise pipeline metrics integration")

    decoder = NvdecDecoder(stream_url, NvdecDecodeConfig(ring_capacity=4))
    preprocessor = GpuPreprocessor()
    preprocessor.configure(_pipeline_like_config())

    decoder.start()
    try:
        decoder.decode_next_into_ring(frame_id=1, timestamp_ns=int(time.time_ns()))
        popped = decoder.ring.pop_ready(timeout_s=5)
        assert popped is not None
        slot, frame = popped

        output = preprocessor.build_output(frame_id=1, frame=frame)
        telemetry = output.telemetry
        assert telemetry is not None
        snapshot = telemetry.snapshot()

        required_keys = [
            "frame_id",
            "nvdec_ms",
            "preprocess_ms",
            "preprocess_nv12_bridge_ms",
            "preprocess_model_yolo_global_ms",
            "preprocess_model_yolo_tiles_ms",
            "preprocess_model_sum_ms",
            "preprocess_model_max_ms",
            "preprocess_critical_path_ms",
            "preprocess_serial_overhead_ms",
            "tensor_pool_allocations",
            "tensor_pool_in_use",
            "tensor_pool_wait_ms",
        ]
        for key in required_keys:
            assert key in snapshot, f"Missing telemetry key: {key}"

        for key, value in snapshot.items():
            if key.endswith("_ms"):
                assert float(value) >= 0.0, f"Telemetry timing must be >= 0 for {key}"

        assert "preprocess_stream_model_yolo_global" in snapshot
        assert "preprocess_stream_model_yolo_tiles" in snapshot
        assert snapshot["preprocess_model_max_ms"] <= snapshot["preprocess_model_sum_ms"]
        assert snapshot["preprocess_critical_path_ms"] <= snapshot["preprocess_ms"]

        budget_report = evaluate_perf_budget(snapshot, fusion_strategy=_pipeline_like_config().get("fusion_strategy"))
        if budget_report.mode != "off":
            print("perf_budget_checked:", budget_report.checked)
            print("perf_budget_summary:", budget_report.summary)
            if budget_report.violations:
                print("perf_budget_violations:", budget_report.violations)
            if budget_report.should_fail:
                assert (
                    not budget_report.violations
                ), f"Performance budget exceeded: {budget_report.violations}"

        print("pipeline_metrics_snapshot:", {k: snapshot[k] for k in sorted(snapshot.keys()) if k.endswith("_ms") or k.startswith("tensor_pool_")})

        output.release_all()
        decoder.ring.release(slot)
    finally:
        decoder.stop()

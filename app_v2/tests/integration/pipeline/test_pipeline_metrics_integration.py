from __future__ import annotations

import importlib
import os
import time
from typing import Any

import pytest

from app_v2.config.test_config import load_test_config
from app_v2.application.pipeline_orchestrator import PipelineOrchestrator
from app_v2.core.result_publisher import ResultPublisher
from app_v2.infrastructure.gpu_preprocessor import GpuPreprocessor
from app_v2.infrastructure.nvdec_decoder import NvdecDecodeConfig, NvdecDecoder
from app_v2.infrastructure.rtsp_frame_source import RTSPFrameSource
from app_v2.tests.integration.pipeline.perf_budget import (
    evaluate_perf_budget,
    render_perf_budget_table,
    write_perf_budget_html_report,
)

CONFIG = load_test_config()


class CapturePublisher(ResultPublisher):
    def __init__(self) -> None:
        self.published: list[tuple[int, list[dict[str, Any]]]] = []

    def publish(self, frame_id: int, payload: list[dict[str, Any]]) -> None:
        self.published.append((frame_id, payload))


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

    try:
        decoder = NvdecDecoder(stream_url, NvdecDecodeConfig(ring_capacity=4))
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if (
            "libnvcuvid" in message
            or "cuda codec" in message
            or "pynvdecoder" in message
            or "unsupported ffmpeg pixel format" in message
        ):
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
            print("perf_budget_table:\n" + render_perf_budget_table(budget_report))
            report_file = write_perf_budget_html_report(budget_report)
            print("perf_budget_html_report:", str(report_file))
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


def test_pipeline_e2e_real_stream_includes_inference_timings() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch must be installed to run e2e inference integration")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to run e2e inference integration")

    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to exercise e2e inference integration")

    try:
        importlib.import_module("PyNvCodec")
    except ModuleNotFoundError:
        pytest.skip("PyNvCodec must be installed to run e2e inference integration")

    publisher = CapturePublisher()
    try:
        source = RTSPFrameSource(stream_url)
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if (
            "libnvcuvid" in message
            or "cuda codec" in message
            or "pynvdecoder" in message
            or "unsupported ffmpeg pixel format" in message
        ):
            pytest.skip(f"NVDEC frame source unavailable: {exc}")
        raise

    orchestrator = PipelineOrchestrator(frame_source=source, max_frames=1, publisher=publisher)
    orchestrator.run()

    assert publisher.published, "Pipeline should publish at least one payload"

    telemetry_snapshot: dict[str, Any] | None = None
    yolo_payloads: list[dict[str, Any]] = []
    for _, payload in publisher.published:
        for item in payload:
            if "telemetry" in item and isinstance(item["telemetry"], dict):
                telemetry_snapshot = item["telemetry"]
            if item.get("model") in {"yolo_global", "yolo_tiles"}:
                yolo_payloads.append(item)

    assert telemetry_snapshot is not None, "Expected telemetry in published payload"
    for key in [
        "fusion_wait_ms",
        "overlay_lag_ms",
        "end_to_end_ms",
        "inference_model_sum_ms",
        "inference_model_max_ms",
    ]:
        assert key in telemetry_snapshot, f"Missing e2e telemetry key: {key}"

    assert yolo_payloads, "Expected at least one YOLO payload"
    assert any("yolo_inference_ms" in item for item in yolo_payloads), "Expected yolo_inference_ms in YOLO payload"

    budget_report = evaluate_perf_budget(telemetry_snapshot, fusion_strategy=orchestrator.config.get("fusion_strategy"))
    if budget_report.mode != "off":
        print("e2e_perf_budget_checked:", budget_report.checked)
        print("e2e_perf_budget_summary:", budget_report.summary)
        print("e2e_perf_budget_table:\n" + render_perf_budget_table(budget_report))
        report_file = write_perf_budget_html_report(budget_report)
        print("e2e_perf_budget_html_report:", str(report_file))

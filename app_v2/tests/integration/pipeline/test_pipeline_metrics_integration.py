from __future__ import annotations

import importlib
import os
import time
from typing import Any

import pytest

from app_v2.config.test_config import load_test_config
from app_v2.application.pipeline_orchestrator import PipelineOrchestrator
from app_v2.core.fusion_strategy import FusionStrategy
from app_v2.enums import FusionStrategyType
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


class PublishAfterThree(FusionStrategy):
    def __init__(self) -> None:
        super().__init__(FusionStrategyType.STRICT_SYNC)

    def should_publish(self, frame_id: int, collected_ids: list[int]) -> bool:
        del frame_id
        return len(collected_ids) >= 3

    def merge(self, payloads: list[Any]) -> list[Any]:
        return list(payloads)


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
        "trt_execution": {
            "enabled": True,
            "strict_shape_check": True,
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
        popped = None
        for fid in range(1, 4):
            decoder.decode_next_into_ring(frame_id=fid, timestamp_ns=int(time.time_ns()))
            popped = decoder.ring.pop_ready(timeout_s=5)
            assert popped is not None
            if fid < 3:
                warm_slot, _ = popped
                decoder.ring.release(warm_slot)
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
            report_file = write_perf_budget_html_report(budget_report, report_name="pipeline_preprocess_metrics")
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

    orchestrator = PipelineOrchestrator(
        frame_source=source,
        max_frames=3,
        publisher=publisher,
        fusion_strategy=PublishAfterThree(),
    )
    orchestrator.run()

    assert publisher.published, "Pipeline should publish at least one payload"

    telemetry_snapshot: dict[str, Any] | None = None
    telemetry_frame_id: int = -1
    yolo_payloads: list[dict[str, Any]] = []
    for _, payload in publisher.published:
        for item in payload:
            if "telemetry" in item and isinstance(item["telemetry"], dict):
                current_frame = int(float(item["telemetry"].get("frame_id", -1)))
                if current_frame >= telemetry_frame_id:
                    telemetry_snapshot = item["telemetry"]
                    telemetry_frame_id = current_frame
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
    assert any("inference_ms" in item for item in yolo_payloads), "Expected inference_ms in YOLO payload"

    by_model: dict[str, float] = {}
    for item in yolo_payloads:
        model_name = item.get("model")
        infer_ms = item.get("inference_ms")
        if isinstance(model_name, str) and isinstance(infer_ms, (int, float)):
            by_model[model_name] = max(by_model.get(model_name, 0.0), float(infer_ms))

    telemetry_snapshot["inference_model_yolo_global_ms"] = by_model.get("yolo_global", 0.0)
    telemetry_snapshot["inference_model_yolo_tiles_ms"] = by_model.get("yolo_tiles", 0.0)
    telemetry_snapshot["inference_model_sum_ms"] = sum(by_model.values())
    telemetry_snapshot["inference_model_max_ms"] = max(by_model.values()) if by_model else 0.0

    budget_report = evaluate_perf_budget(telemetry_snapshot, fusion_strategy=orchestrator.config.get("fusion_strategy"))
    if budget_report.mode != "off":
        print("e2e_perf_budget_checked:", budget_report.checked)
        print("e2e_perf_budget_summary:", budget_report.summary)
        print("e2e_perf_budget_table:\n" + render_perf_budget_table(budget_report))
        report_file = write_perf_budget_html_report(budget_report, report_name="pipeline_e2e_metrics")
        print("e2e_perf_budget_html_report:", str(report_file))


def test_pipeline_e2e_real_stream_trt_opt_in_shape_guard() -> None:
    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to exercise TRT opt-in e2e integration")

    previous_value = os.environ.get("PEOPLE_COUNTER_ENABLE_TRT_EXECUTION")
    os.environ["PEOPLE_COUNTER_ENABLE_TRT_EXECUTION"] = "1"
    publisher = CapturePublisher()
    try:
        source = RTSPFrameSource(stream_url)
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "unsupported ffmpeg pixel format" in message or "pynvdecoder" in message:
            pytest.skip(f"NVDEC frame source unavailable: {exc}")
        raise

    try:
        orchestrator = PipelineOrchestrator(
            frame_source=source,
            max_frames=1,
            publisher=publisher,
            fusion_strategy=PublishAfterThree(),
        )
        orchestrator.run()
    finally:
        if previous_value is None:
            os.environ.pop("PEOPLE_COUNTER_ENABLE_TRT_EXECUTION", None)
        else:
            os.environ["PEOPLE_COUNTER_ENABLE_TRT_EXECUTION"] = previous_value

    assert publisher.published, "Expected at least one publish in TRT opt-in run"
    flattened = [entry for _, payload in publisher.published for entry in payload if isinstance(entry, dict)]
    model_entries = [entry for entry in flattened if entry.get("model") in {"yolo_global", "yolo_tiles"}]
    assert model_entries, "Expected YOLO entries in TRT opt-in run"
    for entry in model_entries:
        prediction = entry.get("prediction")
        if isinstance(prediction, dict):
            status = prediction.get("status")
            assert status in {"ok", "gpu_unavailable"}

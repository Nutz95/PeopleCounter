from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any

from app_v2.application.result_aggregator import ResultAggregator
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.fusion_strategy import FusionStrategy, SimpleFusionStrategy
from app_v2.enums import FusionStrategyType
from app_v2.core.result_publisher import ResultPublisher


class DummyPublisher(ResultPublisher):
    def __init__(self) -> None:
        self.published: list[tuple[int, Sequence[dict[str, Any]]]] = []

    def publish(self, frame_id: int, payload: Sequence[dict[str, Any]]) -> None:
        self.published.append((frame_id, payload))


class PublishAfterTwoStrategy(FusionStrategy):
    def __init__(self) -> None:
        super().__init__(FusionStrategyType.ASYNC_OVERLAY)

    def should_publish(self, frame_id: int, collected_ids: Sequence[int]) -> bool:
        del frame_id
        return len(collected_ids) >= 2

    def merge(self, payloads: Sequence[Any]) -> Any:
        return list(payloads)


def test_result_aggregator_appends_telemetry() -> None:
    frame_id = 7
    publisher = DummyPublisher()
    aggregator = ResultAggregator(SimpleFusionStrategy(), publisher)

    telemetry = FrameTelemetry(frame_id=frame_id)
    telemetry.mark_stage_start("preprocess")
    telemetry.mark_stage_end("preprocess")

    aggregator.attach_telemetry(frame_id, telemetry)
    aggregator.collect(frame_id, {"model": "yolo_global"})

    assert publisher.published, "Aggregator should publish at least one payload"
    _, payload = publisher.published[0]
    assert any("telemetry" in entry for entry in payload)


def test_result_aggregator_runs_release_hooks_on_publish() -> None:
    frame_id = 11
    publisher = DummyPublisher()
    aggregator = ResultAggregator(SimpleFusionStrategy(), publisher)

    released = {"count": 0}

    def _release() -> None:
        released["count"] += 1

    aggregator.attach_release_hook(frame_id, _release)
    aggregator.collect(frame_id, {"model": "yolo_global"})

    assert released["count"] == 1


def test_result_aggregator_adds_fusion_lag_and_end_to_end_metrics() -> None:
    frame_id = 21
    publisher = DummyPublisher()
    aggregator = ResultAggregator(PublishAfterTwoStrategy(), publisher)

    telemetry = FrameTelemetry(frame_id=frame_id)
    telemetry.add_metrics({"frame_timestamp_ns": int(time.time_ns()) - 50_000_000})
    aggregator.attach_telemetry(frame_id, telemetry)

    now_ns = int(time.time_ns())
    aggregator.collect(
        frame_id,
        {"model": "yolo_global", "_inference_done_ns": now_ns - 3_000_000, "yolo_inference_ms": 12.5},
    )
    aggregator.collect(frame_id, {"model": "yolo_tiles", "_inference_done_ns": now_ns, "yolo_inference_ms": 8.0})

    assert publisher.published, "Aggregator should publish payload with telemetry"
    _, payload = publisher.published[-1]
    telemetry_entries = [entry for entry in payload if "telemetry" in entry]
    assert telemetry_entries, "Telemetry entry should be present"
    snapshot = telemetry_entries[0]["telemetry"]
    assert "fusion_wait_ms" in snapshot
    assert "overlay_lag_ms" in snapshot
    assert "end_to_end_ms" in snapshot
    assert "inference_model_yolo_global_ms" in snapshot
    assert "inference_model_yolo_tiles_ms" in snapshot
    assert "inference_model_sum_ms" in snapshot
    assert "inference_model_max_ms" in snapshot
    assert float(snapshot["fusion_wait_ms"]) >= 0.0
    assert float(snapshot["overlay_lag_ms"]) >= 0.0
    assert float(snapshot["end_to_end_ms"]) >= 0.0
    assert float(snapshot["inference_model_sum_ms"]) >= float(snapshot["inference_model_max_ms"])
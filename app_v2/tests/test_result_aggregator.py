from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app_v2.application.result_aggregator import ResultAggregator
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.fusion_strategy import SimpleFusionStrategy
from app_v2.core.result_publisher import ResultPublisher


class DummyPublisher(ResultPublisher):
    def __init__(self) -> None:
        self.published: list[tuple[int, Sequence[dict[str, Any]]]] = []

    def publish(self, frame_id: int, payload: Sequence[dict[str, Any]]) -> None:
        self.published.append((frame_id, payload))


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
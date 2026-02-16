from __future__ import annotations

from time import sleep

from app_v2.core.frame_telemetry import FrameTelemetry


def test_frame_telemetry_records_stage_durations() -> None:
    telemetry = FrameTelemetry(frame_id=99, stream_name="yolo")
    telemetry.mark_stage_start("decode")
    sleep(0.001)
    telemetry.mark_stage_end("decode")

    snapshot = telemetry.snapshot()

    assert snapshot["frame_id"] == float(99)
    assert snapshot["stream_name"] == "yolo"
    assert snapshot["decode_ms"] >= 0.0
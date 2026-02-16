from __future__ import annotations

from app_v2.application.performance_tracker import PerformanceTracker


def test_performance_tracker_records_stages() -> None:
    tracker = PerformanceTracker()
    frame_id = 7

    with tracker.stage(frame_id, "capture"):
        pass

    summary = tracker.get_summary(frame_id)
    assert "capture" in summary

    tracker.clear(frame_id)
    assert tracker.get_summary(frame_id) == {}

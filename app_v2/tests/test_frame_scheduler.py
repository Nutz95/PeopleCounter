from __future__ import annotations

from app_v2.application.frame_scheduler import FrameScheduler


def test_frame_scheduler_assigns_sequential_ids() -> None:
    scheduler = FrameScheduler()
    first_id = scheduler.schedule(object())
    second_id = scheduler.schedule(object())

    assert first_id == 1
    assert second_id == 2

    scheduler.acknowledge(first_id)
    scheduler.acknowledge(first_id)

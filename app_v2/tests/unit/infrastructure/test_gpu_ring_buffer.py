from __future__ import annotations

from collections import Counter

from app_v2.infrastructure.gpu_ring_buffer import GpuFrame, GpuPixelFormat, GpuRingBuffer


def test_gpu_ring_buffer_producer_consumer_cycle() -> None:
    ring = GpuRingBuffer(capacity=2)

    slot = ring.acquire(block=False)
    assert slot in (0, 1)

    ring.commit(
        slot,
        GpuFrame(width=1920, height=1080, pixel_format=GpuPixelFormat.NV12, frame_id=1),
    )

    popped = ring.pop_ready(block=False)
    assert popped is not None
    popped_slot, frame = popped
    assert popped_slot == slot
    assert frame.frame_id == 1
    assert frame.is_nv12()

    ring.release(popped_slot)

    # Slot becomes available again
    assert ring.acquire(block=False) is not None


def test_gpu_ring_buffer_long_run_recycles_slots_without_starvation() -> None:
    ring = GpuRingBuffer(capacity=4)
    iterations = 2000
    slot_usage: Counter[int] = Counter()

    for frame_id in range(iterations):
        slot = ring.acquire(block=False)
        assert slot is not None
        slot_usage[slot] += 1

        ring.commit(
            slot,
            GpuFrame(width=1280, height=720, pixel_format=GpuPixelFormat.NV12, frame_id=frame_id),
        )

        popped = ring.pop_ready(block=False)
        assert popped is not None
        popped_slot, frame = popped
        assert frame.frame_id == frame_id
        ring.release(popped_slot)

    assert len(slot_usage) == ring.capacity
    min_usage = min(slot_usage.values())
    assert min_usage > 0

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Condition
from typing import Deque, Tuple


class GpuPixelFormat(str, Enum):
    """Pixel formats we may encounter in the GPU-only pipeline."""

    NV12 = "NV12"
    RGB = "RGB"
    BGR = "BGR"


@dataclass(frozen=True, slots=True)
class GpuFrame:
    """Metadata describing a GPU-resident decoded frame.

    Notes:
    - For NVDEC the natural output is NV12 in device memory.
    - `device_ptr_y` and `device_ptr_uv` are device pointers (ints) to the planes.
    - No CPU ndarray is stored here by design.
    """

    width: int
    height: int
    pixel_format: GpuPixelFormat

    device_ptr_y: int | None = None
    device_ptr_uv: int | None = None
    pitch: int | None = None

    timestamp_ns: int | None = None
    frame_id: int | None = None

    def is_nv12(self) -> bool:
        return self.pixel_format == GpuPixelFormat.NV12


class GpuRingBuffer:
    """Bounded ring buffer for GPU frames.

    Producer (NVDEC) acquires a free slot, commits a `GpuFrame`.
    Consumer (scheduler/infer) pops ready frames and releases slots.

    This provides backpressure without ever forcing GPU->CPU downloads.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self._capacity = capacity
        self._cond = Condition()
        self._free: Deque[int] = deque(range(capacity))
        self._ready: Deque[int] = deque()
        self._slots: list[GpuFrame | None] = [None] * capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    def acquire(self, *, block: bool = True, timeout_s: float | None = None) -> int | None:
        """Reserve a slot index for the producer."""
        with self._cond:
            if not block and not self._free:
                return None
            if not self._free:
                self._cond.wait(timeout=timeout_s)
            if not self._free:
                return None
            return self._free.popleft()

    def commit(self, slot: int, frame: GpuFrame) -> None:
        """Publish a produced frame into the ring."""
        with self._cond:
            self._slots[slot] = frame
            self._ready.append(slot)
            self._cond.notify_all()

    def pop_ready(self, *, block: bool = True, timeout_s: float | None = None) -> Tuple[int, GpuFrame] | None:
        """Pop a ready slot and its frame for consumption."""
        with self._cond:
            if not block and not self._ready:
                return None
            if not self._ready:
                self._cond.wait(timeout=timeout_s)
            if not self._ready:
                return None
            slot = self._ready.popleft()
            frame = self._slots[slot]
            if frame is None:
                return None
            return slot, frame

    def release(self, slot: int) -> None:
        """Return a slot to the producer."""
        with self._cond:
            self._slots[slot] = None
            self._free.append(slot)
            self._cond.notify_all()

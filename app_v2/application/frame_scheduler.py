from __future__ import annotations

from collections import deque
from typing import Any, Deque


class FrameScheduler:
    """Assigns monotonic frame IDs and maintains an in-flight window."""

    def __init__(self) -> None:
        self._next_id = 1
        self._pending: Deque[int] = deque()

    def schedule(self, frame: Any) -> int:
        """Return a new frame_id for the incoming frame payload."""
        frame_id = self._next_id
        self._next_id += 1
        self._pending.append(frame_id)
        return frame_id

    def acknowledge(self, frame_id: int) -> None:
        """Mark frame_id as delivered so the window can advance."""
        if self._pending and self._pending[0] == frame_id:
            self._pending.popleft()

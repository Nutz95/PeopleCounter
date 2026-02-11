from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FrameSource(ABC):
    """Abstraction for any frame-producing entity that drives a frame_id-aware pipeline."""

    @abstractmethod
    def connect(self) -> None:
        """Open the camera/stream and prepare internal buffers."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release any hardware handles held by the source."""

    @abstractmethod
    def next_frame(self) -> tuple[int, Any]:
        """Return the next (frame_id, payload) tuple or raise StopIteration when done."""

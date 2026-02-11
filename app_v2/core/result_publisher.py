from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class ResultPublisher(ABC):
    """Publishes results to the Flask stream or other downstream systems."""

    @abstractmethod
    def publish(self, frame_id: int, payload: Sequence[Any]) -> None:
        """Push aggregated data for frame_id to the external services."""

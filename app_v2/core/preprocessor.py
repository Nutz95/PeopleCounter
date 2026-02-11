from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class Preprocessor(ABC):
    """Defines GPU-first preprocessing steps before inference."""

    @abstractmethod
    def configure(self, metadata: dict[str, Any]) -> None:
        """Set up size, tiling, or normalization metadata required for the next batch."""

    @abstractmethod
    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        """Produce GPU buffers that will be consumed by the inference model."""

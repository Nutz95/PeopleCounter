from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from app_v2.core.preprocessor_types import PreprocessOutput

class Preprocessor(ABC):
    """Defines GPU-first preprocessing steps before inference."""

    @abstractmethod
    def configure(self, metadata: dict[str, Any]) -> None:
        """Set up size, tiling, or normalization metadata required for the next batch."""

    @abstractmethod
    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        """Produce GPU buffers that will be consumed by the inference model."""

    @abstractmethod
    def build_output(self, frame_id: int, frame: Any) -> "PreprocessOutput":
        """Return a PreprocessOutput so callers can access plans and telemetry."""

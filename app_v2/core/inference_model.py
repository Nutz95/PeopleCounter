from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class InferenceModel(ABC):
    """Contract for a GPU-based inference engine tied to a CUDA stream."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (e.g., yolo_tiles, density)."""

    @property
    @abstractmethod
    def stream_id(self) -> int:
        """Return the CUDA stream slot assigned to this model."""

    @abstractmethod
    def warm_up(self, batch_size: int) -> None:
        """Warm up TensorRT context for the configured batch size."""

    @abstractmethod
    def infer(self, frame_id: int, inputs: Sequence[Any]) -> dict[str, Any]:
        """Execute inference and return a dictionary of intermediate tensors."""

    @abstractmethod
    def close(self) -> None:
        """Destroy TensorRT contexts and release device memory."""

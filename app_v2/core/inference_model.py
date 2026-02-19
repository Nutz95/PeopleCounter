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
    def infer(self, frame_id: int, inputs: Sequence[Any], *, preprocess_events: Sequence[Any] | None = None) -> dict[str, Any]:
        """Execute inference and return a dictionary of intermediate tensors.

        Args:
            preprocess_events: Optional CUDA events recorded on preprocess
                streams.  The inference stream must call ``wait_event()`` on
                each before touching preprocess output tensors, establishing a
                GPU-side ordering dependency without blocking the CPU.
        """

    @abstractmethod
    def close(self) -> None:
        """Destroy TensorRT contexts and release device memory."""

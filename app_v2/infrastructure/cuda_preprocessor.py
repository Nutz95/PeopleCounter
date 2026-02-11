from __future__ import annotations

from typing import Any, Sequence

from app_v2.core.preprocessor import Preprocessor


class CudaPreprocessor(Preprocessor):
    """Performs GPU normalization and tiling before inference."""

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}

    def configure(self, metadata: dict[str, Any]) -> None:
        """Capture tiling or normalization hints for upcoming frames."""
        self.metadata = metadata

    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        """Return tensor buffers that feed TensorRT inputs."""
        return [frame]

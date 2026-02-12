from __future__ import annotations

from typing import Any, Sequence

from app_v2.infrastructure.gpu_preprocessor import GpuPreprocessor


class CudaPreprocessor(GpuPreprocessor):
    """Backward-compatible alias for the new GPU preprocessor implementation."""

    def configure(self, metadata: dict[str, Any]) -> None:
        super().configure(metadata)

    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        return super().process(frame_id, frame)

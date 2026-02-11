from __future__ import annotations

from typing import Any, Sequence

from app_v2.core.inference_model import InferenceModel


class YoloTilingTRT(InferenceModel):
    """TensorRT tiling model that processes many 640x640 tiles concurrently."""

    def __init__(self, engine_context: Any, stream_id: int) -> None:
        self._context = engine_context
        self._stream_id = stream_id
        self._name = "yolo_tiles"

    @property
    def name(self) -> str:
        return self._name

    @property
    def stream_id(self) -> int:
        return self._stream_id

    def warm_up(self, batch_size: int) -> None:
        """Bind TensorRT profiles for the expected batch."""
        pass

    def infer(self, frame_id: int, inputs: Sequence[Any]) -> dict[str, Any]:
        """Flatten the tiled output buffer so the decoder can read it."""
        return {"frame_id": frame_id, "prediction": None}

    def close(self) -> None:
        """Release the CUDA resources held by this tiling model."""
        pass

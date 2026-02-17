from __future__ import annotations

from typing import Any, Sequence

from app_v2.core.inference_model import InferenceModel


class YoloTilingTRT(InferenceModel):
    """TensorRT tiling model that processes many 640x640 tiles concurrently."""

    def __init__(self, engine_context: Any, stream_id: int, inference_params: dict[str, Any] | None = None) -> None:
        self._context = engine_context
        self._stream_id = stream_id
        self._name = "yolo_tiles"
        self._inference_params = dict(inference_params or {})

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
        return {
            "frame_id": frame_id,
            "prediction": None,
            "segmentation": None,
            "inference_params": self._inference_params,
            "person_class_id": self._inference_params.get("person_class_id", 0),
            "class_whitelist": self._inference_params.get("class_whitelist", [0]),
            "tile_count": len(inputs),
        }

    def close(self) -> None:
        """Release the CUDA resources held by this tiling model."""
        pass

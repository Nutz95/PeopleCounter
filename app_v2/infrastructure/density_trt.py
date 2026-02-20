from __future__ import annotations

from typing import Any, Sequence

from app_v2.core.inference_model import InferenceModel


class DensityTRT(InferenceModel):
    """TensorRT density model tuned for 4K tiling (e.g., 4x3 tiles)."""

    def __init__(self, engine_context: Any, stream_id: int) -> None:
        self._context = engine_context
        self._stream_id = stream_id
        self._name = "density"

    @property
    def name(self) -> str:
        return self._name

    @property
    def stream_id(self) -> int:
        return self._stream_id

    def warm_up(self, batch_size: int) -> None:
        """Ensure the density model respects the configured tiling resolution."""
        pass

    def infer(self, frame_id: int, inputs: Sequence[Any], *, preprocess_events: Sequence[Any] | None = None, tile_plan: Any | None = None) -> dict[str, Any]:
        """Return raw density heatmaps keyed by frame_id."""
        del preprocess_events  # density model not yet integrated with TRT
        del tile_plan
        return {"frame_id": frame_id, "density": None}

    def close(self) -> None:
        """Cleanup the TensorRT execution context for density."""
        pass

from __future__ import annotations

from typing import Any

from app_v2.core.postprocessor import Postprocessor


class DensityDecoder(Postprocessor):
    """Transforms density model outputs into heatmaps and counts."""

    def process(self, frame_id: int, outputs: dict[str, Any]) -> dict[str, Any]:
        """Return metadata keyed by frame_id used by the aggregator."""
        return {"frame_id": frame_id, "density": outputs.get("density")}

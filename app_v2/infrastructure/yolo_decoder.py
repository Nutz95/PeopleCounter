from __future__ import annotations

from typing import Any, Sequence

from app_v2.core.postprocessor import Postprocessor


class YoloDecoder(Postprocessor):
    """Decodes YOLO outputs into boxes, scores, and optional masks."""

    def __init__(self, person_class_id: int = 0) -> None:
        self.person_class_id = person_class_id

    def process(self, frame_id: int, outputs: dict[str, Any]) -> dict[str, Any]:
        """Return normalized bounding boxes plus metadata for the aggregator."""
        return {"frame_id": frame_id, "detections": [], "person_class_id": self.person_class_id}

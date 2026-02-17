from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from app_v2.core.postprocessor import Postprocessor


class YoloDecoder(Postprocessor):
    """Decodes YOLO outputs into boxes, scores, and optional masks."""

    def __init__(self, person_class_id: int = 0) -> None:
        self.person_class_id = person_class_id

    def process(self, frame_id: int, outputs: dict[str, Any]) -> dict[str, Any]:
        """Return normalized bounding boxes plus metadata for the aggregator."""
        tensors = outputs.get("output_tensors", []) if isinstance(outputs, dict) else []
        detections = self._decode_person_detections(tensors)
        return {
            "frame_id": frame_id,
            "detections": detections,
            "person_class_id": self.person_class_id,
            "output_count": len(tensors),
        }

    def _decode_person_detections(self, tensors: Sequence[Any]) -> list[dict[str, float | int]]:
        if torch is None:
            return []
        if not tensors:
            return []
        first = tensors[0]
        if not isinstance(first, torch.Tensor):
            return []

        with torch.no_grad():
            cpu = first.detach().float().cpu()
            rows = self._to_rows(cpu)
            detections: list[dict[str, float | int]] = []
            for row in rows:
                if row.numel() < 6:
                    continue
                x1, y1, x2, y2 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                conf = float(row[4])
                cls_id = int(row[5])
                if cls_id != self.person_class_id:
                    continue
                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                        "class_id": cls_id,
                    }
                )
                if len(detections) >= 200:
                    break
            return detections

    @staticmethod
    def _to_rows(tensor: Any) -> Any:
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[-1] >= 6:
            return tensor.reshape(-1, tensor.shape[-1])
        if tensor.ndim == 2 and tensor.shape[-1] >= 6:
            return tensor
        if tensor.ndim == 2 and tensor.shape[0] >= 6:
            return tensor.transpose(0, 1)
        if tensor.ndim == 3 and tensor.shape[0] >= 6:
            return tensor.permute(1, 2, 0).reshape(-1, tensor.shape[0])
        return tensor.reshape(-1, max(1, tensor.shape[-1]))

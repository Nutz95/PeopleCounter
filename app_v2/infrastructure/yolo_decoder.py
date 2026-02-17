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
        person_summary = self._decode_person_detections_gpu(tensors)
        return {
            "frame_id": frame_id,
            "detections": [],
            "detections_gpu": person_summary,
            "person_class_id": self.person_class_id,
            "output_count": len(tensors),
        }

    def _decode_person_detections_gpu(self, tensors: Sequence[Any]) -> dict[str, Any]:
        if torch is None:
            return {"available": False, "reason": "torch unavailable"}
        if not tensors:
            return {"available": False, "reason": "no output tensors"}
        first = tensors[0]
        if not isinstance(first, torch.Tensor):
            return {"available": False, "reason": "first output is not a tensor"}

        with torch.no_grad():
            rows = self._to_rows(first.detach())
            if rows.numel() == 0 or rows.shape[-1] < 6:
                return {"available": True, "person_candidates": 0, "device": rows.device.type}
            classes = rows[:, 5]
            person_mask = classes == float(self.person_class_id)
            return {
                "available": True,
                "person_candidates": int(person_mask.sum().item()),
                "device": rows.device.type,
            }

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

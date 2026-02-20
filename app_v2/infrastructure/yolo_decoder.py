from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from app_v2.core.postprocessor import Postprocessor


class YoloDecoder(Postprocessor):
    """Decodes YOLO outputs into boxes, scores, and optional masks."""

    def __init__(self, person_class_id: int = 0, confidence_threshold: float = 0.25) -> None:
        self.person_class_id = person_class_id
        self.confidence_threshold = confidence_threshold

    def process(self, frame_id: int, outputs: dict[str, Any]) -> dict[str, Any]:
        """Return normalized bounding boxes plus metadata for the aggregator."""
        tensors = outputs.get("output_tensors", []) if isinstance(outputs, dict) else []
        detections = self._decode_detections(tensors)
        person_summary = self._decode_person_detections_gpu(tensors)
        return {
            "frame_id": frame_id,
            "detections": detections,
            "detections_gpu": person_summary,
            "person_class_id": self.person_class_id,
            "output_count": len(tensors),
        }

    def _decode_detections(self, tensors: Sequence[Any]) -> list[dict[str, Any]]:
        """Decode YOLO tensor output into a list of normalized bbox dicts.

        Expects the first output tensor in post-NMS format: ``[N×6]`` or
        ``[1×N×6]`` where each row is ``[x1, y1, x2, y2, conf, class_id]``.
        Coordinates are expected in the model's input pixel space (640×640 by
        default) and are normalised to ``[0, 1]`` before being returned.
        A heuristic detects whether values are already normalised: if the max
        coordinate > 2.0 they are assumed to be pixel values and divided by 640.
        """
        if torch is None or not tensors:
            return []
        first = tensors[0]
        if not isinstance(first, torch.Tensor):
            return []

        with torch.no_grad():
            rows = self._to_rows(first.detach().float())
            if rows.numel() == 0 or rows.shape[-1] < 6:
                return []

            confs = rows[:, 4]
            classes = rows[:, 5]
            mask = (confs >= self.confidence_threshold) & (
                classes == float(self.person_class_id)
            )
            rows = rows[mask]
            if rows.shape[0] == 0:
                return []

            coords = rows[:, :4].cpu()
            # Normalise: if max coord > 2 assume pixel-space (model input = 640)
            if float(coords.max()) > 2.0:
                coords = coords / 640.0
            coords = coords.clamp(0.0, 1.0)

            result: list[dict[str, Any]] = []
            for i in range(rows.shape[0]):
                x1, y1, x2, y2 = coords[i].tolist()
                result.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(float(rows[i, 4]), 4),
                    "label": "person",
                })
        return result

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

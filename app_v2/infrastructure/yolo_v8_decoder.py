"""YOLOv8 decoder — concrete implementation of YoloDecoderBase.

YOLOv8 raw output format:
  Single tensor: ``[1, 4+C, 8400]``  or  ``[1, 4+C+32, 8400]`` (seg).
  Batched tiles: ``[N, 4+C, 8400]``.
  Person score  = ``tensor[..., 4 + person_class_id]``  (direct class confidence,
  no separate objectness term unlike YOLOv5).
"""
from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from app_v2.infrastructure.yolo_decoder_base import YoloDecoderBase


class YoloV8Decoder(YoloDecoderBase):
    """YOLO v8 output decoder."""

    # ── Abstract method implementations ──────────────────────────────────────

    def _compute_person_scores_tiled(
        self, t: Any, n: int, a: int, c: int
    ) -> Any | None:
        """Return ``[N, A]`` person scores: direct class-score column 4+person_class_id."""
        if self.person_class_id >= c - 4:
            return None
        return t[:, :, 4 + self.person_class_id]   # [N, A]

    def _decode_raw(self, rows: Any) -> list[dict[str, Any]]:
        return self._decode_raw_yolov8(rows)

    # ── Format-specific single-tensor decode ─────────────────────────────────

    def _decode_raw_yolov8(self, rows: Any) -> list[dict[str, Any]]:
        """Decode raw YOLOv8 ``[N, 4+C]`` tensor (no NMS applied by the engine).

        Standard ultralytics ONNX/TRT export (opset 12, no baked NMS plugin)
        outputs each anchor as ``[cx, cy, w, h, score_c0, ..., score_cN]`` where
        cx/cy/w/h are in absolute **pixel** coordinates within the 640×640
        letterboxed model input.  This is **xywh center** format, *not* xyxy.
        """
        if torch is None:
            return []
        num_classes = rows.shape[-1] - 4
        if self.person_class_id >= num_classes:
            return []
        person_scores = rows[:, 4 + self.person_class_id]
        mask    = person_scores >= self.confidence_threshold
        rows_f  = rows[mask]
        scores_f = person_scores[mask]
        if rows_f.shape[0] == 0:
            return []

        cx = rows_f[:, 0];  cy = rows_f[:, 1]
        w  = rows_f[:, 2];  h  = rows_f[:, 3]
        lx1 = cx - w / 2.0;  ly1 = cy - h / 2.0
        lx2 = cx + w / 2.0;  ly2 = cy + h / 2.0

        try:
            from torchvision.ops import nms as _nms  # type: ignore[import]
            boxes_nms = torch.stack([lx1, ly1, lx2, ly2], dim=1).float()
            keep = _nms(boxes_nms, scores_f.float(), iou_threshold=0.45)
            lx1, ly1, lx2, ly2 = lx1[keep], ly1[keep], lx2[keep], ly2[keep]
            scores_f = scores_f[keep]
        except Exception:
            pass

        x1 = (lx1 / 640.0).clamp(0.0, 1.0)
        y1 = (ly1 / 640.0).clamp(0.0, 1.0)
        x2 = (lx2 / 640.0).clamp(0.0, 1.0)
        y2 = (ly2 / 640.0).clamp(0.0, 1.0)

        coords_list = torch.stack([x1, y1, x2, y2], dim=1).tolist()
        confs_list  = scores_f.tolist()
        return [
            {"bbox": [bx1, by1, bx2, by2], "conf": round(c, 4), "label": "person"}
            for (bx1, by1, bx2, by2), c in zip(coords_list, confs_list)
        ]

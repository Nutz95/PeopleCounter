"""YOLOv5 decoder — concrete implementation of YoloDecoderBase.

YOLOv5 (YOLO-CROWD) raw output format:
  Single tensor: ``[1, nc+5, N_anchors]``  (strides 4/8/16 → 100800 total anchors).
  Batched tiles: ``[N_tiles, nc+5, N_anchors]``.

  Layout per anchor: ``[cx_px, cy_px, w_px, h_px, obj_conf, cls0_conf, ..., clsN_conf]``
  Person score = ``obj_conf × cls_conf``  (two-stage confidence unlike YOLOv8).

  Minimum box filter (``min_box_px``): the stride-4 P2 head can fire on sub-pixel
  texture blobs; boxes smaller than ``min_box_px`` in both width and height are
  discarded before NMS.
"""
from __future__ import annotations

import os as _os
import sys as _sys
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from app_v2.infrastructure.yolo_decoder_base import YoloDecoderBase

_CROWD_DBG = lambda: _os.environ.get("CROWD_DBG") == "1"  # noqa: E731


class YoloV5Decoder(YoloDecoderBase):
    """YOLO v5 / YOLO-CROWD output decoder."""

    # ── Abstract method implementations ──────────────────────────────────────

    def _compute_person_scores_tiled(
        self, t: Any, n: int, a: int, c: int
    ) -> Any | None:
        """Return ``[N, A]`` person scores: ``obj_conf × cls_conf``."""
        if c < 6 or self.person_class_id >= c - 5:
            return None
        obj_conf      = t[:, :, 4]                          # [N, A]
        cls_conf      = t[:, :, 5 + self.person_class_id]   # [N, A]
        person_scores = obj_conf * cls_conf                  # [N, A]

        if _CROWD_DBG():
            _obj_max  = float(obj_conf.max().item())
            _cls_max  = float(cls_conf.max().item())
            _raw_pass = int((person_scores >= self.confidence_threshold).sum().item())
            print(
                f"[CROWD-DBG] tiles={n} shape={tuple(t.shape)} "
                f"thresh={self.confidence_threshold} "
                f"obj_max={_obj_max:.3f} cls_max={_cls_max:.3f} "
                f"raw_pass={_raw_pass}/{n * a}",
                file=_sys.stderr, flush=True,
            )
        return person_scores

    def _decode_raw(self, rows: Any) -> list[dict[str, Any]]:
        return self._decode_raw_yolov5(rows)

    # ── Optional hook overrides ───────────────────────────────────────────────

    def _apply_post_conf_size_filter(
        self,
        t_flat: Any,
        scores_flat: Any,
        tile_ids: Any,
    ) -> tuple[Any, Any, Any]:
        """Filter out detections whose bounding box is smaller than ``min_box_px``.

        YOLO-CROWD's stride-4 P2/4 head fires on very small texture blobs.
        ``min_box_px`` (default 16 px, configurable via ``model_inference.yaml``)
        matches the smallest P2/4 anchor height ([19.2, 16]) — anything below
        cannot be a real anchor detection.
        """
        _min_px  = float(self.min_box_px)
        w        = t_flat[:, 2]
        h        = t_flat[:, 3]
        _size_ok = (w >= _min_px) & (h >= _min_px)
        if not bool(_size_ok.all()):
            t_flat      = t_flat[_size_ok]
            scores_flat = scores_flat[_size_ok]
            tile_ids    = tile_ids[_size_ok]
        return t_flat, scores_flat, tile_ids

    def _debug_after_tile_nms(
        self,
        n_after_conf: int,
        boxes: Any,
        nms_ok: bool,
        w: Any,
        h: Any,
    ) -> None:
        if not _CROWD_DBG():
            return
        w_min = float(w.min().item()) if w.shape[0] > 0 else 0.0
        w_max = float(w.max().item()) if w.shape[0] > 0 else 0.0
        h_min = float(h.min().item()) if h.shape[0] > 0 else 0.0
        h_max = float(h.max().item()) if h.shape[0] > 0 else 0.0
        print(
            f"[CROWD-DBG] after_conf={n_after_conf} "
            f"after_tile_nms={int(boxes.shape[0])} nms_ok={nms_ok} "
            f"w_range=[{w_min:.1f},{w_max:.1f}] "
            f"h_range=[{h_min:.1f},{h_max:.1f}]",
            file=_sys.stderr, flush=True,
        )

    def _debug_after_cross_nms(self, before: int, after: int) -> None:
        if _CROWD_DBG():
            print(
                f"[CROWD-DBG] before_cross_nms={before} final={after}",
                file=_sys.stderr, flush=True,
            )

    # ── Format-specific single-tensor decode ─────────────────────────────────

    def _decode_raw_yolov5(self, rows: Any) -> list[dict[str, Any]]:
        """Decode pre-decoded YOLOv5 anchor tensor ``[N, nc+5]``.

        YOLOv5 Detect layer (grid pre-baked) output format::

            [cx_px, cy_px, w_px, h_px, obj_conf, cls0_conf, ..., clsN_conf]

        Coordinates are already in pixel space (0..640, sigmoid + stride applied).
        Final detection score = ``obj_conf × cls_conf``.
        """
        if torch is None:
            return []
        num_classes = rows.shape[-1] - 5
        if self.person_class_id >= num_classes:
            return []

        obj_conf      = rows[:, 4]
        cls_conf      = rows[:, 5 + self.person_class_id]
        person_scores = obj_conf * cls_conf

        if _CROWD_DBG():
            _raw_pass = int((person_scores >= self.confidence_threshold).sum().item())
            print(
                f"[CROWD-DBG] global rows={rows.shape[0]} thresh={self.confidence_threshold} "
                f"obj_max={float(obj_conf.max().item()):.3f} "
                f"cls_max={float(cls_conf.max().item()):.3f} "
                f"raw_pass={_raw_pass}",
                file=_sys.stderr, flush=True,
            )

        mask     = person_scores >= self.confidence_threshold
        rows_f   = rows[mask]
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

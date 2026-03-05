"""Abstract base class for YOLO decoders.

Shared pipeline logic lives here (post-NMS, tiled decode, seg masks, unletterbox).
Format-specific score extraction is delegated to concrete subclasses via the
Template Method pattern:

  ``_compute_person_scores_tiled``  — extract per-anchor person scores from a
                                      batched tile tensor (abstract).
  ``_decode_raw``                   — decode a single-tensor raw output (abstract).
  ``_apply_post_conf_size_filter``  — optional post-confidence size filter
                                      (default no-op; YoloV5Decoder overrides).
  ``_debug_after_tile_nms``         — optional debug hook after per-tile NMS.
  ``_debug_after_cross_nms``        — optional debug hook after cross-tile NMS.

Concrete implementations:
  ``YoloV8Decoder`` — ``yolo_v8_decoder.py``
  ``YoloV5Decoder`` — ``yolo_v5_decoder.py``
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import]
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from app_v2.core.postprocessor import Postprocessor


class YoloDecoderBase(Postprocessor, ABC):
    """Abstract YOLO output decoder — format-agnostic shared pipeline."""

    def __init__(self, person_class_id: int = 0, confidence_threshold: float = 0.25) -> None:
        self.person_class_id = person_class_id
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold: float = 0.45
        self.cross_nms_iou_threshold: float = 0.20
        self.seg_mask_enabled: bool = False
        self.seg_mask_clip_to_bbox: bool = True
        self.person_summary_enabled: bool = False
        self.min_box_px: float = 16.0

    # ──────────────────────────────────────────────────────────────────────────
    # Abstract interface — subclasses must implement
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def _compute_person_scores_tiled(
        self, t: Any, n: int, a: int, c: int
    ) -> Any | None:
        """Return ``[N, A]`` float person-score tensor from a batched tile [N,A,C] tensor.

        Returns ``None`` when the tensor format is not supported (e.g. wrong
        column count), which causes ``_decode_tiled_global`` to short-circuit.
        """

    @abstractmethod
    def _decode_raw(self, rows: Any) -> list[dict[str, Any]]:
        """Decode a single-tensor raw YOLO output to a list of bbox dicts."""

    # ──────────────────────────────────────────────────────────────────────────
    # Optional hooks — default no-ops, override for extra behaviour
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_post_conf_size_filter(
        self,
        t_flat: Any,
        scores_flat: Any,
        tile_ids: Any,
    ) -> tuple[Any, Any, Any]:
        """Optional size filter applied after confidence threshold gate.

        Default implementation is a no-op.  ``YoloV5Decoder`` overrides this to
        apply the ``min_box_px`` filter that removes sub-pixel noise from the
        YOLO-CROWD stride-4 P2 head.

        Returns ``(t_flat, scores_flat, tile_ids)`` after filtering.
        """
        return t_flat, scores_flat, tile_ids

    def _debug_after_tile_nms(
        self,
        n_after_conf: int,
        boxes: Any,
        nms_ok: bool,
        w: Any,
        h: Any,
    ) -> None:
        """Optional debug hook called after per-tile batched NMS."""

    def _debug_after_cross_nms(self, before: int, after: int) -> None:
        """Optional debug hook called after cross-tile NMS."""

    # ──────────────────────────────────────────────────────────────────────────
    # Shared pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def process(self, frame_id: int, outputs: dict[str, Any], *, tile_plan: Any = None) -> dict[str, Any]:
        """Return normalized bounding boxes plus metadata for the aggregator."""
        tensors = outputs.get("output_tensors", []) if isinstance(outputs, dict) else []
        detections = self._decode_detections(tensors, tile_plan=tile_plan)
        person_summary = (
            self._decode_person_detections_gpu(tensors)
            if self.person_summary_enabled
            else {"available": False, "reason": "disabled"}
        )

        seg_mask_raw: str | None = None
        seg_mask_w: int = 0
        seg_mask_h: int = 0
        is_global_mode = (
            tile_plan is None
            or (
                hasattr(tile_plan, "metadata")
                and isinstance(tile_plan.metadata, dict)
                and tile_plan.metadata.get("mode") == "global"
            )
        )
        is_tiles_mode = (
            tile_plan is not None
            and hasattr(tile_plan, "metadata")
            and isinstance(tile_plan.metadata, dict)
            and tile_plan.metadata.get("mode") == "tiles"
        )
        if (
            self.seg_mask_enabled
            and is_global_mode
            and len(tensors) >= 2
            and torch is not None
            and isinstance(tensors[1], torch.Tensor)
            and tensors[1].ndim == 4
            and tensors[1].shape[0] == 1
        ):
            seg_data = self._decode_seg_mask(tensors, clip_to_bbox=self.seg_mask_clip_to_bbox)
            if seg_data is not None:
                seg_mask_raw, seg_mask_w, seg_mask_h = seg_data
        elif (
            self.seg_mask_enabled
            and is_tiles_mode
            and len(tensors) >= 2
            and torch is not None
            and isinstance(tensors[1], torch.Tensor)
            and tensors[1].ndim == 4
            and tensors[1].shape[0] > 1
        ):
            seg_data = self._decode_seg_mask_tiled(
                tensors, tile_plan, clip_to_bbox=self.seg_mask_clip_to_bbox
            )
            if seg_data is not None:
                seg_mask_raw, seg_mask_w, seg_mask_h = seg_data

        return {
            "frame_id": frame_id,
            "detections": detections,
            "detections_gpu": person_summary,
            "person_class_id": self.person_class_id,
            "output_count": len(tensors),
            "seg_mask_raw": seg_mask_raw,
            "seg_mask_w": seg_mask_w,
            "seg_mask_h": seg_mask_h,
        }

    def _decode_detections(
        self, tensors: Sequence[Any], *, tile_plan: Any = None
    ) -> list[dict[str, Any]]:
        """Dispatch tensor output to either the tiled or the single-tensor path."""
        if torch is None or not tensors:
            return []
        first = tensors[0]
        if not isinstance(first, torch.Tensor):
            return []

        # ── Batched tiles path ────────────────────────────────────────────────
        if (
            tile_plan is not None
            and first.ndim == 3
            and first.shape[0] > 1
            and hasattr(tile_plan, "tasks")
            and isinstance(tile_plan.metadata, dict)
            and tile_plan.metadata.get("mode") == "tiles"
        ):
            return self._decode_tiled_global(first.detach().float(), tile_plan)

        # ── Single-tensor path ────────────────────────────────────────────────
        with torch.no_grad():
            rows = self._to_rows(first.detach().float())
            if rows.numel() == 0:
                return []

            ncols = rows.shape[-1]

            if ncols >= 6 and rows.shape[0] <= 1000:
                dets = self._decode_postnms(rows)
            elif ncols > 4:
                dets = self._decode_raw(rows)
            else:
                return []

        if (
            tile_plan is not None
            and dets
            and hasattr(tile_plan, "frame_width")
            and isinstance(tile_plan.metadata, dict)
            and tile_plan.metadata.get("mode") == "global"
        ):
            dets = self._apply_global_unletterbox(dets, tile_plan)

        return dets

    def _decode_postnms(self, rows: Any) -> list[dict[str, Any]]:
        """Decode post-NMS ``[N, 6]`` tensor."""
        confs   = rows[:, 4]
        classes = rows[:, 5]
        mask = (confs >= self.confidence_threshold) & (classes == float(self.person_class_id))
        rows = rows[mask]
        if rows.shape[0] == 0:
            return []
        coords = rows[:, :4]
        if float(coords.max().item()) > 2.0:
            coords = coords / 640.0
        coords = coords.clamp(0.0, 1.0)
        coords_list = coords.tolist()
        confs_list  = rows[:, 4].tolist()
        return [
            {"bbox": [x1, y1, x2, y2], "conf": round(c, 4), "label": "person"}
            for (x1, y1, x2, y2), c in zip(coords_list, confs_list)
        ]

    def _decode_tiled_global(self, tensor: Any, plan: Any) -> list[dict[str, Any]]:
        """Decode batched tile tensor to global frame coordinates ``[0, 1]``.

        Fully vectorized across all tiles: zero per-tile Python loops, a single
        ``batched_nms`` GPU kernel for per-tile duplicate removal, and a single
        GPU→CPU transfer at the very end.  This reduces GPU→CPU synchronisations
        from O(N_tiles) to O(1) — about 3 syncs regardless of tile count.

        Format-specific behaviour is delegated to abstract hooks:
          - ``_compute_person_scores_tiled``   — score extraction (V8 or V5).
          - ``_apply_post_conf_size_filter``   — size filter (V5 only by default).
          - ``_debug_after_tile_nms``          — optional debug logging.
          - ``_debug_after_cross_nms``         — optional debug logging.
        """
        if torch is None:
            return []
        fw = plan.frame_width
        fh = plan.frame_height
        n_tiles = min(int(tensor.shape[0]), len(plan.tasks))
        if n_tiles == 0:
            return []
        tasks = plan.tasks[:n_tiles]

        with torch.no_grad():
            t = tensor[:n_tiles]

            # ── Normalize to [N, anchors, features] ──────────────────────
            if t.ndim == 3 and t.shape[1] < t.shape[2]:
                t = t.permute(0, 2, 1)

            n, a, c = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])

            # ── Post-NMS shortcut: [N, N_dets, 6+] ──────────────────────
            if a <= 1000 and c >= 6:
                return self._decode_tiled_postnms_vectorized(t, tasks, fw, fh)

            # ── Format-specific person score extraction (HOOK 1) ─────────
            person_scores = self._compute_person_scores_tiled(t, n, a, c)
            if person_scores is None:
                return []

            # ── Flatten + confidence gate ─────────────────────────────────
            tile_ids = (
                torch.arange(n, device=t.device, dtype=torch.long)
                .unsqueeze(1).expand(n, a).reshape(-1)
            )
            t_flat      = t.reshape(-1, c)
            scores_flat = person_scores.reshape(-1)

            keep_conf   = scores_flat >= self.confidence_threshold
            t_flat      = t_flat[keep_conf]        # sync #1
            scores_flat = scores_flat[keep_conf]
            tile_ids    = tile_ids[keep_conf]

            if t_flat.shape[0] == 0:
                return []

            n_after_conf = int(t_flat.shape[0])

            # ── Optional post-confidence size filter (HOOK 2) ────────────
            t_flat, scores_flat, tile_ids = self._apply_post_conf_size_filter(
                t_flat, scores_flat, tile_ids
            )
            if t_flat.shape[0] == 0:
                return []

            # ── xywh → xyxy ──────────────────────────────────────────────
            cx = t_flat[:, 0];  cy = t_flat[:, 1]
            w  = t_flat[:, 2];  h  = t_flat[:, 3]
            boxes = torch.stack(
                [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=1
            )

            # ── Per-tile batched NMS ──────────────────────────────────────
            _nms_ok  = False
            _nms_iou = float(self.nms_iou_threshold)
            try:
                from torchvision.ops import batched_nms as _batched_nms  # type: ignore[import]
                keep = _batched_nms(
                    boxes.float(), scores_flat.float(), tile_ids, iou_threshold=_nms_iou
                )
                boxes       = boxes[keep]
                scores_flat = scores_flat[keep]
                tile_ids    = tile_ids[keep]
                _nms_ok = True
            except Exception as _exc:
                import sys as _sys
                print(f"[YOLO-DBG] per-tile NMS failed: {_exc}", file=_sys.stderr, flush=True)

            # Optional debug hook (HOOK 3)
            self._debug_after_tile_nms(n_after_conf, boxes, _nms_ok, w, h)

            # ── Normalize + global remap ──────────────────────────────────
            boxes_norm = (boxes / 640.0).clamp(0.0, 1.0)
            dev = t.device
            tile_ox = torch.tensor([task.source_x      for task in tasks], dtype=torch.float32, device=dev)
            tile_oy = torch.tensor([task.source_y      for task in tasks], dtype=torch.float32, device=dev)
            tile_sw = torch.tensor([task.source_width  for task in tasks], dtype=torch.float32, device=dev)
            tile_sh = torch.tensor([task.source_height for task in tasks], dtype=torch.float32, device=dev)

            ox = tile_ox[tile_ids];  oy = tile_oy[tile_ids]
            sw = tile_sw[tile_ids];  sh = tile_sh[tile_ids]

            global_boxes = torch.stack(
                [
                    (ox + boxes_norm[:, 0] * sw) / fw,
                    (oy + boxes_norm[:, 1] * sh) / fh,
                    (ox + boxes_norm[:, 2] * sw) / fw,
                    (oy + boxes_norm[:, 3] * sh) / fh,
                ],
                dim=1,
            ).clamp(0.0, 1.0)

            # ── Cross-tile NMS ────────────────────────────────────────────
            _before_xtile  = int(global_boxes.shape[0])
            _cross_nms_iou = float(self.cross_nms_iou_threshold)
            try:
                from torchvision.ops import nms as _nms  # type: ignore[import]
                keep_final   = _nms(global_boxes.float(), scores_flat.float(), iou_threshold=_cross_nms_iou)
                global_boxes = global_boxes[keep_final]
                scores_flat  = scores_flat[keep_final]
            except Exception as _exc:
                import sys as _sys
                print(f"[YOLO-DBG] cross-tile NMS failed: {_exc}", file=_sys.stderr, flush=True)

            # Optional debug hook (HOOK 4)
            self._debug_after_cross_nms(_before_xtile, int(global_boxes.shape[0]))

            # ── Hard cap ─────────────────────────────────────────────────
            _MAX_DETS = 2000
            if global_boxes.shape[0] > _MAX_DETS:
                topk = torch.topk(scores_flat, _MAX_DETS, largest=True, sorted=False)
                global_boxes = global_boxes[topk.indices]
                scores_flat  = scores_flat[topk.indices]

            # ── Single GPU→CPU transfer ───────────────────────────────────
            boxes_cpu  = global_boxes.tolist()   # sync #2
            scores_cpu = scores_flat.tolist()    # sync #3

        return [
            {"bbox": bbox, "conf": round(score, 4), "label": "person"}
            for bbox, score in zip(boxes_cpu, scores_cpu)
        ]

    # Backward-compatibility alias so existing call-sites using the old name
    # (e.g. timing tests) continue to work without modification.
    _decode_tiled_yolov8_global = _decode_tiled_global

    def _decode_tiled_postnms_vectorized(
        self, t: Any, tasks: Any, fw: int, fh: int
    ) -> list[dict[str, Any]]:
        """Vectorized post-NMS decode for batched tile output ``[N, N_dets, 6+]``."""
        n, n_dets = int(t.shape[0]), int(t.shape[1])

        confs   = t[:, :, 4]
        classes = t[:, :, 5]
        mask = (confs >= self.confidence_threshold) & (classes == float(self.person_class_id))

        tile_ids = (
            torch.arange(n, device=t.device, dtype=torch.long)
            .unsqueeze(1).expand(n, n_dets).reshape(-1)
        )

        t_flat      = t.reshape(-1, t.shape[2])
        scores_flat = confs.reshape(-1)
        mask_flat   = mask.reshape(-1)

        t_flat      = t_flat[mask_flat]       # sync #1
        scores_flat = scores_flat[mask_flat]
        tile_ids    = tile_ids[mask_flat]

        if t_flat.shape[0] == 0:
            return []

        coords = t_flat[:, :4]
        if float(t_flat[:, :4].max().item()) > 2.0:   # sync #2
            coords = coords / 640.0
        coords = coords.clamp(0.0, 1.0)

        dev = t.device
        tile_ox = torch.tensor([task.source_x      for task in tasks], dtype=torch.float32, device=dev)
        tile_oy = torch.tensor([task.source_y      for task in tasks], dtype=torch.float32, device=dev)
        tile_sw = torch.tensor([task.source_width  for task in tasks], dtype=torch.float32, device=dev)
        tile_sh = torch.tensor([task.source_height for task in tasks], dtype=torch.float32, device=dev)

        ox = tile_ox[tile_ids];  oy = tile_oy[tile_ids]
        sw = tile_sw[tile_ids];  sh = tile_sh[tile_ids]

        global_boxes = torch.stack(
            [
                (ox + coords[:, 0] * sw) / fw,
                (oy + coords[:, 1] * sh) / fh,
                (ox + coords[:, 2] * sw) / fw,
                (oy + coords[:, 3] * sh) / fh,
            ],
            dim=1,
        ).clamp(0.0, 1.0)

        try:
            from torchvision.ops import nms as _nms  # type: ignore[import]
            keep = _nms(global_boxes.float(), scores_flat.float(), iou_threshold=0.30)
            global_boxes = global_boxes[keep]
            scores_flat  = scores_flat[keep]
        except Exception:
            pass

        boxes_cpu  = global_boxes.tolist()   # sync #3
        scores_cpu = scores_flat.tolist()    # sync #4

        return [
            {"bbox": bbox, "conf": round(score, 4), "label": "person"}
            for bbox, score in zip(boxes_cpu, scores_cpu)
        ]

    def _apply_global_unletterbox(
        self, dets: list[dict[str, Any]], plan: Any
    ) -> list[dict[str, Any]]:
        """Convert bboxes from ``[0,1]``-in-letterboxed-640-space to ``[0,1]``-in-frame."""
        fw = plan.frame_width
        fh = plan.frame_height
        MODEL = 640.0

        scale  = min(MODEL / fw, MODEL / fh)
        prep_w = fw * scale
        prep_h = fh * scale
        pad_x  = (MODEL - prep_w) / 2.0
        pad_y  = (MODEL - prep_h) / 2.0

        result: list[dict[str, Any]] = []
        for det in dets:
            bx1, by1, bx2, by2 = det["bbox"]
            ox1 = (bx1 * MODEL - pad_x) / prep_w
            oy1 = (by1 * MODEL - pad_y) / prep_h
            ox2 = (bx2 * MODEL - pad_x) / prep_w
            oy2 = (by2 * MODEL - pad_y) / prep_h
            if ox2 <= 0.0 or oy2 <= 0.0 or ox1 >= 1.0 or oy1 >= 1.0:
                continue
            result.append({
                "bbox": [
                    float(max(0.0, min(1.0, ox1))),
                    float(max(0.0, min(1.0, oy1))),
                    float(max(0.0, min(1.0, ox2))),
                    float(max(0.0, min(1.0, oy2))),
                ],
                "conf":  det["conf"],
                "label": det["label"],
            })
        return result

    def _decode_seg_mask(
        self, tensors: Sequence[Any], *, clip_to_bbox: bool = True
    ) -> tuple[str, int, int] | None:
        """Compute a combined person segmentation mask for YOLO-seg models."""
        if torch is None or np is None:
            return None
        det_t   = tensors[0]
        proto_t = tensors[1]
        if not isinstance(det_t, torch.Tensor) or not isinstance(proto_t, torch.Tensor):
            return None

        n_coefs = int(proto_t.shape[1])
        mh      = int(proto_t.shape[2])
        mw      = int(proto_t.shape[3])
        if n_coefs < 1:
            return None

        with torch.no_grad():
            rows = self._to_rows(det_t.detach().float())
            if rows.numel() == 0:
                return None
            n_features = rows.shape[-1]
            n_rows     = rows.shape[0]

            is_postnms = n_features >= 6 and n_rows <= 1000

            if is_postnms:
                coef_start = 6
                conf_col   = rows[:, 4]
                cls_col    = rows[:, 5]
                person_mask = (
                    (conf_col >= self.confidence_threshold)
                    & (cls_col.round().long() == int(self.person_class_id))
                )
                if not person_mask.any():
                    return None
                bboxes_px = rows[person_mask, :4]
            else:
                coef_start = n_features - n_coefs
                if coef_start <= 4 + self.person_class_id:
                    return None
                person_scores = rows[:, 4 + self.person_class_id]
                person_mask   = person_scores >= self.confidence_threshold
                if not person_mask.any():
                    return None
                r = rows[person_mask]
                cx, cy, w, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
                bboxes_px = torch.stack(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
                )

            actual_n_coefs = n_features - coef_start
            if actual_n_coefs != n_coefs:
                coef_start     = n_features - n_coefs
                actual_n_coefs = n_coefs

            coefs  = rows[person_mask, coef_start:].float()
            protos = proto_t[0].float()

            per_person = (
                torch.sigmoid(coefs @ protos.reshape(n_coefs, -1))
                .reshape(-1, mh, mw)
                > 0.5
            )

            if clip_to_bbox:
                sx = mw / 640.0
                sy = mh / 640.0
                clip_masks = torch.zeros_like(per_person)
                for i in range(per_person.shape[0]):
                    x1 = max(0,  int(bboxes_px[i, 0].item() * sx))
                    y1 = max(0,  int(bboxes_px[i, 1].item() * sy))
                    x2 = min(mw, int(bboxes_px[i, 2].item() * sx + 0.5))
                    y2 = min(mh, int(bboxes_px[i, 3].item() * sy + 0.5))
                    if x2 > x1 and y2 > y1:
                        clip_masks[i, y1:y2, x1:x2] = True
                per_person = per_person & clip_masks

            binary  = per_person.any(0).cpu().numpy()
            mask_u8 = binary.astype(np.uint8) * 255

        raw_b64 = base64.b64encode(mask_u8.tobytes()).decode()
        return raw_b64, mw, mh

    def _decode_seg_mask_tiled(
        self,
        tensors: Sequence[Any],
        plan: Any,
        *,
        clip_to_bbox: bool = True,
    ) -> tuple[str, int, int] | None:
        """Compute combined person seg mask for a batched tile inference run."""
        if np is None or torch is None:
            return None
        if len(tensors) < 2:
            return None
        det_t   = tensors[0]
        proto_t = tensors[1]
        if not isinstance(det_t, torch.Tensor) or not isinstance(proto_t, torch.Tensor):
            return None
        if proto_t.ndim != 4 or det_t.ndim != 3:
            return None

        n_tiles = det_t.shape[0]
        n_coefs = int(proto_t.shape[1])
        mh      = int(proto_t.shape[2])
        mw      = int(proto_t.shape[3])
        fw = plan.frame_width
        fh = plan.frame_height

        mask_scale = mh / 640.0
        canvas_h = max(1, round(fh * mask_scale))
        canvas_w = max(1, round(fw * mask_scale))
        canvas = torch.zeros((canvas_h, canvas_w), dtype=torch.bool, device=det_t.device)

        with torch.no_grad():
            for i in range(n_tiles):
                if i >= len(plan.tasks):
                    break
                task = plan.tasks[i]

                det_slice   = det_t[i].float()
                proto_slice = proto_t[i].float()

                rows       = self._to_rows(det_slice)
                n_features = rows.shape[-1]
                n_rows     = rows.shape[0]

                is_postnms = n_features >= 6 and n_rows <= 1000
                if is_postnms:
                    coef_start = 6
                    conf_col   = rows[:, 4]
                    cls_col    = rows[:, 5]
                    person_mask = (
                        (conf_col >= self.confidence_threshold)
                        & (cls_col.round().long() == int(self.person_class_id))
                    )
                    if not person_mask.any():
                        continue
                    bboxes_px = rows[person_mask, :4]
                else:
                    coef_start = n_features - n_coefs
                    if coef_start <= 4 + self.person_class_id:
                        continue
                    person_scores = rows[:, 4 + self.person_class_id]
                    person_mask   = person_scores >= self.confidence_threshold
                    if not person_mask.any():
                        continue
                    r = rows[person_mask]
                    cx, cy, w, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
                    bboxes_px = torch.stack(
                        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1
                    )

                if n_features - coef_start != n_coefs:
                    coef_start = n_features - n_coefs

                coefs  = rows[person_mask, coef_start:].float()
                protos = proto_slice

                per_person = (
                    torch.sigmoid(coefs @ protos.reshape(n_coefs, -1))
                    .reshape(-1, mh, mw)
                    > 0.5
                )

                if clip_to_bbox:
                    sx = mw / 640.0
                    sy = mh / 640.0
                    clip_masks = torch.zeros_like(per_person)
                    for j in range(per_person.shape[0]):
                        x1 = max(0,  int(bboxes_px[j, 0].item() * sx))
                        y1 = max(0,  int(bboxes_px[j, 1].item() * sy))
                        x2 = min(mw, int(bboxes_px[j, 2].item() * sx + 0.5))
                        y2 = min(mh, int(bboxes_px[j, 3].item() * sy + 0.5))
                        if x2 > x1 and y2 > y1:
                            clip_masks[j, y1:y2, x1:x2] = True
                    per_person = per_person & clip_masks

                tile_union = per_person.any(0)

                cx_start = round(task.source_x * mask_scale)
                cy_start = round(task.source_y * mask_scale)
                cx_end   = min(canvas_w, cx_start + mw)
                cy_end   = min(canvas_h, cy_start + mh)
                copy_w   = cx_end - cx_start
                copy_h   = cy_end - cy_start
                if copy_w > 0 and copy_h > 0:
                    canvas[cy_start:cy_end, cx_start:cx_end] |= tile_union[:copy_h, :copy_w]

        mask_u8 = canvas.cpu().numpy().astype(np.uint8) * 255
        raw_b64 = base64.b64encode(mask_u8.tobytes()).decode()
        return raw_b64, canvas_w, canvas_h

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
            if rows.numel() == 0:
                return {"available": True, "person_candidates": 0, "device": rows.device.type}
            ncols = rows.shape[-1]
            if ncols == 6:
                person_mask = rows[:, 5] == float(self.person_class_id)
            elif ncols > 4:
                if 4 + self.person_class_id >= ncols:
                    return {"available": True, "person_candidates": 0, "device": rows.device.type}
                person_mask = rows[:, 4 + self.person_class_id] >= self.confidence_threshold
            else:
                return {"available": True, "person_candidates": 0, "device": rows.device.type}
            return {
                "available": True,
                "person_candidates": int(person_mask.sum().item()),
                "device": rows.device.type,
            }

    @staticmethod
    def _to_rows(tensor: Any) -> Any:
        """Normalise any YOLO output tensor to a 2-D ``[anchors, features]`` array."""
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]

        if tensor.ndim == 3:
            f, a = tensor.shape[1], tensor.shape[2]
            if f < a:
                return tensor.permute(0, 2, 1).reshape(-1, f)
            return tensor.reshape(-1, tensor.shape[-1])

        if tensor.ndim == 2:
            r, c = tensor.shape
            if c > r and r < 300:
                return tensor.T
            return tensor

        return tensor.reshape(-1, max(1, tensor.shape[-1]))

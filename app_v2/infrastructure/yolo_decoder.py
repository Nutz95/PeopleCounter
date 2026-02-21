from __future__ import annotations

import base64
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


class YoloDecoder(Postprocessor):
    """Decodes YOLO outputs into boxes, scores, and optional masks."""

    def __init__(self, person_class_id: int = 0, confidence_threshold: float = 0.25) -> None:
        self.person_class_id = person_class_id
        self.confidence_threshold = confidence_threshold
        # Seg-mask extraction requires a GPU→CPU copy (costly on the hot path).
        # Enable only when a seg-model is actually loaded and the UI toggle is on.
        self.seg_mask_enabled: bool = False
        # detections_gpu summary requires a GPU→CPU sync (.item()) — only
        # needed for integration-test assertions, not the live pipeline.
        self.person_summary_enabled: bool = False

    def process(self, frame_id: int, outputs: dict[str, Any], *, tile_plan: Any = None) -> dict[str, Any]:
        """Return normalized bounding boxes plus metadata for the aggregator.

        Args:
            tile_plan: Optional :class:`PreprocessPlan` produced by
                :class:`GpuPreprocessPlanner`.  When provided the decoder maps
                all bounding boxes to **global frame coordinates** (``[0, 1]``
                fractions of ``frame_width`` / ``frame_height``):

                * **Global model** (single tensor, ``mode='global'``): the
                  letterbox padding introduced by the 640×640 resize is stripped
                  so each bbox covers the matching region in the original frame.
                * **Tiles model** (batched tensor, ``mode='tiles'``): every tile
                  is decoded individually and its pixel offsets
                  (``task.source_x``, ``task.source_y``) are added before
                  normalising, so the output directly addresses the original
                  frame.  A final cross-tile NMS pass removes border duplicates.

            When *tile_plan* is ``None`` the output is in the legacy 640×640
            letterboxed coordinate space (backward-compatible with existing
            tests).
        """
        tensors = outputs.get("output_tensors", []) if isinstance(outputs, dict) else []
        detections = self._decode_detections(tensors, tile_plan=tile_plan)
        person_summary = (
            self._decode_person_detections_gpu(tensors)
            if self.person_summary_enabled
            else {"available": False, "reason": "disabled"}
        )

        # Segmentation mask: global model only (single-batch prototype tensor).
        # Tiling passes produce N-batch proto tensors; skip to avoid coordinate
        # remapping complexity.
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
        if (
            self.seg_mask_enabled
            and is_global_mode
            and len(tensors) >= 2
            and torch is not None
            and isinstance(tensors[1], torch.Tensor)
            and tensors[1].ndim == 4
            and tensors[1].shape[0] == 1
        ):
            seg_data = self._decode_seg_mask(tensors)
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

    def _decode_detections(self, tensors: Sequence[Any], *, tile_plan: Any = None) -> list[dict[str, Any]]:
        """Decode YOLO tensor output into a list of normalized bbox dicts.

        Handles two output formats:

        **Post-NMS** ``[N, 6]``: each row is ``[x1, y1, x2, y2, conf, class_id]``.
        Coordinates in model input pixel-space (640 px); divided by 640 when
        max coord > 2.

        **Raw YOLOv8 (no baked NMS)** ``[1, 4+C, 8400]`` or ``[4+C, 8400]``:
        each *column* is one anchor: ``[cx, cy, w, h, score_c0, …, score_cN]``.
        ``_to_rows`` transposes to ``[8400, 4+C]``.  We then pick the
        ``person_class_id`` score column, threshold, convert cx/cy/w/h →
        x1/y1/x2/y2, and normalise to 0-1.

        When *tile_plan* is provided and the tensor is a batched tile run
        (``shape[0] > 1``, ``mode='tiles'``), each tile is decoded individually
        and its detections are remapped to global frame coordinates (see
        :meth:`_decode_tiled_yolov8_global`).  For a global model with a plan,
        the single-pass letterbox coordinates are un-letterboxed server-side via
        :meth:`_apply_global_unletterbox`.
        """
        if torch is None or not tensors:
            return []
        first = tensors[0]
        if not isinstance(first, torch.Tensor):
            return []

        # ── Batched tiles path: each batch element is one tile crop ──────────
        # Process every tile independently so we can apply its spatial offset
        # before combining.  This preserves tile identity that would otherwise
        # be lost after the reshape in _to_rows.
        if (
            tile_plan is not None
            and first.ndim == 3
            and first.shape[0] > 1
            and hasattr(tile_plan, "tasks")
            and isinstance(tile_plan.metadata, dict)
            and tile_plan.metadata.get("mode") == "tiles"
        ):
            return self._decode_tiled_yolov8_global(first.detach().float(), tile_plan)

        # ── Single-tensor path (global model or legacy no-plan call) ─────────
        with torch.no_grad():
            rows = self._to_rows(first.detach().float())
            if rows.numel() == 0:
                return []

            ncols = rows.shape[-1]

            if ncols == 6:
                # ── Post-NMS: [N, 6] = [x1, y1, x2, y2, conf, cls_id] ─────
                dets = self._decode_postnms(rows)
            elif ncols > 4:
                # ── Raw YOLOv8: [N, 4+C] = [cx, cy, w, h, s0, s1, …] ──────
                dets = self._decode_raw_yolov8(rows)
            else:
                return []

        # Un-letterbox global model bboxes to original frame space when the plan
        # carries the original resolution (mode='global').
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
        confs = rows[:, 4]
        classes = rows[:, 5]
        mask = (confs >= self.confidence_threshold) & (
            classes == float(self.person_class_id)
        )
        rows = rows[mask]
        if rows.shape[0] == 0:
            return []
        coords = rows[:, :4].cpu()
        if float(coords.max()) > 2.0:
            coords = coords / 640.0
        coords = coords.clamp(0.0, 1.0)
        result: list[dict[str, Any]] = []
        for i in range(rows.shape[0]):
            x1, y1, x2, y2 = coords[i].tolist()
            result.append({"bbox": [x1, y1, x2, y2], "conf": round(float(rows[i, 4]), 4), "label": "person"})
        return result

    def _decode_raw_yolov8(self, rows: Any) -> list[dict[str, Any]]:
        """Decode raw YOLOv8 ``[N, 4+C]`` tensor (no NMS applied by the engine).

        Standard ultralytics ONNX/TRT export (opset 12, no baked NMS plugin)
        outputs each anchor as ``[cx, cy, w, h, score_c0, ..., score_cN]`` where
        cx/cy/w/h are in absolute **pixel** coordinates within the 640×640
        letterboxed model input.  This is **xywh center** format, *not* xyxy.

        Reference: ultralytics/engine/exporter.py, ``export_onnx()`` — the raw
        CML head output is forwarded without coordinate conversion.
        """
        num_classes = rows.shape[-1] - 4
        if self.person_class_id >= num_classes:
            return []
        person_scores = rows[:, 4 + self.person_class_id]
        mask = person_scores >= self.confidence_threshold
        rows_f = rows[mask]
        scores_f = person_scores[mask]
        if rows_f.shape[0] == 0:
            return []

        # xywh center → xyxy corner conversion (still in 640-px letterboxed space)
        cx = rows_f[:, 0]
        cy = rows_f[:, 1]
        w  = rows_f[:, 2]
        h  = rows_f[:, 3]
        lx1 = cx - w / 2.0
        ly1 = cy - h / 2.0
        lx2 = cx + w / 2.0
        ly2 = cy + h / 2.0

        # Apply NMS to collapse overlapping proposals for the same person.
        try:
            from torchvision.ops import nms as _nms  # type: ignore[import]
            boxes_nms = torch.stack([lx1, ly1, lx2, ly2], dim=1).float()
            keep = _nms(boxes_nms, scores_f.float(), iou_threshold=0.45)
            lx1, ly1, lx2, ly2 = lx1[keep], ly1[keep], lx2[keep], ly2[keep]
            scores_f = scores_f[keep]
        except Exception:  # torchvision unavailable or only one box
            pass

        # Normalise xyxy pixel coords (0..640) → fraction (0..1) in the
        # letterboxed 640×640 space.  The client un-letterboxes before drawing.
        x1 = (lx1 / 640.0).clamp(0.0, 1.0)
        y1 = (ly1 / 640.0).clamp(0.0, 1.0)
        x2 = (lx2 / 640.0).clamp(0.0, 1.0)
        y2 = (ly2 / 640.0).clamp(0.0, 1.0)

        coords = torch.stack([x1, y1, x2, y2], dim=1).cpu()
        result: list[dict[str, Any]] = []
        for i in range(coords.shape[0]):
            bx1, by1, bx2, by2 = coords[i].tolist()
            result.append({
                "bbox": [bx1, by1, bx2, by2],
                "conf": round(float(scores_f[i]), 4),
                "label": "person",
            })
        return result

    def _decode_tiled_yolov8_global(self, tensor: Any, plan: Any) -> list[dict[str, Any]]:
        """Decode batched tile tensor to global frame coordinates ``[0, 1]``.

        Each batch element ``i`` corresponds to ``plan.tasks[i]``.  The tile
        covers the region ``(source_x, source_y)`` to
        ``(source_x + source_width, source_y + source_height)`` in the original
        frame.  For typical 640×640 tile crops there is no letterboxing, so the
        remapping is simply::

            global_x = (source_x + tile_x_px) / frame_width
            global_y = (source_y + tile_y_px) / frame_height

        A cross-tile NMS pass with ``iou=0.45`` removes duplicate detections
        from overlapping tile borders.
        """
        fw = plan.frame_width
        fh = plan.frame_height
        n_tiles = tensor.shape[0]
        all_dets: list[dict[str, Any]] = []

        with torch.no_grad():
            for i in range(n_tiles):
                if i >= len(plan.tasks):
                    break
                task = plan.tasks[i]

                rows = self._to_rows(tensor[i])  # [8400, 84] for raw YOLOv8
                if rows.numel() == 0:
                    continue
                ncols = rows.shape[-1]
                if ncols == 6:
                    tile_dets = self._decode_postnms(rows)
                elif ncols > 4:
                    tile_dets = self._decode_raw_yolov8(rows)
                else:
                    continue

                # Remap from [0,1] in tile space → [0,1] in original frame.
                # Tile pixels = bbox_norm × source_width (no scale factor since
                # source_width == target_width == 640 for typical tiling plans).
                sw = task.source_width
                sh = task.source_height
                ox = task.source_x
                oy = task.source_y
                for det in tile_dets:
                    bx1, by1, bx2, by2 = det["bbox"]
                    gx1 = (ox + bx1 * sw) / fw
                    gy1 = (oy + by1 * sh) / fh
                    gx2 = (ox + bx2 * sw) / fw
                    gy2 = (oy + by2 * sh) / fh
                    all_dets.append({
                        "bbox": [
                            float(max(0.0, min(1.0, gx1))),
                            float(max(0.0, min(1.0, gy1))),
                            float(max(0.0, min(1.0, gx2))),
                            float(max(0.0, min(1.0, gy2))),
                        ],
                        "conf": det["conf"],
                        "label": det["label"],
                    })

        # Cross-tile NMS: remove detections that span multiple overlapping tiles.
        if len(all_dets) > 1 and torch is not None:
            try:
                from torchvision.ops import nms as _nms  # type: ignore[import]
                boxes = torch.tensor([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]] for d in all_dets])
                scores = torch.tensor([d["conf"] for d in all_dets])
                keep = _nms(boxes, scores, iou_threshold=0.45).tolist()
                all_dets = [all_dets[k] for k in keep]
            except Exception:
                pass

        return all_dets

    def _apply_global_unletterbox(self, dets: list[dict[str, Any]], plan: Any) -> list[dict[str, Any]]:
        """Convert bboxes from ``[0,1]``-in-letterboxed-640-space to ``[0,1]``-in-frame.

        The global YOLO model letterboxes the full frame to 640×640, adding black
        padding on the shorter axis.  This strips that padding so bboxes address
        the original frame directly::

            pad_x = (640 - frame_width  * scale) / 2
            pad_y = (640 - frame_height * scale) / 2
            global_x = (letterbox_x * 640 - pad_x) / (frame_width  * scale)
        """
        fw = plan.frame_width
        fh = plan.frame_height
        MODEL = 640.0

        scale = min(MODEL / fw, MODEL / fh)
        prep_w = fw * scale        # content width in 640-px letterboxed space
        prep_h = fh * scale        # content height
        pad_x = (MODEL - prep_w) / 2.0
        pad_y = (MODEL - prep_h) / 2.0

        result: list[dict[str, Any]] = []
        for det in dets:
            bx1, by1, bx2, by2 = det["bbox"]
            # Denormalise → 640-px space, strip padding, renormalise to [0,1]
            ox1 = (bx1 * MODEL - pad_x) / prep_w
            oy1 = (by1 * MODEL - pad_y) / prep_h
            ox2 = (bx2 * MODEL - pad_x) / prep_w
            oy2 = (by2 * MODEL - pad_y) / prep_h
            # Skip boxes that are entirely inside the padding region
            if ox2 <= 0.0 or oy2 <= 0.0 or ox1 >= 1.0 or oy1 >= 1.0:
                continue
            result.append({
                "bbox": [
                    float(max(0.0, min(1.0, ox1))),
                    float(max(0.0, min(1.0, oy1))),
                    float(max(0.0, min(1.0, ox2))),
                    float(max(0.0, min(1.0, oy2))),
                ],
                "conf": det["conf"],
                "label": det["label"],
            })
        return result

    def _decode_seg_mask(self, tensors: Sequence[Any]) -> tuple[str, int, int] | None:
        """Compute a combined person segmentation mask for YOLO-seg models.

        Requires two output tensors:

        * ``tensors[0]``: ``[1, 4 + n_cls + n_coefs, 8400]`` — detections.
        * ``tensors[1]``: ``[1, n_coefs, H, W]``             — prototype masks
          (typically 160 × 160).

        Returns ``(base64_bytes, width, height)`` where *base64_bytes* encodes a
        flat uint8 array (0 = background, 255 = person pixel) of size ``W × H``,
        or ``None`` when masks cannot be computed.
        """
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
            # Layout: [cx, cy, w, h, score_c0 … score_cN, coef_0 … coef_K-1]
            # Mask coefficients occupy the *last* n_coefs columns.
            coef_start = n_features - n_coefs
            if coef_start <= 4 + self.person_class_id:
                return None  # too few feature columns — not a seg model

            person_scores = rows[:, 4 + self.person_class_id]
            person_mask   = person_scores >= self.confidence_threshold
            if not person_mask.any():
                return None

            coefs  = rows[person_mask, coef_start:].float()   # [N, n_coefs]
            protos = proto_t[0].float()                        # [n_coefs, mh, mw]

            # Per-person mask: sigmoid(coefs @ protos.reshape(n_coefs, mh*mw))
            # → [N, mh*mw]; take union across all persons via max.
            combined = torch.sigmoid(coefs @ protos.reshape(n_coefs, -1))  # [N, mh*mw]
            binary   = (combined.max(0).values > 0.5).cpu().numpy().reshape(mh, mw)
            mask_u8  = (binary.astype(np.uint8) * 255)

        raw_b64 = base64.b64encode(mask_u8.tobytes()).decode()
        return raw_b64, mw, mh

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
                # Post-NMS: [N, 6] = [x1, y1, x2, y2, conf, cls_id]
                person_mask = rows[:, 5] == float(self.person_class_id)
            elif ncols > 4:
                # Raw YOLOv8: [N, 4+C] — person score at col 4+person_class_id
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
        """Normalise any YOLO output tensor to a 2-D ``[anchors, features]`` array.

        Handles:
        * ``[1, features, anchors]`` — batch-1 raw YOLOv8 (e.g. ``[1, 84, 8400]``)
        * ``[B, features, anchors]`` — batched tiles (e.g. ``[6, 84, 8400]``)
        * ``[features, anchors]``    — already squeezed (e.g. ``[84, 8400]``)
        * ``[N, 6]``                 — post-NMS (N detections, 6 values)
        """
        # Squeeze batch=1 away
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]   # → [features, anchors] or [anchors, features]

        # Batched 3-D: [B, features, anchors]
        if tensor.ndim == 3:
            f, a = tensor.shape[1], tensor.shape[2]
            if f < a:
                # [B, features, anchors] → permute → [B*anchors, features]
                return tensor.permute(0, 2, 1).reshape(-1, f)
            # [B, anchors, features]
            return tensor.reshape(-1, tensor.shape[-1])

        # 2-D: detect orientation
        if tensor.ndim == 2:
            r, c = tensor.shape
            # [features, anchors]: feature dim << anchor dim
            # Use threshold 300: YOLO feature dim (4+classes) is always < 300,
            # while anchor count (≥1600) is always ≥ it.
            if c > r and r < 300:
                return tensor.T   # [features, anchors] → [anchors, features]
            return tensor

        return tensor.reshape(-1, max(1, tensor.shape[-1]))

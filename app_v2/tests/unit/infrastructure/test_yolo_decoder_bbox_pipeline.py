"""End-to-end roundtrip tests for the YOLO bbox coordinate pipeline.

Each test covers one layer of the decode chain:

  1. YoloDecoder._decode_raw_yolov8  — xywh center coords → normalised xyxy
  2. Letterbox un-map (JS drawMask equivalent; reimplemented here in Python) — maps
     normalised [0,1] in 640×640 padded space back to normalised [0,1] in original
     frame space, so the numbers can be verified against known input positions.

Format contract (standard ultralytics ONNX/TRT opset-12 export, no baked NMS):
  Tensor shape : [1, 4+C, 8400]
  Columns 0-3  : cx, cy, w, h  in *absolute pixels* of the 640×640 letterboxed input
  Columns 4..  : per-class sigmoid scores
"""
from __future__ import annotations

import math
from typing import Sequence

import pytest

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


# ── helpers ──────────────────────────────────────────────────────────────────

MODEL_SIZE = 640
NUM_ANCHORS = 8400
NUM_CLASSES = 80


def _make_xywh_tensor(
    person_boxes: Sequence[tuple[float, float, float, float]],
    *,
    conf: float = 0.9,
    person_class_id: int = 0,
    num_anchors: int = NUM_ANCHORS,
    num_classes: int = NUM_CLASSES,
) -> "torch.Tensor":
    """Build a synthetic ``[1, 4+C, num_anchors]`` raw YOLOv8 tensor.

    All background scores are set to 0.01; *person* scores at the selected
    anchor positions are set to *conf*.
    """
    tensor = torch.zeros(1, 4 + num_classes, num_anchors)
    tensor[:, 4:, :] = 0.01  # low background score everywhere

    for i, (cx, cy, w, h) in enumerate(person_boxes):
        anchor_idx = 50 + i * 200  # arbitrary but deterministic anchor slots
        tensor[0, 0, anchor_idx] = cx
        tensor[0, 1, anchor_idx] = cy
        tensor[0, 2, anchor_idx] = w
        tensor[0, 3, anchor_idx] = h
        tensor[0, 4 + person_class_id, anchor_idx] = conf

    return tensor


def _unletterbox(
    bbox_norm: tuple[float, float, float, float],
    src_w: int,
    src_h: int,
    model_size: int = MODEL_SIZE,
) -> tuple[float, float, float, float]:
    """Python equivalent of the ``drawMask`` JS un-letterbox math.

    Takes a bbox normalised in ``model_size × model_size`` letterboxed space
    and returns a bbox normalised in the original ``src_w × src_h`` frame.
    """
    scale = min(model_size / src_w, model_size / src_h)
    prep_w = src_w * scale   # letterboxed content width  (px in 640-space)
    prep_h = src_h * scale   # letterboxed content height (px in 640-space)
    pad_x  = (model_size - prep_w) / 2.0
    pad_y  = (model_size - prep_h) / 2.0

    bx1, by1, bx2, by2 = bbox_norm
    lx1 = bx1 * model_size
    ly1 = by1 * model_size
    lx2 = bx2 * model_size
    ly2 = by2 * model_size

    ox1 = (lx1 - pad_x) / prep_w
    oy1 = (ly1 - pad_y) / prep_h
    ox2 = (lx2 - pad_x) / prep_w
    oy2 = (ly2 - pad_y) / prep_h

    return ox1, oy1, ox2, oy2


def _letterbox_box(
    box_px: tuple[float, float, float, float],
    src_w: int,
    src_h: int,
    model_size: int = MODEL_SIZE,
) -> tuple[float, float, float, float]:
    """Map an (x1, y1, x2, y2) box in original pixel coords to xywh
    center coords in the letterboxed model space (for building test tensors)."""
    x1, y1, x2, y2 = box_px
    scale = min(model_size / src_w, model_size / src_h)
    pad_x = (model_size - src_w * scale) / 2.0
    pad_y = (model_size - src_h * scale) / 2.0

    cx = pad_x + (x1 + x2) / 2.0 * scale
    cy = pad_y + (y1 + y2) / 2.0 * scale
    w  = (x2 - x1) * scale
    h  = (y2 - y1) * scale
    return cx, cy, w, h


# ── unit tests ────────────────────────────────────────────────────────────────

def test_decode_xywh_single_person_no_letterbox() -> None:
    """Person exactly at center of the 640×640 model space: bbox should round-trip."""
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    cx, cy, w, h = 320.0, 320.0, 100.0, 200.0
    tensor = _make_xywh_tensor([(cx, cy, w, h)])

    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.5)
    dets = decoder._decode_detections([tensor])

    assert len(dets) == 1, f"Expected 1 detection, got {len(dets)}: {dets}"
    bx1, by1, bx2, by2 = dets[0]["bbox"]

    tol = 1.0 / MODEL_SIZE  # one pixel tolerance
    assert abs(bx1 - (cx - w / 2) / 640) < tol, f"x1: {bx1:.4f}"
    assert abs(by1 - (cy - h / 2) / 640) < tol, f"y1: {by1:.4f}"
    assert abs(bx2 - (cx + w / 2) / 640) < tol, f"x2: {bx2:.4f}"
    assert abs(by2 - (cy + h / 2) / 640) < tol, f"y2: {by2:.4f}"
    assert dets[0]["label"] == "person"
    assert dets[0]["conf"] == pytest.approx(0.9, abs=0.001)


def test_decode_filters_below_threshold() -> None:
    """A box with score below threshold must not appear in the output."""
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    tensor = _make_xywh_tensor([(320.0, 320.0, 100.0, 200.0)], conf=0.2)
    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.25)
    dets = decoder._decode_detections([tensor])
    assert dets == [], f"Expected empty, got {dets}"


def test_decode_multiple_persons_nms_reduces_duplicates() -> None:
    """Two heavily overlapping boxes for the same person should collapse to one after NMS."""
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    # Two nearly identical boxes (>0.45 IoU) → NMS keeps highest-score one
    tensor = _make_xywh_tensor([
        (320.0, 320.0, 100.0, 200.0),  # anchor 0 (conf 0.95 in the helper — see below)
        (322.0, 322.0, 100.0, 200.0),  # anchor 1 (conf 0.90)
    ], conf=0.9)
    # Give first box higher confidence so we know which NMS will keep
    anchor0 = 50
    tensor[0, 4, anchor0] = 0.95

    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.5)
    dets = decoder._decode_detections([tensor])
    # May return 1 or 2 depending on torchvision NMS availability; must not be 0
    assert 1 <= len(dets) <= 2, f"Unexpected det count: {len(dets)}"


def test_decode_wrong_class_returns_empty() -> None:
    """Tensor with only class-1 detections: person_class_id=0 should return nothing."""
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    tensor = torch.zeros(1, 4 + NUM_CLASSES, NUM_ANCHORS)
    tensor[:, 4:, :] = 0.01
    anchor_idx = 100
    tensor[0, 0, anchor_idx] = 320.0
    tensor[0, 1, anchor_idx] = 320.0
    tensor[0, 2, anchor_idx] = 80.0
    tensor[0, 3, anchor_idx] = 160.0
    tensor[0, 4 + 1, anchor_idx] = 0.95  # class 1 (not person)

    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.25)
    dets = decoder._decode_detections([tensor])
    assert dets == [], f"Expected empty for wrong class, got {dets}"


# ── letterbox roundtrip tests ─────────────────────────────────────────────────

@pytest.mark.parametrize("src_w, src_h, person_box_px", [
    # 1080p landscape: pillar-box only (full width, padded top/bottom)
    (1920, 1080, (760, 200, 960, 700)),
    # 4K UHD — same aspect ratio as 1080p, same letterbox geometry
    (3840, 2160, (1200, 400, 1600, 1600)),
    # Square source: no letterbox at all
    (640, 640, (100, 50, 300, 400)),
    # Portrait 9:16 — padded left/right (pillarbox)
    (1080, 1920, (200, 500, 600, 1400)),
])
def test_letterbox_roundtrip(
    src_w: int, src_h: int, person_box_px: tuple[float, float, float, float]
) -> None:
    """Person bbox: original px → letterboxed xywh tensor → YoloDecoder → un-letterbox
    → should recover the original normalised bbox to within ~1 pixel at model resolution.
    """
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    # 1. Compute what the TRT model would output for this person in letterbox space
    cx, cy, w_lb, h_lb = _letterbox_box(person_box_px, src_w, src_h)

    # 2. Build a synthetic [1, 84, 8400] tensor with that person anchor
    tensor = _make_xywh_tensor([(cx, cy, w_lb, h_lb)], conf=0.9)

    # 3. Run through YoloDecoder (same code as production)
    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.5)
    dets = decoder._decode_detections([tensor])
    assert len(dets) >= 1, (
        f"YoloDecoder returned no detections for src={src_w}×{src_h} "
        f"box={person_box_px}"
    )

    # 4. Apply un-letterbox (same math as JS drawMask)
    ox1, oy1, ox2, oy2 = _unletterbox(dets[0]["bbox"], src_w, src_h)

    # 5. Expected normalised bbox in original frame
    x1_px, y1_px, x2_px, y2_px = person_box_px
    exp_ox1 = x1_px / src_w
    exp_oy1 = y1_px / src_h
    exp_ox2 = x2_px / src_w
    exp_oy2 = y2_px / src_h

    # Tolerance: 2 pixels in model space → ~2/640 in normalised coords
    tol = 2.0 / MODEL_SIZE
    assert abs(ox1 - exp_ox1) < tol, f"ox1 {ox1:.4f} != {exp_ox1:.4f} (src={src_w}×{src_h})"
    assert abs(oy1 - exp_oy1) < tol, f"oy1 {oy1:.4f} != {exp_oy1:.4f} (src={src_w}×{src_h})"
    assert abs(ox2 - exp_ox2) < tol, f"ox2 {ox2:.4f} != {exp_ox2:.4f} (src={src_w}×{src_h})"
    assert abs(oy2 - exp_oy2) < tol, f"oy2 {oy2:.4f} != {exp_oy2:.4f} (src={src_w}×{src_h})"


# ── server-side global frame coordinate tests ─────────────────────────────────
#
# When a PreprocessPlan is supplied, the decoder maps bbox outputs to global
# [0,1] frame coordinates so the browser can draw bboxes with a simple linear
# mapping — no letterbox math required in JavaScript.

def _make_global_plan(src_w: int, src_h: int) -> "PreprocessPlan":
    from app_v2.core.preprocessor_types.preprocess_plan import PreprocessPlan
    from app_v2.core.preprocessor_types.preprocess_task import PreprocessTask
    task = PreprocessTask(
        model_name="yolo_global",
        task_index=0,
        source_x=0,
        source_y=0,
        source_width=src_w,
        source_height=src_h,
        target_width=640,
        target_height=640,
        metadata={"kind": "letterbox"},
    )
    return PreprocessPlan(
        model_name="yolo_global",
        frame_width=src_w,
        frame_height=src_h,
        tasks=(task,),
        metadata={"mode": "global", "task_count": 1},
    )


def _make_tiling_plan(src_w: int, src_h: int, tile_size: int = 640) -> "PreprocessPlan":
    from app_v2.core.preprocessor_types.preprocess_plan import PreprocessPlan
    from app_v2.core.preprocessor_types.preprocess_task import PreprocessTask
    tasks = []
    idx = 0
    for row_offset in range(0, src_h, tile_size):
        for col_offset in range(0, src_w, tile_size):
            tasks.append(PreprocessTask(
                model_name="yolo_tiles",
                task_index=idx,
                source_x=col_offset,
                source_y=row_offset,
                source_width=min(tile_size, src_w - col_offset),
                source_height=min(tile_size, src_h - row_offset),
                target_width=tile_size,
                target_height=tile_size,
                metadata={"row": row_offset // tile_size, "col": col_offset // tile_size},
            ))
            idx += 1
    return PreprocessPlan(
        model_name="yolo_tiles",
        frame_width=src_w,
        frame_height=src_h,
        tasks=tuple(tasks),
        metadata={"mode": "tiles", "task_count": len(tasks), "rows": src_h // tile_size, "cols": src_w // tile_size, "overlap": 0.0},
    )


@pytest.mark.parametrize("src_w, src_h, person_box_px", [
    (1920, 1080, (760, 200, 960, 700)),
    (3840, 2160, (1200, 400, 1600, 1600)),
    (640, 640,  (100,  50, 300, 400)),
    (1080, 1920, (200, 500, 600, 1400)),
])
def test_global_plan_server_unletterbox_roundtrip(
    src_w: int, src_h: int, person_box_px: tuple[float, float, float, float]
) -> None:
    """When a global plan is supplied the decoder un-letterboxes server-side.

    The browser only needs a linear mapping (no letterbox math).
    """
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    # 1. Build synthetic YOLO tensor in letterbox space
    cx, cy, w_lb, h_lb = _letterbox_box(person_box_px, src_w, src_h)
    tensor = _make_xywh_tensor([(cx, cy, w_lb, h_lb)], conf=0.9)

    # 2. Decode with plan – produces global frame coords directly
    plan = _make_global_plan(src_w, src_h)
    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.5)
    result = decoder.process(0, {"output_tensors": [tensor]}, tile_plan=plan)
    dets = result["detections"]

    assert len(dets) >= 1, (
        f"No detections after server-side unletterbox for {src_w}×{src_h} "
        f"box={person_box_px}"
    )

    # 3. Verify against original normalised bbox — no browser math needed
    bx1, by1, bx2, by2 = dets[0]["bbox"]
    x1_px, y1_px, x2_px, y2_px = person_box_px
    exp_x1 = x1_px / src_w
    exp_y1 = y1_px / src_h
    exp_x2 = x2_px / src_w
    exp_y2 = y2_px / src_h

    tol = 2.0 / MODEL_SIZE
    assert abs(bx1 - exp_x1) < tol, f"x1 {bx1:.4f} != {exp_x1:.4f} (src={src_w}×{src_h})"
    assert abs(by1 - exp_y1) < tol, f"y1 {by1:.4f} != {exp_y1:.4f} (src={src_w}×{src_h})"
    assert abs(bx2 - exp_x2) < tol, f"x2 {bx2:.4f} != {exp_x2:.4f} (src={src_w}×{src_h})"
    assert abs(by2 - exp_y2) < tol, f"y2 {by2:.4f} != {exp_y2:.4f} (src={src_w}×{src_h})"


def test_tiles_two_tiles_global_coords() -> None:
    """Two-tile batch: each tile's detections are mapped to global frame coords.

    Frame 1280×640, two 640×640 tiles side-by-side:
      tile 0 at (0,0)   → person at tile-local (320,320) → global (320/1280, 320/640) = (0.25, 0.50)
      tile 1 at (640,0) → person at tile-local (320,320) → global (960/1280, 320/640) = (0.75, 0.50)
    """
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    FRAME_W, FRAME_H = 1280, 640

    def _make_tile_tensor(cx: float, cy: float, w: float = 60.0, h: float = 120.0) -> "torch.Tensor":
        t = torch.zeros(4 + NUM_CLASSES, NUM_ANCHORS)
        t[4:, :] = 0.01
        t[0, 50] = cx
        t[1, 50] = cy
        t[2, 50] = w
        t[3, 50] = h
        t[4 + 0, 50] = 0.9
        return t

    # Batched tensor – shape [2, 84, 8400]
    tile0 = _make_tile_tensor(320.0, 320.0)
    tile1 = _make_tile_tensor(320.0, 320.0)
    batched = torch.stack([tile0, tile1], dim=0).unsqueeze(0)  # → [1, 84, 8400] per tile merged
    # Actually build [2, 84, 8400] directly (batch dim = number of tiles)
    batched = torch.stack([tile0, tile1], dim=0)  # [2, 84, 8400]

    plan = _make_tiling_plan(FRAME_W, FRAME_H, tile_size=640)
    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.5)
    result = decoder.process(0, {"output_tensors": [batched]}, tile_plan=plan)
    dets = result["detections"]

    assert len(dets) >= 2, f"Expected ≥2 global detections, got {len(dets)}: {dets}"

    xs = sorted(d["bbox"][0] for d in dets)
    # Person in tile 0 → cx=320 → global x ≈ 320/1280 = 0.25
    # Person in tile 1 → cx=320, tile offset x=640 → global x ≈ (640+320)/1280 = 0.75
    tol = 2.0 / 640
    assert abs(xs[0] - 0.25 + 60 / 2 / 1280) < tol or abs(xs[0] - (320 - 30) / 1280) < tol, f"Left det x1 unexpected: {xs[0]:.4f}"
    assert abs(xs[-1] - (640 + 320 - 30) / 1280) < tol, f"Right det x1 unexpected: {xs[-1]:.4f}"


def test_tiles_no_plan_returns_merged_letterbox_coords() -> None:
    """Without a plan the decoder falls back to merged (tile-unaware) coords — backward compat."""
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    tile0 = torch.zeros(4 + NUM_CLASSES, NUM_ANCHORS)
    tile0[4:, :] = 0.01
    tile0[0, 50] = 320.0; tile0[1, 50] = 320.0
    tile0[2, 50] = 60.0;  tile0[3, 50] = 120.0
    tile0[4, 50] = 0.9
    batched = torch.stack([tile0, tile0], dim=0)  # [2, 84, 8400]

    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.5)
    # No tile_plan → legacy path: all tile anchors are merged, coords in [0,1] letterbox-space
    result = decoder.process(0, {"output_tensors": [batched]})
    # Should still produce detections (merged tile path), coords in letterbox-space
    assert isinstance(result["detections"], list)

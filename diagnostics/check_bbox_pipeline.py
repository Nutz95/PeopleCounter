#!/usr/bin/env python3
"""Diagnostic script: validate the full bbox coordinate pipeline on real images.

Usage (no GPU/TRT required):

    python3 diagnostics/check_bbox_pipeline.py

What it does:
  1. Loads a COCO val image from coco_val_minimal/ (a 1000-image subset).
  2. Applies the same letterbox transform as the GPU preprocessor (640×640).
  3. Places a *synthetic* person detection at a chosen pixel position in the
     ORIGINAL frame (simulating what the TRT model would output).
  4. Runs that synthetic tensor through YoloDecoder.
  5. Applies the JS drawMask un-letterbox math (in Python).
  6. Draws the result bbox on the original image so you can see it visually.

The output images are saved to diagnostics/bbox_check/ and the script prints
a pass/fail summary.

Requirements:
    pip install Pillow
"""
from __future__ import annotations

import math
import os
import sys

# Run from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not installed — visual output disabled; numeric checks still run.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    print("[ERROR] PyTorch not found. Install requirements first.")
    sys.exit(1)

from app_v2.infrastructure.yolo_decoder import YoloDecoder  # noqa: E402

MODEL_SIZE = 640
NUM_ANCHORS = 8400
NUM_CLASSES = 80

OUT_DIR = os.path.join(REPO_ROOT, "diagnostics", "bbox_check")
IMG_DIR = os.path.join(REPO_ROOT, "coco_val_minimal", "images", "val2017")


# ── coordinate helpers ────────────────────────────────────────────────────────

def letterbox_params(src_w: int, src_h: int, model: int = MODEL_SIZE):
    scale = min(model / src_w, model / src_h)
    prep_w = src_w * scale
    prep_h = src_h * scale
    pad_x  = (model - prep_w) / 2.0
    pad_y  = (model - prep_h) / 2.0
    return scale, prep_w, prep_h, pad_x, pad_y


def box_to_letterbox_xywh(box_px, src_w, src_h):
    """Original pixel [x1,y1,x2,y2] → xywh center in 640-space (TRT model output)."""
    x1, y1, x2, y2 = box_px
    scale, _, _, pad_x, pad_y = letterbox_params(src_w, src_h)
    cx = pad_x + (x1 + x2) / 2.0 * scale
    cy = pad_y + (y1 + y2) / 2.0 * scale
    w  = (x2 - x1) * scale
    h  = (y2 - y1) * scale
    return cx, cy, w, h


def unletterbox(bbox_norm, src_w, src_h):
    """Normalised [0,1] in 640-space → normalised [0,1] in original frame (drawMask logic)."""
    scale, prep_w, prep_h, pad_x, pad_y = letterbox_params(src_w, src_h)
    bx1, by1, bx2, by2 = bbox_norm
    lx1 = bx1 * MODEL_SIZE;  ly1 = by1 * MODEL_SIZE
    lx2 = bx2 * MODEL_SIZE;  ly2 = by2 * MODEL_SIZE
    ox1 = (lx1 - pad_x) / prep_w;  oy1 = (ly1 - pad_y) / prep_h
    ox2 = (lx2 - pad_x) / prep_w;  oy2 = (ly2 - pad_y) / prep_h
    return ox1, oy1, ox2, oy2


def make_xywh_tensor(cx, cy, w, h, conf=0.9, person_class_id=0):
    tensor = torch.zeros(1, 4 + NUM_CLASSES, NUM_ANCHORS)
    tensor[:, 4:, :] = 0.01
    idx = 42
    tensor[0, 0, idx] = cx
    tensor[0, 1, idx] = cy
    tensor[0, 2, idx] = w
    tensor[0, 3, idx] = h
    tensor[0, 4 + person_class_id, idx] = conf
    return tensor


# ── test cases ────────────────────────────────────────────────────────────────

CASES = [
    # (description, image_file, src_w, src_h, person_box_px [x1,y1,x2,y2])
    # The person box is where we *pretend* the TRT model detected a person.
    # For 1080p source the image is resized to 1920×1080.
    # For 4K source it is resized to 3840×2160.
    ("1080p_center_person",   "000000000285.jpg", 1920, 1080, (760, 200, 1160, 900)),
    ("4k_center_person",      "000000000724.jpg", 3840, 2160, (1520, 400, 2320, 1800)),
    ("1080p_left_edge_person","000000000885.jpg", 1920, 1080, (10,  200,  300, 700)),
    ("portrait_9x16_person",  "000000001000.jpg", 1080, 1920, (200, 500,  600, 1400)),
]


def run_case(desc, img_file, src_w, src_h, box_px):
    print(f"\n── {desc} ({src_w}×{src_h}) ──")

    # 1. Compute letterboxed xywh coords (what TRT model would output)
    cx, cy, w_lb, h_lb = box_to_letterbox_xywh(box_px, src_w, src_h)
    print(f"   Letterboxed xywh (640-space): cx={cx:.1f}, cy={cy:.1f}, w={w_lb:.1f}, h={h_lb:.1f}")

    # 2. Build synthetic tensor and decode
    tensor = make_xywh_tensor(cx, cy, w_lb, h_lb)
    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.5)
    dets = decoder._decode_detections([tensor])

    if not dets:
        print("   [FAIL] YoloDecoder returned no detections!")
        return False

    decoded_bbox = dets[0]["bbox"]
    print(f"   Decoded bbox (norm 640-space): {[f'{v:.4f}' for v in decoded_bbox]}")

    # 3. Un-letterbox (JS drawMask equivalent)
    ox1, oy1, ox2, oy2 = unletterbox(decoded_bbox, src_w, src_h)
    print(f"   Un-letterboxed (norm orig):    x1={ox1:.4f}, y1={oy1:.4f}, x2={ox2:.4f}, y2={oy2:.4f}")

    # 4. Expected
    x1_px, y1_px, x2_px, y2_px = box_px
    exp_ox1, exp_oy1, exp_ox2, exp_oy2 = (
        x1_px / src_w, y1_px / src_h, x2_px / src_w, y2_px / src_h
    )
    print(f"   Expected      (norm orig):    x1={exp_ox1:.4f}, y1={exp_oy1:.4f}, x2={exp_ox2:.4f}, y2={exp_oy2:.4f}")

    tol = 2.0 / MODEL_SIZE
    errors = {
        "ox1": abs(ox1 - exp_ox1), "oy1": abs(oy1 - exp_oy1),
        "ox2": abs(ox2 - exp_ox2), "oy2": abs(oy2 - exp_oy2),
    }
    pass_all = all(v < tol for v in errors.values())
    for k, v in errors.items():
        status = "OK" if v < tol else f"FAIL (err={v:.4f} > tol={tol:.4f})"
        print(f"   {k}: {status}")

    # 5. Visual output
    if HAS_PIL:
        img_path = os.path.join(IMG_DIR, img_file)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img = img.resize((src_w, src_h), Image.LANCZOS)
            draw = ImageDraw.Draw(img)

            # Ground-truth box (blue)
            draw.rectangle(list(box_px), outline=(0, 80, 255), width=4)
            draw.text((box_px[0] + 4, box_px[1] - 18), "GT", fill=(0, 80, 255))

            # Decoded+un-letterboxed box (green)
            rec_box = (
                ox1 * src_w, oy1 * src_h,
                ox2 * src_w, oy2 * src_h,
            )
            draw.rectangle(list(rec_box), outline=(42, 223, 165), width=4)
            draw.text((rec_box[0] + 4, rec_box[1] - 18), "DECODED", fill=(42, 223, 165))

            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = os.path.join(OUT_DIR, f"{desc}.jpg")
            img.save(out_path, quality=90)
            print(f"   Saved → {out_path}")
        else:
            print(f"   [WARN] Image not found: {img_path} (skipping visual)")

    return pass_all


def main():
    passed = 0
    failed = 0
    for args in CASES:
        if run_case(*args):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("Some coordinate roundtrips failed — check the math above.")
        sys.exit(1)
    else:
        print("All coordinate roundtrips OK ✓")


if __name__ == "__main__":
    main()

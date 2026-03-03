#!/usr/bin/env python3
"""Export YOLO-CROWD .pt model to dynamic-batch ONNX.

The YOLO-CROWD model (https://github.com/zaki1003/YOLO-CROWD) uses YOLOv5 with
custom backbone/head layers (C3RFEM, MultiSEAM) that are not part of standard
ultralytics.  This script loads the model using YOLO-CROWD's own codebase and
exports a single-output ONNX with the detection-head grid pre-baked (decoded
pixel-space [cx, cy, w, h, obj_conf, cls_conf] for each anchor).

Output shape: [batch, 25200, 6]  (for 640×640 input, nc=1)
  - 25200 = 80×80×3 + 40×40×3 + 20×20×3 anchors
  - 6     = [cx_px, cy_px, w_px, h_px, objectness, cls0_score]
  - Coordinates already decoded (sigmoid + stride), in pixel space [0..640]

Usage (default — YOLO-CROWD source cloned automatically by 3_prepare_models.sh):
    python3 export_yolocrowd_to_onnx.py

Usage (custom source path):
    YOLO_CROWD_SRC=/path/to/YOLO-CROWD python3 export_yolocrowd_to_onnx.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# PyTorch 2.6+ changed torch.load default to weights_only=True which breaks
# old YOLOv5/YOLO-CROWD checkpoints that contain numpy arrays.
# Monkey-patch to restore the pre-2.6 behaviour for this trusted local file.
_orig_torch_load = torch.load
def _permissive_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _permissive_torch_load

# ---------------------------------------------------------------------------
# Locate YOLO-CROWD source (needed for custom layers C3RFEM, MultiSEAM)
# ---------------------------------------------------------------------------
_YOLO_CROWD_CANDIDATES = [
    os.environ.get("YOLO_CROWD_SRC", ""),
    "models/yolo-crowd-src",          # default: cloned by 3_prepare_models.sh
    "/yolo_crowd",                    # legacy Docker mount (kept for compatibility)
]
_yolo_crowd_src: str | None = None
for _candidate in _YOLO_CROWD_CANDIDATES:
    if _candidate and os.path.isfile(os.path.join(_candidate, "models", "yolo.py")):
        _yolo_crowd_src = _candidate
        break

if _yolo_crowd_src is None:
    print(
        "[CROWD] ERROR: YOLO-CROWD source not found.\n"
        "  Set YOLO_CROWD_SRC env var or mount the repo at /yolo_crowd in Docker."
    )
    sys.exit(1)

print(f"[CROWD] Using YOLO-CROWD source from: {_yolo_crowd_src}")
sys.path.insert(0, _yolo_crowd_src)

# Now we can import YOLO-CROWD's codebase
from models.experimental import attempt_load  # type: ignore[import]
from utils.activations import Hardswish, SiLU  # type: ignore[import]
import models as _yolo_crowd_models  # type: ignore[import]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHTS = "models/pt/yolo-crowd.pt"
ONNX_OUT = "models/pt/yolo-crowd-dynamic.onnx"
IMG_SIZE = (640, 640)


class _GridDecodeWrapper(nn.Module):
    """Wraps a YOLO-CROWD model to return only the decoded anchor tensor.

    The Detect layer returns ``(torch.cat(z, 1), x)`` in inference mode, where
    ``z`` is the decoded ``[B, 25200, 6]`` tensor and ``x`` is the list of raw
    feature maps.  We want exactly one ONNX output to keep the TRT engine
    simple, so this wrapper discards the raw feature maps.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]  # [B, 25200, 6] — decoded anchors only
        return out  # already a tensor (shouldn't happen in inference mode)


def export_onnx(weights: str = WEIGHTS, onnx_out: str = ONNX_OUT) -> None:
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[CROWD] ERROR: Weights not found: {weights_path}")
        sys.exit(1)

    onnx_path = Path(onnx_out)
    if onnx_path.exists():
        print(f"[CROWD] ONNX already exists, skipping export: {onnx_path}")
        return

    print(f"[CROWD] Loading weights from {weights_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = attempt_load(str(weights_path), map_location=device)  # type: ignore[call-arg]
    base_model.eval()

    # Apply export-friendly activations (same as YOLO-CROWD export.py)
    for _, m in base_model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, _yolo_crowd_models.common.Conv):
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()

    # Force inference mode → grid pre-computed → decoded output [B, 25200, 6]
    # export=False AND not training → Detect returns (cat_z, raw_x)
    base_model.model[-1].export = False

    wrapped = _GridDecodeWrapper(base_model)
    wrapped.eval()

    dummy = torch.zeros(1, 3, *IMG_SIZE, device=device)
    with torch.no_grad():
        _ = wrapped(dummy)  # dry run

    print(f"[CROWD] Exporting to {onnx_path} (opset 12, dynamic batch)...")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        verbose=False,
        opset_version=12,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={
            "images":  {0: "batch"},
            "output0": {0: "batch"},
        },
    )

    # PyTorch 2.x dynamo exporter may split weights into an external .data file
    # which TensorRT's ONNX parser cannot resolve.  Re-save with embedded tensors
    # to produce a single self-contained ONNX file TRT can load directly.
    import onnx  # type: ignore[import]
    proto = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(proto, str(onnx_path), save_as_external_data=False)

    # Remove the orphaned .data file if it was created
    data_file = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_file.exists():
        data_file.unlink()

    onnx.checker.check_model(onnx.load(str(onnx_path)))

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[CROWD] ONNX export success: {onnx_path} ({size_mb:.1f} MB)")
    print(f"[CROWD] Output: output0 = [batch, 25200, 6] — YOLOv5 decoded format")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLO-CROWD to dynamic-batch ONNX")
    parser.add_argument("--weights", default=WEIGHTS, help="Input .pt weights path")
    parser.add_argument("--onnx-out", default=ONNX_OUT, help="Output ONNX path")
    args = parser.parse_args()
    export_onnx(args.weights, args.onnx_out)

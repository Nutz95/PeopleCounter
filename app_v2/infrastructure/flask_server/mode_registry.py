"""Inference mode registry — constants and config detection logic.

All mode-to-model mappings, display labels, and overlay definitions live
here so that both the Flask server and the pipeline orchestrator share the
same source of truth without importing the full Flask stack.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Inference mode registry
# Each mode maps to {model_name: enabled}.  Keys mirror the model names used
# in pipeline.yaml → models.*  and in ModelBuilder.
# ---------------------------------------------------------------------------
_INFERENCE_MODES: dict[str, dict[str, bool]] = {
    "passthrough":          {"yolo_global": False, "yolo_tiles": False, "density": False},
    "density":              {"yolo_global": False, "yolo_tiles": False, "density": True},
    "yolo_global":          {"yolo_global": True,  "yolo_tiles": False, "density": False},
    "yolo_tiles":           {"yolo_global": False, "yolo_tiles": True,  "density": False},
    "density_yolo_global":  {"yolo_global": True,  "yolo_tiles": False, "density": True},
    "density_yolo_tiles":   {"yolo_global": False, "yolo_tiles": True,  "density": True},
}

# Maps model names to their corresponding preprocess branch keys in pipeline.yaml
_PREPROCESS_BRANCH_MAP: dict[str, str] = {
    "yolo_global": "yolo_global_preprocess",
    "yolo_tiles":  "yolo_tiles_preprocess",
    "density":     "density_preprocess",
}

# Human-readable labels surfaced to the frontend via /api/config
_MODE_LABELS: dict[str, str] = {
    "passthrough":         "Passthrough",
    "density":             "Densité",
    "yolo_global":         "YOLO Global",
    "yolo_tiles":          "YOLO Tiling",
    "density_yolo_global": "YOLO Global + Densité",
    "density_yolo_tiles":  "YOLO Tiling + Densité",
}

# Which overlay checkboxes are relevant per mode (matched to frontend toggle IDs)
_MODE_OVERLAYS: dict[str, list[str]] = {
    "passthrough":         [],
    "density":             ["heatmap"],
    "yolo_global":         ["bbox", "seg"],
    "yolo_tiles":          ["bbox", "seg"],
    "density_yolo_global": ["bbox", "seg", "heatmap"],
    "density_yolo_tiles":  ["bbox", "seg", "heatmap"],
}


def detect_mode_from_config(config: dict[str, Any]) -> str:
    """Return the mode string matching the enabled models in a pipeline config."""
    models_cfg = config.get("models", {})
    state = {
        name: bool(models_cfg.get(name, {}).get("enabled", False))
        for name in ("yolo_global", "yolo_tiles", "density")
    }
    for mode_name, mode_state in _INFERENCE_MODES.items():
        if mode_state == state:
            return mode_name
    return "passthrough"  # fallback

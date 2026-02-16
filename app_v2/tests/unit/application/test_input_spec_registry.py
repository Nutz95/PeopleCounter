from __future__ import annotations

from typing import Any

import pytest

from app_v2.application.input_spec_registry import InputSpecRegistry


def test_input_spec_registry_builds_specs_from_config() -> None:
    metadata: dict[str, Any] = {
        "models": {
            "yolo_global": {"enabled": True},
            "yolo_tiles": {"enabled": True},
        },
        "preprocess": {
            "yolo_global": {
                "target_width": 800,
                "target_height": 800,
                "mode": "global",
                "overlap": 0.0,
            },
            "yolo_tiles": {
                "target_width": 640,
                "target_height": 640,
                "mode": "tiles",
                "overlap": 0.25,
            },
        },
    }

    registry = InputSpecRegistry()
    registry.configure(metadata)

    global_spec = registry.by_model("yolo_global")
    tiles_spec = registry.by_model("yolo_tiles")

    assert global_spec is not None
    assert global_spec.target_width == 800
    assert global_spec.target_height == 800
    assert global_spec.mode == "global"
    assert global_spec.overlap == 0.0

    assert tiles_spec is not None
    assert tiles_spec.target_width == 640
    assert tiles_spec.target_height == 640
    assert tiles_spec.mode == "tiles"
    assert tiles_spec.overlap == 0.25


def test_input_spec_registry_requires_preprocess_keys() -> None:
    metadata: dict[str, Any] = {
        "models": {"yolo_global": {"enabled": True}},
        "preprocess": {"yolo_global": {"target_width": 640, "mode": "global", "overlap": 0.0}},
    }

    registry = InputSpecRegistry()
    with pytest.raises(ValueError, match="missing required keys"):
        registry.configure(metadata)


def test_input_spec_registry_needs_preprocess_block() -> None:
    metadata: dict[str, Any] = {"models": {"yolo_global": {"enabled": True}}}

    registry = InputSpecRegistry()
    with pytest.raises(ValueError, match="preprocess configuration is required"):
        registry.configure(metadata)

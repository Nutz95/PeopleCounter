from __future__ import annotations

from typing import Any

import pytest

from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.core.preprocessor_types import PreprocessMode


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
    assert global_spec.mode is PreprocessMode.GLOBAL
    assert global_spec.overlap == 0.0

    assert tiles_spec is not None
    assert tiles_spec.target_width == 640
    assert tiles_spec.target_height == 640
    assert tiles_spec.mode is PreprocessMode.TILES
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


def test_input_spec_registry_honors_preprocess_branch_toggles() -> None:
    metadata: dict[str, Any] = {
        "models": {
            "yolo_global": {"enabled": True},
            "yolo_tiles": {"enabled": True},
        },
        "preprocess_branches": {
            "yolo_global_preprocess": True,
            "yolo_tiles_preprocess": False,
        },
        "preprocess": {
            "yolo_global": {
                "target_width": 640,
                "target_height": 640,
                "mode": "global",
                "overlap": 0.0,
            },
            "yolo_tiles": {
                "target_width": 640,
                "target_height": 640,
                "mode": "tiles",
                "overlap": 0.2,
            },
        },
    }

    registry = InputSpecRegistry()
    registry.configure(metadata)

    assert registry.by_model("yolo_global") is not None
    assert registry.by_model("yolo_tiles") is None


def test_input_spec_registry_rejects_invalid_preprocess_branches_type() -> None:
    metadata: dict[str, Any] = {
        "models": {"yolo_global": {"enabled": True}},
        "preprocess_branches": ["not", "a", "mapping"],
        "preprocess": {
            "yolo_global": {
                "target_width": 640,
                "target_height": 640,
                "mode": "global",
                "overlap": 0.0,
            }
        },
    }

    registry = InputSpecRegistry()
    with pytest.raises(ValueError, match="preprocess_branches must be a mapping"):
        registry.configure(metadata)


def test_input_spec_registry_rejects_invalid_mode_value() -> None:
    metadata: dict[str, Any] = {
        "models": {"yolo_global": {"enabled": True}},
        "preprocess": {
            "yolo_global": {
                "target_width": 640,
                "target_height": 640,
                "mode": "invalid-mode",
                "overlap": 0.0,
            }
        },
    }

    registry = InputSpecRegistry()
    with pytest.raises(ValueError, match="Unsupported preprocess mode"):
        registry.configure(metadata)

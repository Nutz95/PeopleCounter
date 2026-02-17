from __future__ import annotations

from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.config import load_pipeline_config


def test_pipeline_preprocess_configuration_is_valid() -> None:
    config = load_pipeline_config()
    streams = config.get("streams")
    assert isinstance(streams, dict) and streams, "streams block must exist in pipeline config"
    for key in ("yolo_global_preprocess", "yolo_tiles_preprocess", "density_preprocess"):
        assert key in streams, f"streams.{key} must be configured"

    preprocess_branches = config.get("preprocess_branches")
    assert isinstance(preprocess_branches, dict), "preprocess_branches block must exist in pipeline config"
    for key in ("yolo_global_preprocess", "yolo_tiles_preprocess", "density_preprocess"):
        assert key in preprocess_branches, f"preprocess_branches.{key} must be configured"
        assert isinstance(preprocess_branches[key], bool), f"preprocess_branches.{key} must be boolean"

    tensor_pool = config.get("tensor_pool")
    assert isinstance(tensor_pool, dict), "tensor_pool block must exist in pipeline config"
    assert int(tensor_pool.get("max_per_key", 0)) > 0, "tensor_pool.max_per_key must be > 0"

    trt_execution = config.get("trt_execution")
    assert isinstance(trt_execution, dict), "trt_execution block must exist in pipeline config"
    assert isinstance(trt_execution.get("enabled", False), bool), "trt_execution.enabled must be boolean"
    assert isinstance(trt_execution.get("strict_shape_check", True), bool), "trt_execution.strict_shape_check must be boolean"

    preprocess = config.get("preprocess")
    assert isinstance(preprocess, dict) and preprocess, "preprocess block must exist in pipeline config"

    for model_name, entry in preprocess.items():
        assert isinstance(entry, dict), f"preprocess.{model_name} must be a mapping"
        assert "target_width" in entry and "target_height" in entry, "dimensions must be defined"
        assert "mode" in entry and entry["mode"], "mode must be a non-empty string"
        assert "overlap" in entry, "overlap must be provided"

        width = entry["target_width"]
        height = entry["target_height"]
        overlap = entry["overlap"]
        assert isinstance(width, int) and width > 0, "target_width must be a positive int"
        assert isinstance(height, int) and height > 0, "target_height must be a positive int"
        assert isinstance(overlap, (int, float)) and overlap >= 0.0, "overlap must be non-negative"

    registry = InputSpecRegistry()
    registry.configure(config)
    assert registry.all_specs(), "Registry should build specs for enabled preprocess entries"

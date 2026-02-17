from __future__ import annotations

from app_v2.infrastructure.preprocess_stream_manager import PreprocessStreamManager


def test_preprocess_stream_manager_builds_model_stream_map() -> None:
    manager = PreprocessStreamManager()
    metadata = {
        "streams": {
            "yolo": 1,
            "density": 2,
            "yolo_global_preprocess": 3,
            "yolo_tiles_preprocess": 4,
            "density_preprocess": 5,
        },
        "preprocess": {
            "yolo_global": {"mode": "global"},
            "yolo_tiles": {"mode": "tiles"},
            "density": {"mode": "tiles"},
            "other": {"mode": "global"},
        },
    }

    manager.configure(metadata)

    assert manager.stream_for_model("yolo_global") == 3
    assert manager.stream_for_model("yolo_tiles") == 4
    assert manager.stream_for_model("density") == 5
    assert manager.stream_for_model("other") == 0


def test_preprocess_stream_manager_defaults_to_zero_when_missing() -> None:
    manager = PreprocessStreamManager()
    manager.configure({"preprocess": {"yolo_global": {"mode": "global"}}})

    assert manager.stream_for_model("yolo_global") == 0
    assert manager.stream_for_model("unknown") == 0

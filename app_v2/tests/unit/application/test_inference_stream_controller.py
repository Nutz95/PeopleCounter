from __future__ import annotations

from app_v2.application.inference_stream_controller import InferenceStreamController


def test_inference_stream_controller_tracks_models_and_streams() -> None:
    config = {
        "models": {
            "yolo_tiles": {"enabled": True, "engine": "models/yolo_tiles.engine"},
            "density": {"enabled": False, "engine": "models/density.engine"},
        },
        "inference_streams": {"yolo_tiles": True, "density": False},
    }
    controller = InferenceStreamController(config)

    assert controller.is_model_enabled("yolo_tiles")
    assert not controller.is_model_enabled("density")
    assert controller.get_engine_path("yolo_tiles") == "models/yolo_tiles.engine"
    assert controller.is_stream_available("yolo_tiles")

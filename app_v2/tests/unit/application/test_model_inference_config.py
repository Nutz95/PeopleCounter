from __future__ import annotations

from app_v2.config import load_model_inference_config


def test_model_inference_config_contains_yolo_person_filtering_defaults() -> None:
    cfg = load_model_inference_config()

    assert "yolo_global" in cfg
    assert "yolo_tiles" in cfg

    for model_name in ("yolo_global", "yolo_tiles"):
        model_cfg = cfg[model_name]
        assert model_cfg["person_class_id"] == 0
        assert model_cfg["confidence_threshold"] > 0.0
        assert model_cfg["nms_iou_threshold"] > 0.0
        assert model_cfg["segmentation_threshold"] > 0.0
        assert model_cfg["max_detections"] > 0
        assert model_cfg["class_whitelist"] == [0]

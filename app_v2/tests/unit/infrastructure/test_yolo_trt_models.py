from __future__ import annotations

from typing import Any

from app_v2.infrastructure.yolo_global_trt import YoloGlobalTRT
from app_v2.infrastructure.yolo_tiling_trt import YoloTilingTRT


class DummyExecutionContext:
    def __init__(self) -> None:
        self.bound: list[str] = []
        self.released: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def bind_stream(self, stream_key: str) -> None:
        self.bound.append(stream_key)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(inputs))
        return {"status": "ok", "segmentation": {"mask": None}}

    def release_stream(self, stream_key: str) -> None:
        self.released.append(stream_key)


def test_yolo_global_trt_infer_includes_metrics_and_person_filter() -> None:
    ctx = DummyExecutionContext()
    model = YoloGlobalTRT(
        ctx,
        stream_id=1,
        inference_params={"person_class_id": 0, "class_whitelist": [0], "confidence_threshold": 0.25},
    )

    result = model.infer(frame_id=7, inputs=["tensor0"])

    assert result["frame_id"] == 7
    assert result["person_class_id"] == 0
    assert result["class_whitelist"] == [0]
    assert result["input_count"] == 1
    assert "yolo_inference_ms" in result
    assert float(result["yolo_inference_ms"]) >= 0.0
    assert ctx.bound == ["model:yolo_global"]
    assert ctx.released == ["model:yolo_global"]


def test_yolo_tiles_trt_infer_includes_metrics_and_tile_count() -> None:
    ctx = DummyExecutionContext()
    model = YoloTilingTRT(
        ctx,
        stream_id=2,
        inference_params={"person_class_id": 0, "class_whitelist": [0], "confidence_threshold": 0.25},
    )

    result = model.infer(frame_id=9, inputs=["t0", "t1", "t2"])

    assert result["frame_id"] == 9
    assert result["person_class_id"] == 0
    assert result["class_whitelist"] == [0]
    assert result["tile_count"] == 3
    assert "yolo_inference_ms" in result
    assert float(result["yolo_inference_ms"]) >= 0.0
    assert ctx.bound == ["model:yolo_tiles"]
    assert ctx.released == ["model:yolo_tiles"]

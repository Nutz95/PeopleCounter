from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from app_v2.application.pipeline_orchestrator import PipelineOrchestrator
from app_v2.core.preprocessor_types import PreprocessOutput
from app_v2.core.frame_source import FrameSource
from app_v2.core.inference_model import InferenceModel
from app_v2.core.result_publisher import ResultPublisher
from app_v2.infrastructure.gpu_ring_buffer import GpuFrame, GpuPixelFormat


class DummyPublisher(ResultPublisher):
    def __init__(self) -> None:
        self.published: list[tuple[int, Sequence[dict[str, Any]]]] = []

    def publish(self, frame_id: int, payload: Sequence[dict[str, Any]]) -> None:
        self.published.append((frame_id, payload))


class DummyFrameSource(FrameSource):
    def __init__(self, total_frames: int) -> None:
        self.total_frames = total_frames
        self.frames_consumed = 0
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def next_frame(self, frame_id: int) -> GpuFrame:
        if self.frames_consumed >= self.total_frames:
            raise StopIteration
        self.frames_consumed += 1
        return GpuFrame(
            width=16,
            height=16,
            pixel_format=GpuPixelFormat.NV12,
            device_ptr_y=frame_id,
            device_ptr_uv=frame_id + 1,
            pitch=16,
            timestamp_ns=int(time.time_ns()),
            frame_id=frame_id,
        )


class DummyPreprocessor:
    def configure(self, metadata: dict[str, Any]) -> None:
        del metadata

    def build_output(self, frame_id: int, frame: Any) -> PreprocessOutput:
        del frame
        return PreprocessOutput(
            frame_id=frame_id,
            plans={},
            model_inputs={
                "yolo_global": ("g0",),
                "yolo_tiles": ("t0", "t1"),
            },
        )


class SpyModel(InferenceModel):
    def __init__(self, name: str) -> None:
        self._name = name
        self._stream_id = 0
        self.calls: list[list[Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def stream_id(self) -> int:
        return self._stream_id

    def warm_up(self, batch_size: int) -> None:
        del batch_size

    def infer(self, frame_id: int, inputs: Sequence[Any], *, preprocess_events: Sequence[Any] | None = None) -> dict[str, Any]:
        del frame_id
        del preprocess_events
        captured = list(inputs)
        self.calls.append(captured)
        return {"model": self._name}

    def close(self) -> None:
        return


def test_pipeline_orchestrator_runs_one_cycle() -> None:
    frame_source = DummyFrameSource(total_frames=1)
    publisher = DummyPublisher()
    orchestrator = PipelineOrchestrator(frame_source=frame_source, max_frames=1, publisher=publisher)
    orchestrator.preprocessor = DummyPreprocessor()
    orchestrator._models = [SpyModel("yolo_global")]

    orchestrator.run()

    assert frame_source.frames_consumed == 1
    assert publisher.published, "Pipeline should publish at least one payload"


def test_pipeline_orchestrator_routes_inputs_by_model_name() -> None:
    frame_source = DummyFrameSource(total_frames=1)
    publisher = DummyPublisher()
    orchestrator = PipelineOrchestrator(frame_source=frame_source, max_frames=1, publisher=publisher)
    orchestrator.preprocessor = DummyPreprocessor()

    global_model = SpyModel("yolo_global")
    tiles_model = SpyModel("yolo_tiles")
    orchestrator._models = [global_model, tiles_model]

    orchestrator.run()

    assert global_model.calls == [["g0"]]
    assert tiles_model.calls == [["t0", "t1"]]

from __future__ import annotations

import time

from app_v2.application.frame_scheduler import FrameScheduler
from app_v2.application.inference_stream_controller import InferenceStreamController
from app_v2.application.model_builder import ModelBuilder
from app_v2.application.processing_graph import ProcessingGraph
from app_v2.application.performance_tracker import PerformanceTracker
from app_v2.application.result_aggregator import ResultAggregator
from app_v2.config import load_pipeline_config
from app_v2.core.fusion_strategy import FusionStrategy, SimpleFusionStrategy
from app_v2.core.frame_source import FrameSource
from app_v2.core.result_publisher import ResultPublisher
from app_v2.infrastructure.cuda_preprocessor import CudaPreprocessor
from app_v2.infrastructure.flask_stream_server import FlaskStreamServer
from app_v2.infrastructure.stream_pool import SimpleStreamPool
from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning


class PipelineOrchestrator:
    """Drives the NVDEC loop, preprocessors, inference contexts, and result fusion."""

    def __init__(
        self,
        frame_source: FrameSource,
        max_frames: int | None = None,
        publisher: ResultPublisher | None = None,
        fusion_strategy: FusionStrategy | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.config = load_pipeline_config()
        self.scheduler = FrameScheduler()
        self.processing_graph = ProcessingGraph()
        self.preprocessor = CudaPreprocessor()
        self.stream_pool = SimpleStreamPool()
        self.inference_controller = InferenceStreamController(self.config)
        self.model_builder = ModelBuilder(self.config, self.inference_controller, self.stream_pool)
        self.fusion_strategy = fusion_strategy or SimpleFusionStrategy()
        self.publisher = publisher or FlaskStreamServer()
        self.aggregator = ResultAggregator(self.fusion_strategy, self.publisher)
        self.performance_tracker = PerformanceTracker()
        self.max_frames = max_frames
        self._models = self.model_builder.build_models()
        self._frame_counter = 0
        self._running = False
        log_info(LogChannel.GLOBAL, "PipelineOrchestrator components initialized")

    def run(self) -> None:
        log_info(LogChannel.GLOBAL, f"Starting app_v2 pipeline with config {self.config}")
        log_info(LogChannel.GLOBAL, "Frame scheduling will track frame IDs until fusion completes.")
        self._start_publisher()
        self.frame_source.connect()
        self.preprocessor.configure(self.config)
        self._running = True
        try:
            while self._should_continue():
                frame_id = self.scheduler.schedule(None)
                with self.performance_tracker.stage(frame_id, "nvdec"):
                    frame = self.frame_source.next_frame(frame_id)
                output = self.preprocessor.build_output(frame_id, frame)
                self.aggregator.attach_telemetry(frame_id, output.telemetry)
                self.aggregator.attach_release_hook(frame_id, output.release_all)
                for model in self._models:
                    processed = output.flatten_inputs(model.name)
                    with self.performance_tracker.stage(frame_id, model.name):
                        prediction = model.infer(frame_id, processed, preprocess_events=list(output.cuda_events.values()))
                        if isinstance(prediction, dict):
                            prediction["_inference_done_ns"] = int(time.time_ns())
                        self.processing_graph.register(model.name, {"frame_id": frame_id})
                        self.aggregator.collect(frame_id, prediction)
                self.scheduler.acknowledge(frame_id)
                self.performance_tracker.clear(frame_id)
                self._frame_counter += 1
        except StopIteration:
            log_info(LogChannel.GLOBAL, "Frame source signaled completion")
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"Pipeline aborted during iteration: {exc}")
        finally:
            self._running = False
            self.frame_source.disconnect()
            self._shutdown()

    def _should_continue(self) -> bool:
        if not self._running:
            return False
        if self.max_frames is None:
            return True
        return self._frame_counter < self.max_frames

    def _start_publisher(self) -> None:
        if not hasattr(self.publisher, "start"):
            return
        try:
            self.publisher.start()
            log_info(LogChannel.GLOBAL, f"FlaskStreamServer started on {self.publisher.host}:{self.publisher.port}")
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"FlaskStreamServer failed to start: {exc}")

    def _shutdown(self) -> None:
        for model in self._models:
            try:
                model.close()
            except Exception as exc:
                log_warning(LogChannel.GLOBAL, f"Model {model.name} closed with error: {exc}")

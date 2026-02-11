from __future__ import annotations

from app_v2.config import load_pipeline_config
from app_v2.core.fusion_strategy import SimpleFusionStrategy
from app_v2.core.frame_source import FrameSource
from app_v2.application.frame_scheduler import FrameScheduler
from app_v2.application.processing_graph import ProcessingGraph
from app_v2.application.performance_tracker import PerformanceTracker
from app_v2.application.result_aggregator import ResultAggregator
from app_v2.infrastructure.flask_stream_server import FlaskStreamServer
from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning


class PipelineOrchestrator:
    """Drives the frame loop, schedules models, and publishes fused results."""

    def __init__(self, frame_source: FrameSource) -> None:
        self.frame_source = frame_source
        self.fusion_strategy = SimpleFusionStrategy()
        self.config = load_pipeline_config()
        self.scheduler = FrameScheduler()
        self.processing_graph = ProcessingGraph()
        self.publisher = FlaskStreamServer()
        self.aggregator = ResultAggregator(self.fusion_strategy, self.publisher)
        self.performance_tracker = PerformanceTracker()

    def run(self) -> None:
        log_info(LogChannel.GLOBAL, f"Starting app_v2 pipeline with config {self.config}")
        log_info(LogChannel.GLOBAL, "Frame scheduling will track frame IDs until fusion completes.")
        self.frame_source.connect()
        try:
            with self.performance_tracker.stage(0, "bootstrap"):
                log_info(LogChannel.GLOBAL, "Pipeline scaffolding is ready. Replace the loop with TensorRT execution code.")
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"Pipeline aborted during scaffolding: {exc}")
        finally:
            self.frame_source.disconnect()
            log_info(LogChannel.GLOBAL, "Frame source disconnected.")

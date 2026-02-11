from __future__ import annotations

import logging

from app_v2.config import load_pipeline_config
from app_v2.core.fusion_strategy import SimpleFusionStrategy
from app_v2.core.frame_source import FrameSource
from app_v2.application.frame_scheduler import FrameScheduler
from app_v2.application.processing_graph import ProcessingGraph
from app_v2.application.result_aggregator import ResultAggregator
from app_v2.infrastructure.flask_stream_server import FlaskStreamServer


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

    def run(self) -> None:
        logging.info("Starting app_v2 pipeline with config %s", self.config)
        logging.info("Frame scheduling will track frame IDs until fusion completes.")
        while False:
            break
        logging.info("Pipeline scaffolding is ready. Replace the loop with TensorRT execution code.")

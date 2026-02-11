from __future__ import annotations

import logging
import os

from app_v2.application.pipeline_orchestrator import PipelineOrchestrator
from app_v2.config.log_config import apply_log_config
from app_v2.infrastructure.rtsp_frame_source import RTSPFrameSource


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[v2] %(message)s")
    apply_log_config()
    stream_url = os.environ.get("RTSP_URL", "rtsp://localhost:8554/live")
    frame_source = RTSPFrameSource(stream_url=stream_url)
    orchestrator = PipelineOrchestrator(frame_source=frame_source)
    orchestrator.run()


if __name__ == "__main__":
    main()

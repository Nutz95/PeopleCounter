from __future__ import annotations

import logging
import os
import time

from app_v2.application.pipeline_orchestrator import PipelineOrchestrator
from app_v2.config.log_config import apply_log_config
from app_v2.infrastructure.rtsp_frame_source import RTSPFrameSource


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[v2] %(message)s")
    apply_log_config()
    stream_url = os.environ.get("RTSP_URL", "rtsp://localhost:8554/live")

    # Outer reconnect loop: when the source disconnects (camera unplugged, timeout…)
    # the orchestrator exits cleanly.  We wait briefly and restart so the container
    # stays alive and resumes automatically when the stream comes back.
    reconnect_delay = 2.0
    _MAX_RECONNECT_DELAY = 30.0
    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            logging.info("[v2] Reconnect attempt %d — waiting %.0f s…", attempt, reconnect_delay)
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_DELAY)
        logging.info("[v2] Connecting to %s", stream_url)
        try:
            frame_source = RTSPFrameSource(stream_url=stream_url)
            orchestrator = PipelineOrchestrator(frame_source=frame_source)
            orchestrator.run()
        except KeyboardInterrupt:
            logging.info("[v2] Interrupted — shutting down")
            break
        except Exception as exc:
            logging.warning("[v2] Pipeline exited with error: %s — will retry", exc)
        else:
            # Clean exit (stream ended, not an error) — restart to pick up again
            logging.info("[v2] Pipeline ended cleanly — restarting")
        reconnect_delay = 2.0  # reset delay after any clean handshake


if __name__ == "__main__":
    main()

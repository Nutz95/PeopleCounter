from __future__ import annotations

from typing import Sequence

from app_v2.core.result_publisher import ResultPublisher


class FlaskStreamServer(ResultPublisher):
    """Publishes fused payloads to the Flask backend without inline inference."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port

    def publish(self, frame_id: int, payload: Sequence[dict[str, object]]) -> None:
        """Push metadata downstream (mask timing, detection counts, etc.)."""
        print(f"[V2] Publishing frame {frame_id} with {len(payload)} entries")

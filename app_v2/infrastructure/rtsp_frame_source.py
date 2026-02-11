from __future__ import annotations

from typing import Any

from app_v2.core.frame_source import FrameSource


class RTSPFrameSource(FrameSource):
    """Concrete frame source that consumes a remote RTSP or MJPEG stream."""

    def __init__(self, stream_url: str) -> None:
        self.stream_url = stream_url
        self.connected = False

    def connect(self) -> None:
        """Open the RTSP connection and start fetching frames."""
        self.connected = True

    def disconnect(self) -> None:
        """Tear down the RTSP connection cleanly."""
        self.connected = False

    def next_frame(self) -> tuple[int, Any]:
        """Return the next frame_id and raw image payload."""
        if not self.connected:
            raise RuntimeError("Frame source is disconnected")
        raise NotImplementedError("RTSPFrameSource.next_frame must be implemented to pull from the stream")

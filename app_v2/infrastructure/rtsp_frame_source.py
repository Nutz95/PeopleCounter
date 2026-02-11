from __future__ import annotations

from typing import Any

from app_v2.core.frame_source import FrameSource
from app_v2.infrastructure.nvdec_decoder import NvdecDecoder
from logger.filtered_logger import LogChannel, info as log_info


class RTSPFrameSource(FrameSource):
    """Concrete frame source that consumes a remote RTSP or MJPEG stream."""

    def __init__(self, stream_url: str) -> None:
        self.stream_url = stream_url
        self.connected = False
        self.decoder = NvdecDecoder(stream_url)
        log_info(LogChannel.GLOBAL, "RTSPFrameSource initialized")

    def connect(self) -> None:
        """Open the RTSP connection and start fetching frames."""
        log_info(LogChannel.GLOBAL, "RTSPFrameSource connecting")
        self.decoder.start()
        self.connected = True

    def disconnect(self) -> None:
        """Tear down the RTSP connection cleanly."""
        log_info(LogChannel.GLOBAL, "RTSPFrameSource disconnecting")
        self.decoder.stop()
        self.connected = False

    def next_frame(self) -> tuple[int, Any]:
        """Return the next frame_id and raw image payload."""
        if not self.connected:
            raise RuntimeError("Frame source is disconnected")
        raise NotImplementedError("RTSPFrameSource.next_frame must be implemented to pull from the stream")

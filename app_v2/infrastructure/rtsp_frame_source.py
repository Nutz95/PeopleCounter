from __future__ import annotations

import time
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

    def next_frame(self, frame_id: int) -> Any:
        """Return a decoded GPU frame for the scheduled frame_id."""
        if not self.connected:
            raise RuntimeError("Frame source is disconnected")
        slot = self.decoder.decode_next_into_ring(frame_id=frame_id, timestamp_ns=int(time.time_ns()))
        popped = self.decoder.ring.pop_ready(block=True)
        if popped is None:
            raise RuntimeError("No decoded frame available")
        slot, frame = popped
        self.decoder.ring.release(slot)
        return frame

from __future__ import annotations

from typing import Any

from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning


class NvdecDecoder:
    """Represents an NVDEC-powered decoder that leaves buffers resident on the GPU."""

    def __init__(self, stream_url: str) -> None:
        self.stream_url = stream_url
        self._connected = False
        log_info(LogChannel.GLOBAL, f"NvdecDecoder created for stream {stream_url}")

    def start(self) -> None:
        log_info(LogChannel.GLOBAL, "Starting NVDEC decoder")
        self._connected = True

    def stop(self) -> None:
        log_info(LogChannel.GLOBAL, "Stopping NVDEC decoder")
        self._connected = False

    def decode_next(self) -> Any:
        if not self._connected:
            log_warning(LogChannel.GLOBAL, "NVDEC decoder requested before start")
            raise RuntimeError("Decoder not started")
        raise NotImplementedError("NVDEC decoding logic must live in GPU-specific layer")
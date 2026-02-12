from __future__ import annotations

from dataclasses import dataclass

from app_v2.infrastructure.gpu_ring_buffer import GpuFrame, GpuPixelFormat, GpuRingBuffer
from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning


@dataclass(frozen=True, slots=True)
class NvdecDecodeConfig:
    """NVDEC decode settings for the GPU-only pipeline."""

    ring_capacity: int = 8


class NvdecDecoder:
    """NVDEC-powered decoder that keeps decoded surfaces resident on the GPU.

    Implementation note:
    - The real decode path requires NVIDIA Video Codec SDK bindings (e.g., PyNvCodec/VPF or a custom extension).
    - This class defines the API so the rest of app_v2 can be built/tested without CPU downloads.
    """

    def __init__(self, stream_url: str, config: NvdecDecodeConfig | None = None) -> None:
        self.stream_url = stream_url
        self._connected = False
        self._config = config or NvdecDecodeConfig()
        self._ring = GpuRingBuffer(self._config.ring_capacity)
        log_info(LogChannel.GLOBAL, f"NvdecDecoder created for stream {stream_url}")
        log_info(LogChannel.GLOBAL, f"NVDEC ring capacity: {self._ring.capacity}")

    @property
    def ring(self) -> GpuRingBuffer:
        return self._ring

    def start(self) -> None:
        log_info(LogChannel.GLOBAL, "Starting NVDEC decoder")
        self._connected = True

    def stop(self) -> None:
        log_info(LogChannel.GLOBAL, "Stopping NVDEC decoder")
        self._connected = False

    def decode_next_into_ring(self, *, frame_id: int, timestamp_ns: int | None = None) -> int:
        """Decode the next frame into a ring-buffer slot and return the slot index.

        The committed frame MUST be GPU-resident (NV12) and MUST NOT allocate CPU arrays.
        """
        if not self._connected:
            log_warning(LogChannel.GLOBAL, "NVDEC decoder requested before start")
            raise RuntimeError("Decoder not started")

        slot = self._ring.acquire(block=True)
        if slot is None:
            raise RuntimeError("NVDEC ring buffer acquire timed out")

        # TODO(NVDEC): Replace this stub with PyNvCodec/VPF (or custom bindings) that output NV12 device pointers.
        stub_frame = GpuFrame(
            width=0,
            height=0,
            pixel_format=GpuPixelFormat.NV12,
            device_ptr_y=None,
            device_ptr_uv=None,
            pitch=None,
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
        )
        self._ring.commit(slot, stub_frame)
        return slot
from __future__ import annotations

from dataclasses import dataclass
import time
import threading
from typing import Any

from app_v2.core.frame_source import FrameSource
from app_v2.infrastructure.gpu_ring_buffer import GpuFrame, GpuPixelFormat
from app_v2.infrastructure.nvdec_decoder import NvdecDecoder
from app_v2.kernels.nv12_cuda_bridge import _copy_plane_2d_async, _extract_device_pointer
from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class _StableNv12Slot:
    """One stable GPU-backed NV12 frame slot owned by RTSPFrameSource."""

    y_plane: Any | None = None
    uv_plane: Any | None = None
    width: int = 0
    height: int = 0
    pitch: int = 0
    timestamp_ns: int | None = None
    telemetry: Any | None = None


class RTSPFrameSource(FrameSource):
    """RTSP frame source with background NVDEC and stable GPU frame slots.

    The decode thread continuously pulls surfaces from NVDEC at stream cadence,
    copies them into a small pool of owned NV12 CUDA buffers, and publishes only
    the latest stable frame to the consumer. This prevents inference latency from
    throttling calls into the decoder and helps keep NVDEC synchronized.
    """

    _STABLE_SLOT_COUNT = 4

    def __init__(self, stream_url: str) -> None:
        self.stream_url = stream_url
        self.connected = False
        self.decoder = NvdecDecoder(stream_url)

        self._decode_thread: threading.Thread | None = None
        self._decode_cond = threading.Condition()
        self._latest_seq: int = 0
        self._delivered_seq: int = 0
        self._latest_slot_idx: int | None = None
        self._consumer_slot_idx: int | None = None
        self._write_cursor: int = 0
        self._decode_counter: int = 0
        self._stable_slots: list[_StableNv12Slot] = [
            _StableNv12Slot() for _ in range(self._STABLE_SLOT_COUNT)
        ]
        self._copy_stream = (
            torch.cuda.Stream() if torch is not None and torch.cuda.is_available() else None
        )
        self._async_decode_enabled = self._copy_stream is not None
        log_info(LogChannel.GLOBAL, "RTSPFrameSource initialized")

    def connect(self) -> None:
        """Open the RTSP connection and start background decoding."""
        log_info(LogChannel.GLOBAL, "RTSPFrameSource connecting")
        self.decoder.start()
        self.connected = True
        if self._async_decode_enabled and (self._decode_thread is None or not self._decode_thread.is_alive()):
            self._decode_thread = threading.Thread(
                target=self._decode_loop,
                daemon=True,
                name="rtsp-nvdec",
            )
            self._decode_thread.start()

    def disconnect(self) -> None:
        """Tear down the RTSP connection cleanly."""
        log_info(LogChannel.GLOBAL, "RTSPFrameSource disconnecting")
        self.connected = False
        with self._decode_cond:
            self._decode_cond.notify_all()
        self.decoder.stop()
        if self._decode_thread is not None and self._decode_thread.is_alive():
            self._decode_thread.join(timeout=1.0)

    def next_frame(self, frame_id: int) -> Any:
        """Return the latest stable decoded GPU frame for the scheduled frame_id."""
        if not self.connected:
            raise RuntimeError("Frame source is disconnected")

        if not self._async_decode_enabled:
            # Fallback path for environments without CUDA-enabled PyTorch.
            decode_slot = self.decoder.decode_next_into_ring(frame_id=frame_id)
            popped = self.decoder.ring.pop_ready(block=True)
            if popped is None:
                raise RuntimeError("No decoded frame available")
            popped_slot, frame = popped
            self.decoder.ring.release(popped_slot)
            # Defensive cleanup if decode slot != popped slot in a future impl.
            if decode_slot != popped_slot:
                self.decoder.ring.release(decode_slot)
            return frame

        wait_start_ns = time.perf_counter_ns()
        with self._decode_cond:
            # Consumer has finished with the previously delivered slot.
            self._consumer_slot_idx = None
            self._decode_cond.notify_all()

            while self.connected and self._latest_seq == self._delivered_seq:
                self._decode_cond.wait(timeout=1.0)

            if not self.connected:
                raise RuntimeError("Frame source is disconnected")

            slot_idx = self._latest_slot_idx
            if slot_idx is None:
                raise RuntimeError("No decoded frame available")

            self._delivered_seq = self._latest_seq
            self._consumer_slot_idx = slot_idx
            slot = self._stable_slots[slot_idx]
        wait_ms = (time.perf_counter_ns() - wait_start_ns) / 1_000_000.0

        telemetry = slot.telemetry
        if telemetry is not None:
            telemetry.frame_id = frame_id
            frame_age_ms = 0.0
            if isinstance(slot.timestamp_ns, int) and slot.timestamp_ns > 0:
                frame_age_ms = max(0.0, (time.time_ns() - slot.timestamp_ns) / 1_000_000.0)
            telemetry.add_metrics(
                {
                    "frame_source_wait_latest_ms": float(wait_ms),
                    "frame_source_age_at_consume_ms": float(frame_age_ms),
                }
            )

        return GpuFrame(
            width=slot.width,
            height=slot.height,
            pixel_format=GpuPixelFormat.NV12,
            device_ptr_y=int(slot.y_plane.data_ptr()) if slot.y_plane is not None else None,
            device_ptr_uv=int(slot.uv_plane.data_ptr()) if slot.uv_plane is not None else None,
            pitch=slot.pitch,
            timestamp_ns=slot.timestamp_ns,
            frame_id=frame_id,
            telemetry=telemetry,
        )

    # ------------------------------------------------------------------
    # Background decode loop
    # ------------------------------------------------------------------

    def _decode_loop(self) -> None:
        consecutive_errors = 0
        while self.connected:
            self._decode_counter += 1
            decode_slot: int | None = None
            try:
                decode_slot = self.decoder.decode_next_into_ring(frame_id=self._decode_counter)
                popped = self.decoder.ring.pop_ready(block=True)
                if popped is None:
                    if decode_slot is not None:
                        self.decoder.ring.release(decode_slot)
                    continue

                popped_slot, frame = popped
                decode_slot = None  # popped_slot now owns the release responsibility.
                stable_idx = self._reserve_writable_slot()
                if stable_idx is None:
                    self.decoder.ring.release(popped_slot)
                    continue

                self._copy_frame_into_slot(frame, self._stable_slots[stable_idx])
                self.decoder.ring.release(popped_slot)
                consecutive_errors = 0

                with self._decode_cond:
                    self._latest_slot_idx = stable_idx
                    self._latest_seq += 1
                    self._decode_cond.notify_all()
            except RuntimeError as exc:
                if decode_slot is not None:
                    try:
                        self.decoder.ring.release(decode_slot)
                    except Exception:
                        pass
                consecutive_errors += 1
                if consecutive_errors in (1, 5) or consecutive_errors % 25 == 0:
                    log_warning(
                        LogChannel.GLOBAL,
                        f"Background NVDEC decode skipped ({consecutive_errors}): {exc}",
                    )
            except Exception as exc:
                if decode_slot is not None:
                    try:
                        self.decoder.ring.release(decode_slot)
                    except Exception:
                        pass
                log_warning(LogChannel.GLOBAL, f"Background NVDEC loop error: {exc}")

    def _reserve_writable_slot(self) -> int | None:
        with self._decode_cond:
            consumer = self._consumer_slot_idx
            for offset in range(self._STABLE_SLOT_COUNT):
                idx = (self._write_cursor + offset) % self._STABLE_SLOT_COUNT
                if idx == consumer:
                    continue
                self._write_cursor = (idx + 1) % self._STABLE_SLOT_COUNT
                return idx
        return None

    def _copy_frame_into_slot(self, frame: GpuFrame, slot: _StableNv12Slot) -> None:
        if torch is None or self._copy_stream is None:
            raise RuntimeError("CUDA copy stream unavailable")

        width = int(getattr(frame, "width", 0))
        height = int(getattr(frame, "height", 0))
        pitch = int(getattr(frame, "pitch", 0) or width)
        half_h = max(1, height // 2)

        if (
            slot.y_plane is None
            or slot.uv_plane is None
            or slot.width != width
            or slot.height != height
        ):
            slot.y_plane = torch.empty((height, width), dtype=torch.uint8, device="cuda")
            slot.uv_plane = torch.empty((half_h, width), dtype=torch.uint8, device="cuda")
            slot.width = width
            slot.height = height
            slot.pitch = width

        y_ptr = _extract_device_pointer(getattr(frame, "device_ptr_y", None))
        uv_ptr = _extract_device_pointer(getattr(frame, "device_ptr_uv", None))
        if y_ptr <= 0:
            raise RuntimeError("Decoded NV12 frame missing device_ptr_y")
        if uv_ptr <= 0 or uv_ptr == y_ptr:
            uv_ptr = y_ptr + pitch * height

        copy_start_ns = time.perf_counter_ns()
        with torch.cuda.stream(self._copy_stream):
            _copy_plane_2d_async(
                destination_ptr=int(slot.y_plane.data_ptr()),
                destination_pitch_bytes=width,
                source_ptr=y_ptr,
                source_pitch_bytes=pitch,
                width_bytes=width,
                height_rows=height,
            )
            _copy_plane_2d_async(
                destination_ptr=int(slot.uv_plane.data_ptr()),
                destination_pitch_bytes=width,
                source_ptr=uv_ptr,
                source_pitch_bytes=pitch,
                width_bytes=width,
                height_rows=half_h,
            )
        copy_enqueued_ns = time.perf_counter_ns()
        self._copy_stream.synchronize()
        copy_done_ns = time.perf_counter_ns()

        telemetry = getattr(frame, "telemetry", None)
        if telemetry is not None:
            telemetry.add_metrics(
                {
                    "frame_source_copy_enqueue_ms": float((copy_enqueued_ns - copy_start_ns) / 1_000_000.0),
                    "frame_source_copy_sync_ms": float((copy_done_ns - copy_enqueued_ns) / 1_000_000.0),
                    "frame_source_copy_total_ms": float((copy_done_ns - copy_start_ns) / 1_000_000.0),
                }
            )

        slot.timestamp_ns = getattr(frame, "timestamp_ns", None)
        slot.telemetry = telemetry

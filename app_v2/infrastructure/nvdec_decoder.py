from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

from app_v2.infrastructure.gpu_ring_buffer import GpuFrame, GpuPixelFormat, GpuRingBuffer
from logger.filtered_logger import LogChannel, debug as log_debug, info as log_info, warning as log_warning


@dataclass(frozen=True, slots=True)
class NvdecDecodeConfig:
    """NVDEC decode settings for the GPU-only pipeline."""

    ring_capacity: int = 8
    gpu_id: int = 0
    surface_retry_limit: int = 5


class NvdecDecoder:
    """NVDEC-powered decoder that keeps decoded surfaces resident on the GPU."""

    def __init__(self, stream_url: str, config: NvdecDecodeConfig | None = None) -> None:
        self.stream_url = stream_url
        self._connected = False
        self._config = config or NvdecDecodeConfig()
        self._ring = GpuRingBuffer(self._config.ring_capacity)
        self._pynvcodec = self._load_pynvcodec()
        self._demuxer = self._create_demuxer()
        self._decoder = self._create_decoder()
        self._width, self._height = self._dimensions()
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
        """Decode the next frame into a ring-buffer slot and return the slot index."""
        if not self._connected:
            log_warning(LogChannel.GLOBAL, "NVDEC decoder requested before start")
            raise RuntimeError("Decoder not started")

        slot = self._ring.acquire(block=True)
        if slot is None:
            raise RuntimeError("NVDEC ring buffer acquire timed out")

        try:
            surface = self._decode_surface()
            if surface is None:
                raise RuntimeError("NVDEC decoder failed to produce a surface")
            frame = self._surface_to_gpu_frame(surface, frame_id, timestamp_ns)
            self._ring.commit(slot, frame)
            log_debug(LogChannel.NVDEC, f"Decoded frame {frame_id} into slot {slot}")
            return slot
        except Exception:
            self._ring.release(slot)
            raise

    def _load_pynvcodec(self) -> Any:
        try:
            return importlib.import_module("PyNvCodec")
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyNvCodec must be installed to use NvdecDecoder") from exc

    def _create_demuxer(self) -> Any:
        demuxer_attempts = (
            ("FFmpegDemuxer", (self.stream_url, 0)),
            ("FFmpegDemuxer", (self.stream_url,)),
            ("PyFFmpegDemuxer", (self.stream_url,)),
        )
        last_exc: Exception | None = None
        for name, args in demuxer_attempts:
            demuxer_cls = getattr(self._pynvcodec, name, None)
            if demuxer_cls is None:
                continue
            try:
                return demuxer_cls(*args)
            except Exception as exc:  # pragma: no cover - depends on PyNvCodec build
                last_exc = exc
        message = "PyNvCodec FFmpeg-style demuxer is unavailable"
        if last_exc is not None:
            raise RuntimeError(message) from last_exc
        raise RuntimeError(message)

    def _create_decoder(self) -> Any:
        try:
            return self._pynvcodec.PyNvDecoder(self._demuxer, self._config.gpu_id)
        except (TypeError, AttributeError):
            try:
                return self._pynvcodec.PyNvDecoder(self.stream_url, self._config.gpu_id)
            except (TypeError, AttributeError, RuntimeError):
                width, height = self._dimensions()
                pixel_format = getattr(self._pynvcodec.PixelFormat, "NV12", None)
                if pixel_format is None:
                    raise RuntimeError("Cannot resolve NV12 pixel format for PyNvDecoder")
                cuda_codec = self._call_demuxer_method("Codec")
                if cuda_codec is None:
                    raise RuntimeError("Cannot resolve CUDA codec for PyNvDecoder")
                members = getattr(self._pynvcodec.CudaVideoCodec, "__members__", {})
                codec_enum = None
                for member in members.values():
                    if getattr(member, "value", None) == cuda_codec:
                        codec_enum = member
                        break
                if codec_enum is None:
                    raise RuntimeError("CUDA codec value is not supported by PyNvDecoder")
                try:
                    return self._pynvcodec.PyNvDecoder(width, height, pixel_format, codec_enum, self._config.gpu_id)
                except TypeError as exc:
                    raise RuntimeError("PyNvDecoder width/height constructor is unavailable") from exc

    def _dimensions(self) -> tuple[int, int]:
        width = self._call_demuxer_method("Width") or 0
        height = self._call_demuxer_method("Height") or 0
        return width, height

    def _call_demuxer_method(self, method_name: str) -> int | None:
        method = getattr(self._demuxer, method_name, None)
        if callable(method):
            try:
                return int(method())
            except TypeError:
                try:
                    return int(method)
                except Exception:
                    return None
        return None

    def _decode_surface(self) -> Any | None:
        for _ in range(self._config.surface_retry_limit):
            surface = self._decoder.DecodeSingleSurface()
            if surface is None or self._surface_is_empty(surface):
                continue
            return surface
        log_warning(LogChannel.GLOBAL, "NVDEC decoder could not decode a valid surface")
        return None

    def _surface_is_empty(self, surface: Any) -> bool:
        method = getattr(surface, "Empty", None)
        if callable(method):
            try:
                return bool(method())
            except TypeError:
                return False
        return False

    def _surface_to_gpu_frame(self, surface: Any, frame_id: int, timestamp_ns: int | None) -> GpuFrame:
        width = self._call_surface_method(surface, "Width") or self._width
        height = self._call_surface_method(surface, "Height") or self._height
        device_ptr_y = self._call_surface_method(surface, "PlanePtr", 0)
        device_ptr_uv = self._call_surface_method(surface, "PlanePtr", 1)
        pitch = self._call_surface_method(surface, "Pitch", 0)
        timestamp = timestamp_ns or self._surface_timestamp(surface)
        return GpuFrame(
            width=width,
            height=height,
            pixel_format=GpuPixelFormat.NV12,
            device_ptr_y=device_ptr_y,
            device_ptr_uv=device_ptr_uv,
            pitch=pitch,
            timestamp_ns=timestamp,
            frame_id=frame_id,
        )

    def _call_surface_method(self, surface: Any, method_name: str, *args: Any) -> int | None:
        method = getattr(surface, method_name, None)
        if not callable(method):
            return None
        try:
            return method(*args)
        except TypeError:
            try:
                return method()
            except Exception:
                return None

    def _surface_timestamp(self, surface: Any) -> int:
        for attr in ("Pts", "PresentationTimeStamp", "TimeStamp", "PtsAbs"):
            method = getattr(surface, attr, None)
            if callable(method):
                try:
                    value = method()
                except TypeError:
                    continue
                if isinstance(value, int):
                    return value
        return int(time.time_ns())

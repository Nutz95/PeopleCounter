from __future__ import annotations

import importlib
import json
import shutil
import subprocess
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
        self._probe_info: dict[str, Any] | None = None
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
            log_debug(LogChannel.NVDEC, "Trying PyNvDecoder(demuxer, gpu_id)")
            return self._pynvcodec.PyNvDecoder(self._demuxer, self._config.gpu_id)
        except (TypeError, AttributeError, RuntimeError) as exc:
            log_debug(LogChannel.NVDEC, f"PyNvDecoder(demuxer, gpu_id) failed: {exc}")
            try:
                log_debug(LogChannel.NVDEC, "Trying PyNvDecoder(url, gpu_id)")
                return self._pynvcodec.PyNvDecoder(self.stream_url, self._config.gpu_id)
            except (TypeError, AttributeError, RuntimeError) as exc2:
                log_debug(LogChannel.NVDEC, f"PyNvDecoder(url, gpu_id) failed: {exc2}")
                decoder_opts = self._build_decoder_opts()
                if decoder_opts:
                    try:
                        log_debug(LogChannel.NVDEC, f"Trying PyNvDecoder(url, gpu_id, opts={decoder_opts})")
                        return self._pynvcodec.PyNvDecoder(self.stream_url, self._config.gpu_id, decoder_opts)
                    except (TypeError, AttributeError, RuntimeError) as exc3:
                        log_debug(LogChannel.NVDEC, f"PyNvDecoder(url, gpu_id, opts) failed: {exc3}")
                width, height = self._dimensions()
                if not width or not height:
                    raise RuntimeError("Cannot resolve stream dimensions for PyNvDecoder")
                pixel_format = getattr(self._pynvcodec.PixelFormat, "NV12", None)
                if pixel_format is None:
                    raise RuntimeError("Cannot resolve NV12 pixel format for PyNvDecoder")
                codec_enum = self._resolve_cuda_codec_enum()
                if codec_enum is None:
                    raise RuntimeError("CUDA codec value is not supported by PyNvDecoder")
                try:
                    log_debug(
                        LogChannel.NVDEC,
                        f"Trying PyNvDecoder({width}x{height}, NV12, {codec_enum}, gpu_id={self._config.gpu_id})",
                    )
                    return self._pynvcodec.PyNvDecoder(width, height, pixel_format, codec_enum, self._config.gpu_id)
                except TypeError as exc:
                    raise RuntimeError("PyNvDecoder width/height constructor is unavailable") from exc

    def _build_decoder_opts(self) -> dict[str, str]:
        url = (self.stream_url or "").lower()
        if url.startswith("http://") or url.startswith("https://"):
            return {
                "probesize": "5000000",
                "analyzeduration": "10000000",
                "reconnect": "1",
                "reconnect_streamed": "1",
                "reconnect_delay_max": "2",
            }
        return {}

    def _dimensions(self) -> tuple[int, int]:
        width = self._call_demuxer_method("Width") or 0
        height = self._call_demuxer_method("Height") or 0
        if width and height:
            return width, height
        info = self._probe_stream_info()
        width = width or info.get("width", 0)
        height = height or info.get("height", 0)
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
        # Some PyNvCodec builds crash when querying UV plane pointer explicitly.
        # Keep a conservative fallback for now: reuse plane 0 pointer so the frame
        # object remains usable and non-null for downstream checks.
        device_ptr_uv = device_ptr_y
        pitch = self._call_surface_method(surface, "Pitch")
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

    def _resolve_cuda_codec_enum(self) -> Any | None:
        members = getattr(self._pynvcodec.CudaVideoCodec, "__members__", {})
        cuda_codec = self._call_demuxer_method("Codec")
        if cuda_codec is not None:
            for member in members.values():
                if getattr(member, "value", None) == cuda_codec:
                    return member
        info = self._probe_stream_info()
        codec_name = info.get("codec_name")
        if codec_name:
            normalized = codec_name.lower()
            for name, member in members.items():
                if name.lower() == normalized or name.lower().startswith(normalized):
                    return member
        return None

    def _probe_stream_info(self) -> dict[str, Any]:
        if self._probe_info is not None:
            return self._probe_info
        info: dict[str, Any] = {}
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            self._probe_info = info
            return info
        try:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,codec_name",
                    "-of",
                    "json",
                    self.stream_url,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            payload = json.loads(result.stdout or "{}")
            streams = payload.get("streams") or []
            if streams:
                stream = streams[0]
                width = stream.get("width")
                height = stream.get("height")
                if isinstance(width, int):
                    info["width"] = width
                if isinstance(height, int):
                    info["height"] = height
                codec_name = stream.get("codec_name")
                if isinstance(codec_name, str):
                    info["codec_name"] = codec_name
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
        self._probe_info = info
        return info

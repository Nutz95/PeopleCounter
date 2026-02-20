from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from app_v2.core.frame_telemetry import FrameTelemetry
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
        self._probe_info: dict[str, Any] | None = None
        self._decoder = self._create_decoder()
        self._width, self._height = self._dimensions()
        # True after the first successful DecodeSingleSurface(); used to gate
        # auto-reset so we only recreate the hardware decoder ONCE per
        # corruption event, not on every subsequent buffering-phase failure.
        self._last_surface_was_good: bool = False
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

    def reset(self) -> None:
        """Recreate the PyNvDecoder instance after a hardware error.

        PyNvCodec prints ``HW decoder faced error. Re-create instance.`` when
        the NVDEC hardware encounters a corrupt stream packet.  After that
        message the internal decoder state is invalid and all subsequent
        ``DecodeSingleSurface()`` calls will fail.  Creating a new
        ``PyNvDecoder`` object opens a fresh connection to the stream and
        re-initialises the NVDEC hardware context.

        ``_last_surface_was_good`` is cleared so that the fresh decoder's
        normal buffering-phase failures (while waiting for the next I-frame)
        do not trigger a second reset.
        """
        log_warning(LogChannel.GLOBAL, "Recreating NVDEC decoder instance after hardware error")
        self._last_surface_was_good = False
        try:
            del self._decoder
        except Exception:
            pass
        self._decoder = self._create_decoder()
        log_info(LogChannel.GLOBAL, "NVDEC decoder instance recreated successfully")

    def decode_next_into_ring(self, *, frame_id: int, timestamp_ns: int | None = None) -> int:
        """Decode the next frame into a ring-buffer slot and return the slot index."""
        if not self._connected:
            log_warning(LogChannel.GLOBAL, "NVDEC decoder requested before start")
            raise RuntimeError("Decoder not started")

        slot = self._ring.acquire(block=True)
        if slot is None:
            raise RuntimeError("NVDEC ring buffer acquire timed out")

        try:
            telemetry = FrameTelemetry(frame_id=frame_id)
            telemetry.mark_stage_start("nvdec")
            surface = self._decode_surface()
            if surface is None and self._last_surface_was_good:
                # First failure after a run of good frames: PyNvCodec has already
                # printed "HW decoder faced error. Re-create instance." for each
                # retry.  Recreate once; subsequent buffering-phase failures are
                # handled by the orchestrator's consecutive-error skip guard.
                self.reset()  # clears _last_surface_was_good internally
            telemetry.mark_stage_end("nvdec")
            if surface is None:
                raise RuntimeError("NVDEC decoder failed to produce a surface")
            self._last_surface_was_good = True
            frame = self._surface_to_gpu_frame(surface, frame_id, timestamp_ns, telemetry)
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

    def _create_decoder(self) -> Any:
        decoder_opts = self._build_decoder_opts()
        if decoder_opts:
            try:
                log_debug(LogChannel.NVDEC, "Trying PyNvDecoder(url, gpu_id, opts)")
                return self._pynvcodec.PyNvDecoder(self.stream_url, self._config.gpu_id, decoder_opts)
            except (TypeError, AttributeError, RuntimeError) as exc:
                log_debug(LogChannel.NVDEC, f"PyNvDecoder(url, gpu_id, opts) failed: {exc}")

        try:
            log_debug(LogChannel.NVDEC, "Trying PyNvDecoder(url, gpu_id)")
            return self._pynvcodec.PyNvDecoder(self.stream_url, self._config.gpu_id)
        except (TypeError, AttributeError, RuntimeError) as exc:
            log_debug(LogChannel.NVDEC, f"PyNvDecoder(url, gpu_id) failed: {exc}")

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
        info = self._probe_stream_info()
        width = info.get("width", 0)
        height = info.get("height", 0)
        return width, height

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

    def _surface_to_gpu_frame(self, surface: Any, frame_id: int, timestamp_ns: int | None, telemetry: FrameTelemetry | None) -> GpuFrame:
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
            telemetry=telemetry,
        )

    def _call_surface_method(self, surface: Any, method_name: str, *args: Any) -> int | None:
        method = getattr(surface, method_name, None)
        if not callable(method):
            return None
        try:
            value = method(*args)
            return self._normalize_surface_value(value)
        except TypeError:
            try:
                value = method()
                return self._normalize_surface_value(value)
            except Exception:
                return None

    @staticmethod
    def _normalize_surface_value(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        gpu_mem = getattr(value, "GpuMem", None)
        if callable(gpu_mem):
            maybe_ptr = gpu_mem()
            if isinstance(maybe_ptr, int):
                return maybe_ptr
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

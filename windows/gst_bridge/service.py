from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import re
import socket
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

from .catalog import MediaCatalog
from .config import BridgeConfig
from .mediamtx_server import MediaMtxServer
from .models import MediaItem, RuntimeMetrics, ValidationResult
from .persistent import (
    PersistentFrameBridge,
    SourceDecoder,
    StreamEncoder,
    build_black_frame,
    find_ffmpeg,
)
from .profiling import ProfileRecorder
from .validation import build_runtime_env

H264_DISCOVERY_LINE = re.compile(r"video\s+#\d+:\s+H\.264", re.IGNORECASE)
H265_DISCOVERY_LINE = re.compile(r"video\s+#\d+:\s+H\.265", re.IGNORECASE)
WIDTH_LINE = re.compile(r"^\s*Width:\s*(?P<width>\d+)", re.IGNORECASE | re.MULTILINE)
HEIGHT_LINE = re.compile(r"^\s*Height:\s*(?P<height>\d+)", re.IGNORECASE | re.MULTILINE)
RTSP_WRITE_QUEUE_FULL_LINE = re.compile(r"\brtsp\b.*\bwrite queue is full\b", re.IGNORECASE)
KIND_LABELS = {
    "camera": "Camera",
    "image": "Image",
    "video": "Video",
}


class GstBridgeService:
    def __init__(self, config: BridgeConfig, validation: ValidationResult) -> None:
        self._config = config
        self._validation = validation
        self._ffmpeg_path = find_ffmpeg(config.base_dir)
        self._catalog = MediaCatalog(
            config.media_dir,
            config.image_dir,
            self._ffmpeg_path,
            config.width,
            config.height,
            config.fps,
        )
        self._metrics = RuntimeMetrics(gst_version=validation.gst_version)
        self._metrics_lock = threading.Lock()
        self._items: list[MediaItem] = []
        self._current_source_id: str | None = None
        self._current_source_item: MediaItem | None = None
        self._blocked_sources: dict[str, str] = {}
        self._execution_log: deque[str] = deque(maxlen=1200)
        self._execution_log_lock = threading.Lock()
        self._mediamtx = MediaMtxServer(
            validation.mediamtx_path,
            config.rtsp_port,
            config.rtmp_port,
            on_log=self._on_mediamtx_log,
        )
        self._black_frame = build_black_frame(config.width, config.height)
        self._encoder = StreamEncoder(
            self._ffmpeg_path,
            config,
            on_log=self._on_encoder_log,
            on_terminated=self._on_encoder_terminated,
        )
        self._bridge = PersistentFrameBridge(
            self._encoder,
            config.fps,
            self._black_frame,
            on_metrics=self._on_fps_metrics,
        )
        self._active_decoder: SourceDecoder | None = None
        self._switch_cancel = threading.Event()
        self._profile_recorder: ProfileRecorder | None = None
        self._profile_thread: threading.Thread | None = None
        self._profile_stop = threading.Event()
        self._shutdown_requested = False
        self._append_execution_log(f"Service initialized (ffmpeg={self._ffmpeg_path})")

    def start(self) -> None:
        self._shutdown_requested = False
        self._append_execution_log("Starting MediaMTX")
        self._mediamtx.start()
        self._append_execution_log("Starting persistent RTMP publisher")
        self._encoder.start()
        self._bridge.start()
        with self._metrics_lock:
            self._metrics.pipeline_state = "starting"
            self._metrics.last_log = "Persistent publisher started"
        self._append_execution_log(f"Publish target: {self._config.rtmp_publish_url}")
        self._append_execution_log("Refreshing media catalog")
        self.refresh_items()
        if self._config.profile:
            output_dir = self._build_profile_output_dir()
            self._append_execution_log(f"Profiling enabled -> {output_dir}")
            self._profile_recorder = ProfileRecorder(output_dir)
            self._profile_stop.clear()
            self._profile_thread = threading.Thread(target=self._profile_loop, daemon=True, name="gst-profile")
            self._profile_thread.start()
        startup_item = self._choose_startup_item()
        if startup_item is not None:
            self._append_execution_log(f"Auto-start source selected: {startup_item.name}")
            self.select_source(startup_item.source_id)
        else:
            with self._metrics_lock:
                self._metrics.pipeline_state = "playing"
                self._metrics.last_log = "No auto-start compatible source found; publishing black frames"
            self._append_execution_log("No auto-start compatible source found; publishing black frames")

    def stop(self) -> None:
        self._shutdown_requested = True
        self._switch_cancel.set()
        self._append_execution_log("Stopping bridge")
        self._profile_stop.set()
        self._bridge.stop()
        if self._active_decoder is not None:
            self._active_decoder.stop()
            self._active_decoder = None
        self._encoder.stop()
        self._mediamtx.stop()
        if self._profile_thread is not None:
            self._profile_thread.join(timeout=2.0)
        if self._profile_recorder is not None:
            csv_path, png_path = self._profile_recorder.finalize()
            with self._metrics_lock:
                self._metrics.last_profile_csv = str(csv_path)
                self._metrics.last_profile_png = str(png_path) if png_path else ""
            self._profile_recorder = None
        with self._metrics_lock:
            self._metrics.pipeline_state = "stopped"
        self._append_execution_log("Bridge stopped")

    def refresh_items(self) -> list[MediaItem]:
        self._items = self._annotate_and_sort_items(self._catalog.list_items())
        return list(self._items)

    def list_items(self) -> list[MediaItem]:
        return list(self._items)

    def select_source(self, source_id: str) -> None:
        item = next((candidate for candidate in self._items if candidate.source_id == source_id), None)
        if item is None:
            raise ValueError(f"Unknown source id: {source_id}")
        self._append_execution_log("=" * 72)
        self._append_execution_log(f"Play requested: {item.name}")
        blocked_reason = self._blocked_sources.get(source_id)
        if blocked_reason is not None:
            with self._metrics_lock:
                self._metrics.selected_source = item.name
                self._metrics.pipeline_state = "unsupported"
                self._metrics.last_log = blocked_reason
                self._metrics.errors += 1
            self._append_execution_log(f"Blocked source: {blocked_reason}")
            raise RuntimeError(blocked_reason)
        self._switch_to_source(item)

    def get_stream_url(self) -> str:
        host = self._local_ip()
        return f"rtsp://{host}:{self._config.rtsp_port}/{self._config.rtsp_path}"

    def get_status_text(self) -> str:
        with self._metrics_lock:
            source = self._metrics.selected_source
            state = self._metrics.pipeline_state
            detail = self._metrics.last_log
        if state == "unsupported" and detail:
            return f"Current source: {source} ({state}) - {detail[:120]}"
        return f"Current source: {source} ({state})"

    def get_current_source_id(self) -> str | None:
        return self._current_source_id

    def get_metrics_snapshot(self) -> RuntimeMetrics:
        with self._metrics_lock:
            return RuntimeMetrics(**self._metrics.__dict__)

    def get_ffmpeg_path(self) -> Path:
        return self._ffmpeg_path

    def get_current_source_metadata_lines(self) -> list[str]:
        item = self._current_source_item
        if item is None:
            return [
                "No source selected yet.",
                "",
                "Click a camera, image, or video thumbnail on the left to start playback.",
            ]
        return list(item.metadata_lines or self._fallback_metadata_lines(item))

    def get_execution_log_lines(self) -> list[str]:
        with self._execution_log_lock:
            return list(self._execution_log)

    def _switch_to_source(self, item: MediaItem) -> None:
        self._switch_cancel.set()
        self._switch_cancel = threading.Event()
        cancel = self._switch_cancel
        with self._metrics_lock:
            self._metrics.selected_source = item.name
            self._metrics.pipeline_state = "switching" if self._current_source_item is not None else "starting"
            self._metrics.decoder_fps = 0.0
            self._metrics.output_fps = 0.0
            self._metrics.dropped_fps = 0.0
            self._metrics.queue_fill = "preroll"
            self._metrics.warnings = 0
            self._metrics.rtsp_queue_warnings = 0
            self._metrics.errors = 0
            self._metrics.last_log = "Preparing decoder"
        threading.Thread(
            target=self._preroll_and_switch,
            args=(item, cancel),
            daemon=True,
            name=f"source-switch:{item.name}",
        ).start()

    def _preroll_and_switch(self, item: MediaItem, cancel: threading.Event) -> None:
        decoder = SourceDecoder(
            self._ffmpeg_path,
            item,
            self._config,
            self._black_frame,
            on_log=self._on_decoder_log,
        )
        try:
            decoder.start()
        except Exception as exc:
            self._append_execution_log(f"Decoder startup failed for {item.name}: {exc}")
            with self._metrics_lock:
                self._metrics.pipeline_state = "error"
                self._metrics.last_log = str(exc)
                self._metrics.errors += 1
            return
        preroll_target = 8
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cancel.is_set() or self._shutdown_requested:
                decoder.stop()
                return
            if decoder.queue_size >= preroll_target:
                break
            if not decoder.alive and decoder.queue_size == 0:
                message = f"Decoder exited before preroll completed for {item.name}"
                self._append_execution_log(message)
                decoder.stop()
                with self._metrics_lock:
                    self._metrics.pipeline_state = "error"
                    self._metrics.last_log = message
                    self._metrics.errors += 1
                return
            time.sleep(0.02)
        if cancel.is_set() or self._shutdown_requested:
            decoder.stop()
            return
        self._bridge.switch(decoder)
        self._active_decoder = decoder
        self._current_source_id = item.source_id
        self._current_source_item = item
        with self._metrics_lock:
            self._metrics.pipeline_state = "playing"
            self._metrics.queue_fill = "live"
            self._metrics.last_log = "Source ready"
        self._append_execution_log(f"Source ready: {item.name}")

    def _on_encoder_log(self, line: str) -> None:
        with self._metrics_lock:
            self._metrics.last_log = line
        self._append_execution_log(line)

    def _on_encoder_terminated(self, exit_code: int) -> None:
        self._append_execution_log(f"Persistent publisher exited with code {exit_code}")
        if self._shutdown_requested:
            return
        with self._metrics_lock:
            self._metrics.pipeline_state = "error"
            self._metrics.last_log = f"publisher exited with code {exit_code}"
            self._metrics.errors += 1

    def _on_decoder_log(self, line: str) -> None:
        self._append_execution_log(line)

    def _on_fps_metrics(self, current_fps: float, average_fps: float, dropped_fps: float) -> None:
        with self._metrics_lock:
            self._metrics.decoder_fps = current_fps
            self._metrics.output_fps = average_fps
            self._metrics.dropped_fps = dropped_fps
            self._metrics.queue_fill = "live"
        if current_fps > 0 or average_fps > 0 or dropped_fps > 0:
            self._append_execution_log(
                f"fps update: decode={current_fps:.2f} output={average_fps:.2f} dropped={dropped_fps:.2f}"
            )

    def _on_mediamtx_log(self, line: str) -> None:
        if RTSP_WRITE_QUEUE_FULL_LINE.search(line):
            with self._metrics_lock:
                self._metrics.warnings += 1
                self._metrics.rtsp_queue_warnings += 1
                self._metrics.last_log = "RTSP client write queue is full"
        self._append_execution_log(line)

    def _profile_loop(self) -> None:
        while not self._profile_stop.wait(0.5):
            if self._profile_recorder is None:
                continue
            self._profile_recorder.add_sample(self.get_metrics_snapshot())

    def _build_profile_output_dir(self) -> Path:
        base_dir = self._config.profile_output_dir
        if base_dir is None:
            raise RuntimeError("Profiling requested without output directory")
        return base_dir / time.strftime("gst-%Y%m%d-%H%M%S")

    def _annotate_and_sort_items(self, items: list[MediaItem]) -> list[MediaItem]:
        discoverer = self._validation.gst_launch_path.parent / "gst-discoverer-1.0.exe"
        env = build_runtime_env(self._validation.gstreamer_root)
        self._blocked_sources = {}
        ranked_items: list[tuple[int, MediaItem]] = []
        for item in items:
            codec, resolution, blocked_reason = self._probe_source(item, discoverer, env)
            probe_data = self._probe_ffmpeg_metadata(item)
            details = item.details
            if codec is not None:
                details = f"{details} | codec {codec}"
            if resolution is not None:
                details = f"{details} | {resolution[0]}x{resolution[1]}"
            if blocked_reason is not None:
                details = f"{details} | {blocked_reason}"
                self._blocked_sources[item.source_id] = blocked_reason
            metadata_lines = self._build_metadata_lines(item, codec, resolution, blocked_reason, probe_data)
            ranked_items.append(
                (
                    self._source_priority(item, codec, resolution, blocked_reason),
                    replace(item, details=details, metadata_lines=tuple(metadata_lines)),
                )
            )
        ranked_items.sort(key=lambda entry: (entry[0], entry[1].name.lower()))
        return [item for _, item in ranked_items]

    def _choose_startup_item(self) -> MediaItem | None:
        for item in self._items:
            if item.source_id not in self._blocked_sources:
                return item
        return None

    def _probe_source(
        self,
        item: MediaItem,
        discoverer: Path,
        env: dict[str, str],
    ) -> tuple[str | None, tuple[int, int] | None, str | None]:
        if item.kind != "video" or not discoverer.exists():
            return None, None, None
        result = subprocess.run(
            [str(discoverer), str(item.path)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        resolution = self._parse_resolution(output)
        if H264_DISCOVERY_LINE.search(output):
            return "H.264", resolution, None
        if H265_DISCOVERY_LINE.search(output):
            return "H.265", resolution, None
        return None, resolution, None

    def _probe_ffmpeg_metadata(self, item: MediaItem) -> dict[str, str]:
        if item.path is None:
            return {}
        ffprobe_path = self._ffmpeg_path.with_name("ffprobe.exe")
        if not ffprobe_path.exists():
            return {}
        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,profile,width,height,sample_aspect_ratio,display_aspect_ratio,pix_fmt,avg_frame_rate:format=duration,format_name",
                "-of",
                "json",
                str(item.path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
        stream = next(iter(payload.get("streams") or []), {})
        format_info = payload.get("format") or {}
        metadata: dict[str, str] = {}
        for key in (
            "codec_name",
            "profile",
            "width",
            "height",
            "sample_aspect_ratio",
            "display_aspect_ratio",
            "pix_fmt",
            "avg_frame_rate",
        ):
            value = stream.get(key)
            if value not in (None, "", "0/0"):
                metadata[key] = str(value)
        for key in ("duration", "format_name"):
            value = format_info.get(key)
            if value not in (None, ""):
                metadata[key] = str(value)
        return metadata

    def _build_metadata_lines(
        self,
        item: MediaItem,
        codec: str | None,
        resolution: tuple[int, int] | None,
        blocked_reason: str | None,
        probe_data: dict[str, str],
    ) -> list[str]:
        lines = [
            f"name       : {item.name}",
            f"type       : {KIND_LABELS.get(item.kind, item.kind.title())}",
        ]
        if item.path is not None:
            lines.append(f"path       : {item.path}")
        if item.source_spec:
            lines.append(f"input spec : {item.source_spec}")
        if codec is not None:
            lines.append(f"codec      : {codec}")
        elif "codec_name" in probe_data:
            lines.append(f"codec      : {probe_data['codec_name']}")
        if "profile" in probe_data:
            lines.append(f"profile    : {probe_data['profile']}")
        if resolution is not None:
            lines.append(f"resolution : {resolution[0]}x{resolution[1]}")
        elif "width" in probe_data and "height" in probe_data:
            lines.append(f"resolution : {probe_data['width']}x{probe_data['height']}")
        if "display_aspect_ratio" in probe_data:
            lines.append(f"display AR : {probe_data['display_aspect_ratio']}")
        if "sample_aspect_ratio" in probe_data:
            lines.append(f"sample AR  : {probe_data['sample_aspect_ratio']}")
        if "avg_frame_rate" in probe_data:
            lines.append(f"fps        : {self._format_frame_rate(probe_data['avg_frame_rate'])}")
        if "duration" in probe_data:
            lines.append(f"duration   : {self._format_duration(probe_data['duration'])}")
        if "pix_fmt" in probe_data:
            lines.append(f"pixel fmt  : {probe_data['pix_fmt']}")
        if "format_name" in probe_data:
            lines.append(f"container  : {probe_data['format_name']}")
        lines.append("")
        lines.append(f"summary    : {item.details}")
        if blocked_reason is not None:
            lines.append("")
            lines.append(f"warning    : {blocked_reason}")
        return lines

    @staticmethod
    def _fallback_metadata_lines(item: MediaItem) -> list[str]:
        lines = [
            f"name       : {item.name}",
            f"type       : {KIND_LABELS.get(item.kind, item.kind.title())}",
        ]
        if item.path is not None:
            lines.append(f"path       : {item.path}")
        if item.source_spec is not None:
            lines.append(f"input spec : {item.source_spec}")
        lines.append("")
        lines.append(f"summary    : {item.details}")
        return lines

    @staticmethod
    def _format_duration(duration_text: str) -> str:
        try:
            total_seconds = float(duration_text)
        except ValueError:
            return duration_text
        minutes, seconds = divmod(total_seconds, 60.0)
        hours, minutes = divmod(minutes, 60.0)
        if hours >= 1:
            return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        return f"{int(minutes):02d}:{seconds:05.2f}"

    @staticmethod
    def _format_frame_rate(frame_rate_text: str) -> str:
        if "/" not in frame_rate_text:
            return frame_rate_text
        numerator_text, denominator_text = frame_rate_text.split("/", 1)
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError:
            return frame_rate_text
        if denominator == 0:
            return frame_rate_text
        return f"{(numerator / denominator):.2f}"

    @staticmethod
    def _source_priority(
        item: MediaItem,
        codec: str | None,
        resolution: tuple[int, int] | None,
        blocked_reason: str | None,
    ) -> int:
        if blocked_reason is not None:
            return 100
        if item.kind == "video" and codec == "H.264":
            if resolution is None:
                return 20
            width, height = resolution
            pixels = width * height
            if pixels <= 640 * 640:
                return 0
            if pixels <= 1280 * 720:
                return 5
            if pixels <= 1920 * 1080:
                return 10
            return 30
        if item.kind == "image":
            return 10
        if item.kind == "camera":
            return 15
        if item.kind == "video":
            return 20
        return 50

    @staticmethod
    def _parse_resolution(output: str) -> tuple[int, int] | None:
        width_match = WIDTH_LINE.search(output)
        height_match = HEIGHT_LINE.search(output)
        if not width_match or not height_match:
            return None
        try:
            return int(width_match.group("width")), int(height_match.group("height"))
        except ValueError:
            return None

    @staticmethod
    def _timestamp_line(message: str) -> str:
        return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"

    def _append_execution_log(self, message: str) -> None:
        with self._execution_log_lock:
            self._execution_log.append(self._timestamp_line(message))

    @staticmethod
    def _local_ip() -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            sock.close()

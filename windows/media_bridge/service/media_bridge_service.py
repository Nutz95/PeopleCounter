from __future__ import annotations

import socket
from pathlib import Path

from ..catalog.media_catalog import MediaCatalog
from ..configuration.bridge_config import BridgeConfig
from ..ffmpeg_tools import ensure_ffmpeg, resolve_encoder
from ..models.media_item import MediaItem
from ..streaming.ffmpeg_rtsp_streamer import FfmpegRtspStreamer
from ..streaming.media_mtx_server import MediaMtxServer


class MediaBridgeService:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.catalog: MediaCatalog | None = None
        self.rtsp_server: MediaMtxServer | None = None
        self.streamer: FfmpegRtspStreamer | None = None
        self._items: list[MediaItem] = []
        self._items_by_id: dict[str, MediaItem] = {}
        self._current_source_id: str | None = None
        self._status_message: str | None = None
        self._ffmpeg_path: Path | None = None
        self._encoder_name: str | None = None

    def get_ip(self) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            sock.close()

    def get_stream_url(self) -> str:
        return f"rtsp://{self.get_ip()}:{self.config.port}/{self.config.rtsp_path}"

    def list_items(self) -> list[MediaItem]:
        return list(self._items)

    def refresh_items(self) -> list[MediaItem]:
        if self.catalog is None:
            return self.list_items()
        self._items = self.catalog.list_items()
        self._items_by_id = {item.source_id: item for item in self._items}
        return self.list_items()

    def select_source(self, source: str | Path) -> None:
        if isinstance(source, Path):
            resolved = source.resolve()
            item = next((entry for entry in self._items if entry.path == resolved), None)
            if item is None:
                raise FileNotFoundError(f"Media file not found in catalog: {source}")
        else:
            item = self._items_by_id.get(source)
            if item is None:
                raise KeyError(f"Unknown source id: {source}")
        if self.streamer is None:
            raise RuntimeError("Media bridge is not started")
        if item.kind == "camera":
            self._select_camera_source(item)
            return
        self._select_static_source(item)

    def get_current_source_label(self) -> str:
        if self._current_source_id is None:
            return "none"
        item = self._items_by_id.get(self._current_source_id)
        return item.name if item is not None else self._current_source_id

    def get_current_source_id(self) -> str | None:
        return self._current_source_id

    def get_status_text(self) -> str:
        if self._status_message:
            return self._status_message
        return f"Current source: {self.get_current_source_label()}"

    def start(self) -> None:
        ffmpeg_path = ensure_ffmpeg()
        if ffmpeg_path is None:
            raise RuntimeError("FFmpeg is not available")
        encoder_name = resolve_encoder(ffmpeg_path, self.config.encoder)
        self._ffmpeg_path = ffmpeg_path
        self._encoder_name = encoder_name
        self.catalog = MediaCatalog(
            media_dir=self.config.media_dir,
            image_dir=self.config.image_dir,
            ffmpeg_path=ffmpeg_path,
            output_width=self.config.width,
            output_height=self.config.height,
            target_fps=self.config.fps,
            reference_image=self.config.reference_image,
            initial_source=self.config.initial_source,
            include_cameras=False,
        )
        self._items = self.catalog.list_items()
        if not self._items:
            raise RuntimeError("No media or camera source available")
        self._items_by_id = {item.source_id: item for item in self._items}
        initial_id = self.catalog.resolve_initial_item_id(self._items)
        if initial_id is None:
            raise RuntimeError("Unable to resolve the initial source")
        self.rtsp_server = MediaMtxServer(
            bin_dir=ffmpeg_path.parent.parent.parent,
            port=self.config.port,
        )
        self.rtsp_server.start()
        self._start_streamer_for_item(self._items_by_id[initial_id])
        self._current_source_id = initial_id

    def stop(self) -> None:
        self._stop_streamer()
        if self.rtsp_server is not None:
            self.rtsp_server.stop()
            self.rtsp_server = None

    def _set_status(self, message: str | None) -> None:
        self._status_message = message

    def _select_static_source(self, item: MediaItem) -> None:
        if self._current_source_id == item.source_id:
            return
        self._set_status(f"Switching to {item.name}...")
        try:
            self._start_streamer_for_item(item)
            self._current_source_id = item.source_id
        finally:
            self._set_status(None)

    def _select_camera_source(self, item: MediaItem) -> None:
        raise RuntimeError("Camera sources are disabled in the static media demo")

    def _create_streamer(
        self,
        items: list[MediaItem],
        encoder_name: str | None = None,
    ) -> FfmpegRtspStreamer:
        ffmpeg_path = self._ffmpeg_path or ensure_ffmpeg()
        if ffmpeg_path is None:
            raise RuntimeError("FFmpeg is not available")
        resolved_encoder = encoder_name or self._encoder_name or resolve_encoder(ffmpeg_path, self.config.encoder)
        zmq_port = self._allocate_zmq_port()
        return FfmpegRtspStreamer(
            ffmpeg_path=ffmpeg_path,
            items=items,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            bitrate_kbps=self.config.bitrate_kbps,
            encoder_name=resolved_encoder,
            rtsp_port=self.config.port,
            rtsp_path=self.config.rtsp_path,
            zmq_port=zmq_port,
        )

    def _start_streamer_for_item(self, item: MediaItem) -> None:
        previous_streamer = self.streamer
        self.streamer = None
        if previous_streamer is not None:
            previous_streamer.stop()
        next_streamer = self._create_streamer([item], encoder_name=self._encoder_name)
        next_streamer.start(item.source_id)
        try:
            next_streamer.wait_until_ready(item.source_id)
        except Exception:
            next_streamer.stop()
            raise
        self.streamer = next_streamer

    def _stop_streamer(self) -> None:
        if self.streamer is None:
            return
        self.streamer.stop()
        self.streamer = None

    def _allocate_zmq_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
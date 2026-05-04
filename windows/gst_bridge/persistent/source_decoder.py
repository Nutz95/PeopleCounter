from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from ..config import BridgeConfig
from ..models import MediaItem
from .utils import build_scale_filter


class SourceDecoder:
    def __init__(
        self,
        ffmpeg_path: Path,
        item: MediaItem,
        config: BridgeConfig,
        black_frame: bytes,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self._ffmpeg_path = ffmpeg_path
        self._item = item
        self._config = config
        self._frame_bytes = config.width * config.height * 3 // 2
        self._black_frame = black_frame
        self._on_log = on_log
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=16)
        self._last_frame: bytes = black_frame
        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        cmd = [
            str(self._ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-threads",
            "4",
            *self._build_input_args(),
            "-vf",
            build_scale_filter(self._config),
            "-an",
            "-pix_fmt",
            "nv12",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=self._frame_bytes,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True, name=f"ffmpeg-dec:{self._item.name}")
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True, name=f"ffmpeg-dec-log:{self._item.name}")
        self._stderr_thread.start()

    def _build_input_args(self) -> list[str]:
        if self._item.kind == "image":
            if self._item.path is None:
                raise RuntimeError(f"Image source has no path: {self._item.name}")
            return ["-re", "-loop", "1", "-framerate", str(self._config.fps), "-i", str(self._item.path)]
        if self._item.kind == "camera":
            if not self._item.source_spec:
                raise RuntimeError(f"Camera source has no input spec: {self._item.name}")
            return [
                "-f",
                "dshow",
                "-rtbufsize",
                "256M",
                "-framerate",
                str(self._config.fps),
                "-video_size",
                f"{self._config.width}x{self._config.height}",
                "-i",
                self._item.source_spec,
            ]
        if self._item.path is None:
            raise RuntimeError(f"Video source has no path: {self._item.name}")
        return ["-re", "-stream_loop", "-1", "-i", str(self._item.path)]

    def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        while not self._stop.is_set():
            try:
                data = self._process.stdout.read(self._frame_bytes)
            except Exception as exc:
                self._emit_log(f"decoder read failed for {self._item.name}: {exc}")
                break
            if len(data) < self._frame_bytes:
                break
            self._last_frame = data
            try:
                self._queue.put_nowait(data)
            except queue.Full:
                pass
        self._emit_log(f"decoder stream ended for {self._item.name}")

    def _stderr_loop(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        for raw_line in self._process.stderr:
            if self._stop.is_set():
                break
            line = raw_line.decode("utf-8", "replace").strip()
            if line:
                self._emit_log(f"[decoder:{self._item.name}] {line}")

    def _emit_log(self, line: str) -> None:
        if self._on_log is not None:
            self._on_log(line)

    def get(self) -> tuple[bytes, bool]:
        try:
            return self._queue.get_nowait(), False
        except queue.Empty:
            return self._last_frame or self._black_frame, True

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None:
            try:
                if self._process.stdout is not None:
                    self._process.stdout.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2.0)
            except Exception:
                self._process.kill()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)

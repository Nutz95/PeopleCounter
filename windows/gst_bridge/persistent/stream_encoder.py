from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from ..config import BridgeConfig


class StreamEncoder:
    def __init__(
        self,
        ffmpeg_path: Path,
        config: BridgeConfig,
        on_log: Callable[[str], None] | None = None,
        on_terminated: Callable[[int], None] | None = None,
    ) -> None:
        self._ffmpeg_path = ffmpeg_path
        self._config = config
        self._on_log = on_log
        self._on_terminated = on_terminated
        self._process: subprocess.Popen[bytes] | None = None
        self._pending_lock = threading.Lock()
        self._pending_frame: bytes | None = None
        self._pending_event = threading.Event()
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._wait_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.alive:
            return
        maxrate = self._config.bitrate_kbps
        bufsize = self._config.bitrate_kbps
        gop_size = max(self._config.fps * 2, self._config.fps)
        cmd = [
            str(self._ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-init_hw_device",
            "qsv=hw",
            "-f",
            "rawvideo",
            "-pixel_format",
            "nv12",
            "-video_size",
            f"{self._config.width}x{self._config.height}",
            "-framerate",
            str(self._config.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "h264_qsv",
            "-preset",
            "veryfast",
            "-scenario",
            "livestreaming",
            "-low_delay_brc",
            "1",
            "-bitrate_limit",
            "1",
            "-skip_frame",
            "insert_dummy",
            "-bf",
            "0",
            "-async_depth",
            "1",
            "-look_ahead",
            "0",
            "-g",
            str(gop_size),
            "-keyint_min",
            str(gop_size),
            "-rc_init_occupancy",
            f"{bufsize}k",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-b:v",
            f"{self._config.bitrate_kbps}k",
            "-maxrate",
            f"{maxrate}k",
            "-bufsize",
            f"{bufsize}k",
            "-f",
            "flv",
            self._config.rtmp_publish_url,
        ]
        self._stop_event.clear()
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="ffmpeg-enc-writer")
        self._writer_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True, name="ffmpeg-enc-log")
        self._stderr_thread.start()
        self._wait_thread = threading.Thread(target=self._wait_loop, daemon=True, name="ffmpeg-enc-wait")
        self._wait_thread.start()

    def submit(self, frame: bytes) -> bool:
        if not self.alive:
            return False
        with self._pending_lock:
            self._pending_frame = frame
            self._pending_event.set()
        return True

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._pending_event.wait(0.1):
                continue
            with self._pending_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._pending_event.clear()
            if frame is None or self._process is None or self._process.stdin is None:
                continue
            try:
                self._process.stdin.write(frame)
                self._process.stdin.flush()
            except OSError as exc:
                self._emit_log(f"encoder write failed: {exc}")
                return

    def _stderr_loop(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        for raw_line in self._process.stderr:
            if self._stop_event.is_set():
                break
            line = raw_line.decode("utf-8", "replace").strip()
            if line:
                self._emit_log(f"[encoder] {line}")

    def _wait_loop(self) -> None:
        if self._process is None:
            return
        exit_code = self._process.wait()
        if self._on_terminated is not None:
            self._on_terminated(exit_code)

    def _emit_log(self, line: str) -> None:
        if self._on_log is not None:
            self._on_log(line)

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self) -> None:
        self._stop_event.set()
        self._pending_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)
        if self._process is not None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2.0)
            except Exception:
                self._process.kill()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
        if self._wait_thread is not None:
            self._wait_thread.join(timeout=2.0)

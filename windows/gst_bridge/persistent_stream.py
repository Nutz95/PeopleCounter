from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .config import BridgeConfig
from .models import MediaItem


def find_ffmpeg(base_dir: Path) -> Path:
    search_root = base_dir / "bin"
    if search_root.exists():
        found = next(search_root.rglob("ffmpeg.exe"), None)
        if found is not None:
            return found
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return Path(ffmpeg_path)
    raise RuntimeError(
        f"ffmpeg.exe not found under {search_root} and not available in PATH. "
        "The persistent Windows bridge requires a local FFmpeg binary."
    )


def build_black_frame(width: int, height: int) -> bytes:
    luma_size = width * height
    chroma_size = luma_size // 2
    return bytes([16]) * luma_size + bytes([128]) * chroma_size


def _build_scale_filter(config: BridgeConfig) -> str:
    return (
        f"scale={config.width}:{config.height}"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2:out_range=tv,"
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={config.fps},setsar=1,format=nv12"
    )


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
        if self._item.kind == "image":
            input_args = ["-re", "-loop", "1", "-framerate", str(self._config.fps), "-i", str(self._item.path)]
        else:
            input_args = ["-re", "-stream_loop", "-1", "-i", str(self._item.path)]
        cmd = [
            str(self._ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-threads",
            "4",
            *input_args,
            "-vf",
            _build_scale_filter(self._config),
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
        maxrate = int(self._config.bitrate_kbps * 1.5)
        bufsize = int(self._config.bitrate_kbps * 2)
        gop_size = self._config.fps
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


class FramePacer:
    def __init__(self, fps: int) -> None:
        self._interval_s = 1.0 / fps
        self._next_deadline = time.perf_counter()

    def reset(self) -> None:
        self._next_deadline = time.perf_counter()

    def wait_next(self, stop_event: threading.Event | None = None) -> None:
        self._next_deadline += self._interval_s
        remaining = self._next_deadline - time.perf_counter()
        if remaining <= 0:
            return
        if stop_event is not None:
            stop_event.wait(remaining)
        else:
            time.sleep(remaining)


class PersistentFrameBridge:
    def __init__(
        self,
        encoder: StreamEncoder,
        fps: int,
        black_frame: bytes,
        on_metrics: Callable[[float, float, float], None] | None = None,
    ) -> None:
        self._encoder = encoder
        self._fps = fps
        self._black_frame = black_frame
        self._on_metrics = on_metrics
        self._decoder: SourceDecoder | None = None
        self._decoder_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_since_report = 0
        self._freeze_since_report = 0
        self._report_started_at = time.monotonic()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="persistent-frame-bridge")
        self._thread.start()

    def switch(self, new_decoder: SourceDecoder) -> None:
        with self._decoder_lock:
            old_decoder = self._decoder
            self._decoder = new_decoder
        if old_decoder is not None:
            threading.Thread(target=old_decoder.stop, daemon=True, name="decoder-stop").start()

    def _loop(self) -> None:
        pacer = FramePacer(self._fps)
        while not self._stop.is_set():
            with self._decoder_lock:
                decoder = self._decoder
            if decoder is None:
                frame = self._black_frame
                freeze = False
            else:
                frame, freeze = decoder.get()
            if not self._encoder.submit(frame):
                self._stop.wait(0.1)
                pacer.reset()
                continue
            self._frames_since_report += 1
            if freeze:
                self._freeze_since_report += 1
            now = time.monotonic()
            elapsed = now - self._report_started_at
            if self._on_metrics is not None and elapsed >= 1.0:
                fps = self._frames_since_report / elapsed
                freeze_fps = self._freeze_since_report / elapsed
                self._on_metrics(fps, fps, freeze_fps)
                self._frames_since_report = 0
                self._freeze_since_report = 0
                self._report_started_at = now
            pacer.wait_next(self._stop)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._decoder_lock:
            decoder = self._decoder
            self._decoder = None
        if decoder is not None:
            decoder.stop()

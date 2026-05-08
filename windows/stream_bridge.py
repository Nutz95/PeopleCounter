#!/usr/bin/env python3
"""stream_bridge.py — Hardware-accelerated streaming bridge

Architecture:
  [SourceDecoder] → rawvideo NV12 queue → [FrameBridge thread]
                                               ↓  30 fps
  [StreamEncoder] ← stdin pipe ←────────────────
       ↓ h264_qsv (Intel Arc)
  [MediaMTX RTSP relay]  →  VLC / NVDEC

Key properties:
  • Encoder NEVER restarts on source switch → no VLC disconnection
  • Source switch: preroll new decoder 400 ms, then atomic swap
  • 4K 30 fps NV12 output, no audio
  • Pillow optional (thumbnails disabled if not installed)
"""
from __future__ import annotations

import argparse
import atexit
import csv
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
import zipfile
import ctypes as _ctypes
import ctypes.wintypes as _wintypes
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR    = SCRIPT_DIR / "bin"

OUTPUT_W   = 3840
OUTPUT_H   = 2160
OUTPUT_FPS = 30
# GOP size used by the RTSP encoder.
# Default to all-IDR (1) to minimize decoder resync jitter/latency, especially
# for static-image sources and frequent client reconnects. Override at runtime
# with STREAM_BRIDGE_GOP (e.g. 30) for bandwidth/quality experiments.
ENCODER_GOP = max(1, int(os.environ.get("STREAM_BRIDGE_GOP", "1")))
# NV12 frame size: Y plane (W×H) + interleaved UV plane (W×H/2)
FRAME_BYTES = OUTPUT_W * OUTPUT_H * 3 // 2  # 12 441 600 bytes

BITRATE_K  = 50_000          # kbps target
RTSP_PORT  = 5002
RTSP_PATH  = "live"
RTSP_URL   = f"rtsp://127.0.0.1:{RTSP_PORT}/{RTSP_PATH}"

# Directories to scan for media (first match wins per extension)
MEDIA_SCAN_DIRS: list[Path] = [
    Path(r"E:\SequencesVideo"),
    SCRIPT_DIR / "ref_videos",
    SCRIPT_DIR / "media",
]
IMAGE_SCAN_DIRS: list[Path] = [
    SCRIPT_DIR / "ref_images",
    SCRIPT_DIR / "images",
]
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".m2ts", ".m4v", ".wmv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MEDIAMTX_DL = (
    "https://github.com/bluenviron/mediamtx/releases/download/"
    "v1.11.3/mediamtx_v1.11.3_windows_amd64.zip"
)
FFMPEG_DL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

# NV12 TV-range black: Y=16 (video black), UV=128 (neutral chroma)
BLACK_FRAME: bytes = bytes([16]) * (OUTPUT_W * OUTPUT_H) + bytes([128]) * (OUTPUT_W * OUTPUT_H // 2)

# ─────────────────────────────────────────────────────────────
# Optional Pillow (thumbnails)
# ─────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageTk  # type: ignore
    _PILLOW = True
except ImportError:
    _PILLOW = False

# Windows: request a 1 ms scheduler quantum for legacy waits and as a
# fallback when the high-resolution waitable timer API is unavailable.
_TIMER_RESOLUTION_MS = 1
try:
    _ctypes.windll.winmm.timeBeginPeriod(_TIMER_RESOLUTION_MS)
    atexit.register(lambda: _ctypes.windll.winmm.timeEndPeriod(_TIMER_RESOLUTION_MS))
except Exception:
    pass

_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_TIMER_ALL_ACCESS = 0x001F0003
_WAIT_OBJECT_0 = 0x00000000
_WAIT_FAILED = 0xFFFFFFFF
_INFINITE = 0xFFFFFFFF

try:
    _kernel32 = _ctypes.windll.kernel32
    _CreateWaitableTimerExW = _kernel32.CreateWaitableTimerExW
    _CreateWaitableTimerExW.argtypes = [_ctypes.c_void_p, _wintypes.LPCWSTR, _wintypes.DWORD, _wintypes.DWORD]
    _CreateWaitableTimerExW.restype = _wintypes.HANDLE
    _SetWaitableTimer = _kernel32.SetWaitableTimer
    _SetWaitableTimer.argtypes = [
        _wintypes.HANDLE,
        _ctypes.POINTER(_ctypes.c_longlong),
        _wintypes.LONG,
        _ctypes.c_void_p,
        _ctypes.c_void_p,
        _wintypes.BOOL,
    ]
    _SetWaitableTimer.restype = _wintypes.BOOL
    _WaitForSingleObject = _kernel32.WaitForSingleObject
    _WaitForSingleObject.argtypes = [_wintypes.HANDLE, _wintypes.DWORD]
    _WaitForSingleObject.restype = _wintypes.DWORD
    _CancelWaitableTimer = _kernel32.CancelWaitableTimer
    _CancelWaitableTimer.argtypes = [_wintypes.HANDLE]
    _CancelWaitableTimer.restype = _wintypes.BOOL
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [_wintypes.HANDLE]
    _CloseHandle.restype = _wintypes.BOOL
except Exception:
    _CreateWaitableTimerExW = None
    _SetWaitableTimer = None
    _WaitForSingleObject = None
    _CancelWaitableTimer = None
    _CloseHandle = None


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def _find_or_download(exe_name: str, glob: str, url: str, dest: Path) -> Path:
    """Return path to exe, downloading + extracting if necessary."""
    # Search known locations first
    for search_root in (dest, BIN_DIR, SCRIPT_DIR):
        found = next(search_root.rglob(glob), None) if search_root.exists() else None
        if found:
            return found
    # Try system PATH
    which = shutil.which(exe_name)
    if which:
        return Path(which)
    # Download
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{exe_name}.zip"
    print(f"[i] Downloading {exe_name} …")
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink(missing_ok=True)
    found = next(dest.rglob(glob), None)
    if not found:
        raise FileNotFoundError(f"{glob} not found after download into {dest}")
    return found


def find_ffmpeg() -> Path:
    return _find_or_download("ffmpeg", "ffmpeg.exe", FFMPEG_DL, BIN_DIR)


def find_mediamtx() -> Path:
    return _find_or_download("mediamtx", "mediamtx.exe", MEDIAMTX_DL, BIN_DIR / "mediamtx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hardware-accelerated RTSP stream bridge with optional profiling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", action="store_true", help="Record live metrics to CSV and render a PNG plot on exit.")
    parser.add_argument("--profile-dir", help="Directory where profiling artifacts are written.")
    parser.add_argument("--profile-seconds", type=float, help="Automatically stop the app after N seconds (useful for automated profiling runs).")
    return parser.parse_args()


def resolve_encoder(ffmpeg: Path) -> str:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=False,
    )
    text = result.stdout + result.stderr
    for enc in ("h264_qsv", "h264_nvenc", "libx264"):
        if enc in text:
            return enc
    return "libx264"


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


@dataclass(frozen=True)
class MetricsSnapshot:
    decoder_frames: int
    decoder_drops: int
    decoder_read_errors: int
    bridge_frames_sent: int
    bridge_freeze_frames: int
    bridge_black_frames: int
    encoder_frames_written: int
    encoder_overwrites: int
    encoder_write_failures: int
    encoder_write_calls: int
    encoder_write_total_ms: float
    encoder_write_max_ms: float
    pacer_waits: int
    pacer_late_total_ms: float
    pacer_late_max_ms: float
    queue_depth: int
    queue_depth_max: int


class StreamMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._decoder_frames = 0
        self._decoder_drops = 0
        self._decoder_read_errors = 0
        self._bridge_frames_sent = 0
        self._bridge_freeze_frames = 0
        self._bridge_black_frames = 0
        self._encoder_frames_written = 0
        self._encoder_overwrites = 0
        self._encoder_write_failures = 0
        self._encoder_write_calls = 0
        self._encoder_write_total_ms = 0.0
        self._encoder_write_max_ms = 0.0
        self._pacer_waits = 0
        self._pacer_late_total_ms = 0.0
        self._pacer_late_max_ms = 0.0
        self._queue_depth = 0
        self._queue_depth_max = 0

    def record_decoder_frame(self) -> None:
        with self._lock:
            self._decoder_frames += 1

    def record_decoder_drop(self) -> None:
        with self._lock:
            self._decoder_drops += 1

    def record_decoder_read_error(self) -> None:
        with self._lock:
            self._decoder_read_errors += 1

    def record_bridge_frame(self, freeze: bool, black: bool) -> None:
        with self._lock:
            self._bridge_frames_sent += 1
            if freeze:
                self._bridge_freeze_frames += 1
            if black:
                self._bridge_black_frames += 1

    def record_encoder_write(self, elapsed_ms: float, ok: bool) -> None:
        with self._lock:
            self._encoder_write_calls += 1
            self._encoder_write_total_ms += elapsed_ms
            if elapsed_ms > self._encoder_write_max_ms:
                self._encoder_write_max_ms = elapsed_ms
            if ok:
                self._encoder_frames_written += 1
            else:
                self._encoder_write_failures += 1

    def record_encoder_overwrite(self) -> None:
        with self._lock:
            self._encoder_overwrites += 1

    def record_pacer_wait(self, late_ms: float) -> None:
        with self._lock:
            self._pacer_waits += 1
            self._pacer_late_total_ms += late_ms
            if late_ms > self._pacer_late_max_ms:
                self._pacer_late_max_ms = late_ms

    def record_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depth = depth
            if depth > self._queue_depth_max:
                self._queue_depth_max = depth

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                decoder_frames=self._decoder_frames,
                decoder_drops=self._decoder_drops,
                decoder_read_errors=self._decoder_read_errors,
                bridge_frames_sent=self._bridge_frames_sent,
                bridge_freeze_frames=self._bridge_freeze_frames,
                bridge_black_frames=self._bridge_black_frames,
                encoder_frames_written=self._encoder_frames_written,
                encoder_overwrites=self._encoder_overwrites,
                encoder_write_failures=self._encoder_write_failures,
                encoder_write_calls=self._encoder_write_calls,
                encoder_write_total_ms=self._encoder_write_total_ms,
                encoder_write_max_ms=self._encoder_write_max_ms,
                pacer_waits=self._pacer_waits,
                pacer_late_total_ms=self._pacer_late_total_ms,
                pacer_late_max_ms=self._pacer_late_max_ms,
                queue_depth=self._queue_depth,
                queue_depth_max=self._queue_depth_max,
            )


class ProfileRecorder:
    def __init__(self, metrics: StreamMetrics, output_dir: Path, sample_interval_s: float = 0.5) -> None:
        self._metrics = metrics
        self._output_dir = output_dir
        self._sample_interval_s = sample_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict[str, float | int]] = []
        self._start_ts = 0.0
        self._prev_snapshot: MetricsSnapshot | None = None
        self._prev_ts = 0.0
        self.csv_path = self._output_dir / "metrics.csv"
        self.png_path = self._output_dir / "metrics.png"

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._start_ts = time.perf_counter()
        self._prev_ts = self._start_ts
        self._prev_snapshot = self._metrics.snapshot()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="profile-recorder")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._sample_interval_s):
            self._capture_sample()

    def _capture_sample(self) -> None:
        now = time.perf_counter()
        snap = self._metrics.snapshot()
        if self._prev_snapshot is None:
            self._prev_snapshot = snap
            self._prev_ts = now
            return

        elapsed = max(0.001, now - self._prev_ts)
        prev = self._prev_snapshot
        write_calls_delta = snap.encoder_write_calls - prev.encoder_write_calls
        write_total_delta = snap.encoder_write_total_ms - prev.encoder_write_total_ms
        pacer_waits_delta = snap.pacer_waits - prev.pacer_waits
        pacer_total_delta = snap.pacer_late_total_ms - prev.pacer_late_total_ms

        self._rows.append({
            "t_s": now - self._start_ts,
            "decode_fps": (snap.decoder_frames - prev.decoder_frames) / elapsed,
            "drops_per_s": (snap.decoder_drops - prev.decoder_drops) / elapsed,
            "output_fps": (snap.encoder_frames_written - prev.encoder_frames_written) / elapsed,
            "freeze_per_s": (snap.bridge_freeze_frames - prev.bridge_freeze_frames) / elapsed,
            "queue_depth": snap.queue_depth,
            "queue_depth_max": snap.queue_depth_max,
            "enc_overwrites_per_s": (snap.encoder_overwrites - prev.encoder_overwrites) / elapsed,
            "enc_write_avg_ms": (write_total_delta / write_calls_delta) if write_calls_delta > 0 else 0.0,
            "enc_write_max_ms_run": snap.encoder_write_max_ms,
            "enc_failures": snap.encoder_write_failures,
            "pacer_late_avg_ms": (pacer_total_delta / pacer_waits_delta) if pacer_waits_delta > 0 else 0.0,
            "pacer_late_max_ms_run": snap.pacer_late_max_ms,
            "black_frames": snap.bridge_black_frames,
            "read_errors": snap.decoder_read_errors,
        })
        self._prev_snapshot = snap
        self._prev_ts = now

    def stop(self) -> tuple[Path, Path | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._capture_sample()
        self._write_csv()
        return self.csv_path, self._render_plot()

    def _write_csv(self) -> None:
        if not self._rows:
            return
        fieldnames = list(self._rows[0].keys())
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

    def _render_plot(self) -> Path | None:
        if not self._rows:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None

        t = [float(row["t_s"]) for row in self._rows]
        decode = [float(row["decode_fps"]) for row in self._rows]
        output = [float(row["output_fps"]) for row in self._rows]
        drops = [float(row["drops_per_s"]) for row in self._rows]
        freeze = [float(row["freeze_per_s"]) for row in self._rows]
        queue = [float(row["queue_depth"]) for row in self._rows]
        overwrites = [float(row["enc_overwrites_per_s"]) for row in self._rows]
        enc_avg = [float(row["enc_write_avg_ms"]) for row in self._rows]
        pacer_avg = [float(row["pacer_late_avg_ms"]) for row in self._rows]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle("Stream Bridge Profiling")

        axes[0].plot(t, decode, label="decode fps", color="#4daf4a")
        axes[0].plot(t, output, label="output fps", color="#377eb8")
        axes[0].axhline(OUTPUT_FPS, color="#999999", linestyle="--", linewidth=1, label="target fps")
        axes[0].set_ylabel("fps")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(t, queue, label="queue depth", color="#984ea3")
        axes[1].plot(t, drops, label="drops/s", color="#e41a1c")
        axes[1].plot(t, freeze, label="freeze/s", color="#ff7f00")
        axes[1].plot(t, overwrites, label="enc overwrites/s", color="#377eb8")
        axes[1].set_ylabel("buffer")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(t, enc_avg, label="enc write avg ms", color="#a65628")
        axes[2].axhline(1000.0 / OUTPUT_FPS, color="#999999", linestyle="--", linewidth=1, label="33.3 ms budget")
        axes[2].set_ylabel("ms")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.25)

        axes[3].plot(t, pacer_avg, label="pacer late avg ms", color="#f781bf")
        axes[3].set_ylabel("ms")
        axes[3].set_xlabel("time (s)")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(self.png_path, dpi=140)
        plt.close(fig)
        return self.png_path


class FramePacer:
    """High-precision frame scheduler for Windows.

    Strategy:
      1. keep an absolute deadline sequence from perf_counter()
      2. use a high-resolution waitable timer for the coarse sleep
      3. busy-spin only for the last sub-millisecond slice

    This avoids the burst / pause behaviour of plain time.sleep() while still
    keeping CPU usage modest.
    """

    def __init__(self, fps: int, metrics: StreamMetrics | None = None, spin_window_s: float = 0.00075) -> None:
        self._interval_s = 1.0 / fps
        self._metrics = metrics
        self._spin_window_s = spin_window_s
        self._next_deadline = time.perf_counter()
        self._handle = self._create_timer_handle()

    def _create_timer_handle(self):
        if _CreateWaitableTimerExW is None:
            return None
        try:
            return _CreateWaitableTimerExW(
                None,
                None,
                _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                _TIMER_ALL_ACCESS,
            )
        except Exception:
            return None

    def reset(self) -> None:
        self._next_deadline = time.perf_counter()

    def wait_next(self, stop_event: threading.Event | None = None) -> None:
        self._next_deadline += self._interval_s
        self._wait_until(self._next_deadline, stop_event)
        if self._metrics is not None:
            late_ms = max(0.0, (time.perf_counter() - self._next_deadline) * 1000.0)
            self._metrics.record_pacer_wait(late_ms)

    def _wait_until(self, deadline: float, stop_event: threading.Event | None) -> None:
        while True:
            if stop_event is not None and stop_event.is_set():
                return

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return

            if self._handle and _SetWaitableTimer and _WaitForSingleObject and remaining > (self._spin_window_s + 0.001):
                coarse_wait_s = remaining - self._spin_window_s
                due_time_100ns = -max(1, int(coarse_wait_s * 10_000_000))
                due_time = _ctypes.c_longlong(due_time_100ns)
                if _SetWaitableTimer(self._handle, _ctypes.byref(due_time), 0, None, None, False):
                    result = _WaitForSingleObject(self._handle, _INFINITE)
                    if result == _WAIT_OBJECT_0:
                        continue
                    if result == _WAIT_FAILED:
                        self.close()
                        continue

            if remaining > self._spin_window_s:
                coarse_fallback_s = min(remaining - self._spin_window_s, 0.005)
                if stop_event is not None:
                    stop_event.wait(coarse_fallback_s)
                else:
                    time.sleep(coarse_fallback_s)
                continue

            while time.perf_counter() < deadline:
                if stop_event is not None and stop_event.is_set():
                    return
            return

    def close(self) -> None:
        if self._handle and _CancelWaitableTimer and _CloseHandle:
            try:
                _CancelWaitableTimer(self._handle)
            except Exception:
                pass
            try:
                _CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None


# ─────────────────────────────────────────────────────────────
# Media catalog
# ─────────────────────────────────────────────────────────────
def scan_catalog() -> list[dict]:
    items: list[dict] = []
    for d in MEDIA_SCAN_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT:
                items.append({"path": p, "type": "video", "name": p.name})
    for d in IMAGE_SCAN_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT:
                items.append({"path": p, "type": "image", "name": p.name})
    return items


# ─────────────────────────────────────────────────────────────
# MediaMTX relay
# ─────────────────────────────────────────────────────────────
class MediaMtx:
    def __init__(self, exe: Path) -> None:
        self._exe = exe
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        cfg = self._exe.parent / "bridge.yml"
        cfg.write_text(
            f"rtspAddress: :{RTSP_PORT}\n"
            "paths:\n"
            "  all_others:\n"
            "    source: publisher\n",
            encoding="utf-8",
        )
        self._proc = subprocess.Popen(
            [str(self._exe), str(cfg)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=str(self._exe.parent),
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("MediaMTX exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", RTSP_PORT), timeout=0.3):
                    return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("MediaMTX RTSP port never opened")

    def stop(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


# ─────────────────────────────────────────────────────────────
# Source decoder
#   • One FFmpeg subprocess per source
#   • Outputs raw NV12 4K30 frames to stdout
#   • Reader thread pushes frames to an internal queue
#   • Caller calls .get() for the next frame (non-blocking)
# ─────────────────────────────────────────────────────────────
def _build_vf() -> str:
    # force_divisible_by=2 ensures even dimensions for H.264.
    # fps filter is last so all prior filters run at source fps.
    return (
        f"scale={OUTPUT_W}:{OUTPUT_H}"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2:out_range=tv,"
        f"pad={OUTPUT_W}:{OUTPUT_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={OUTPUT_FPS},setsar=1,format=nv12"
    )


class SourceDecoder:
    def __init__(self, ffmpeg: Path, path: Path, is_image: bool, metrics: StreamMetrics) -> None:
        self._ffmpeg  = ffmpeg
        self._path    = path
        self._is_image = is_image
        self._metrics = metrics
        # 16-frame ring buffer (~0.5 s at 30 fps).
        # -re throttles the decoder to realtime so this buffer smooths
        # brief read/write jitter without accumulating real drift.
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=16)
        self._last_frame: bytes = BLACK_FRAME   # freeze instead of black on starvation
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop  = threading.Event()

    def start(self) -> None:
        if self._is_image:
            # -re with -loop 1 outputs one frame per 1/fps seconds (realtime).
            input_args = ["-re", "-loop", "1", "-framerate", str(OUTPUT_FPS), "-i", str(self._path)]
        else:
            # -re is CRITICAL: without it FFmpeg decodes at full CPU/GPU speed,
            # fills the queue in <200 ms, then put_nowait drops content frames.
            # Result without -re: bridge reads every N-th frame → N× playback speed.
            input_args = ["-re", "-stream_loop", "-1", "-i", str(self._path)]

        cmd = [
            str(self._ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
            # Allow FFmpeg to use several threads for software decode/scale.
            "-threads", "4",
            *input_args,
            "-vf", _build_vf(),
            "-an", "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=FRAME_BYTES,
        )
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"dec:{self._path.name[:24]}",
        )
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while not self._stop.is_set():
            try:
                data = self._proc.stdout.read(FRAME_BYTES)
            except Exception:
                self._metrics.record_decoder_read_error()
                break
            if len(data) < FRAME_BYTES:
                break  # EOF or truncated — source finished
            self._last_frame = data  # always track latest frame for freeze fallback
            self._metrics.record_decoder_frame()
            # Non-blocking put: if queue is full the decoder is already well ahead;
            # drop this frame rather than blocking (which would stall the pipe and
            # cause the 500 ms burst-stall-burst pattern).
            try:
                self._q.put_nowait(data)
            except queue.Full:
                self._metrics.record_decoder_drop()
                pass

    def get(self) -> tuple[bytes, bool]:
        """Return (frame, used_fallback) where fallback means queue underrun."""
        try:
            frame = self._q.get_nowait()
            self._last_frame = frame
            self._metrics.record_queue_depth(self._q.qsize())
            return frame, False
        except queue.Empty:
            self._metrics.record_queue_depth(0)
            return self._last_frame, True  # freeze instead of black on momentary starvation

    @property
    def queue_size(self) -> int:
        return self._q.qsize()

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
                if self._proc.stdout:
                    self._proc.stdout.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────
# Stream encoder
#   • Single persistent FFmpeg process: rawvideo NV12 stdin → h264_qsv → RTSP
#   • Never restarted on source switch
# ─────────────────────────────────────────────────────────────
class StreamEncoder:
    def __init__(self, ffmpeg: Path, encoder: str, metrics: StreamMetrics) -> None:
        self._ffmpeg  = ffmpeg
        self._encoder = encoder
        self._metrics = metrics
        self._proc: subprocess.Popen | None = None
        self._pending_lock = threading.Lock()
        self._pending_frame: bytes | None = None
        self._pending_event = threading.Event()
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None

    def start(self) -> None:
        use_qsv  = self._encoder == "h264_qsv"
        use_nvenc = self._encoder == "h264_nvenc"
        maxrate  = int(BITRATE_K * 1.5)
        bufsize  = int(BITRATE_K * 2)
        gop_size = ENCODER_GOP

        cmd = [str(self._ffmpeg), "-hide_banner", "-loglevel", "warning", "-nostdin"]
        if use_qsv:
            cmd += ["-init_hw_device", "qsv=hw"]
        cmd += [
            "-f", "rawvideo", "-pixel_format", "nv12",
            "-video_size", f"{OUTPUT_W}x{OUTPUT_H}",
            "-framerate", str(OUTPUT_FPS),
            "-i", "pipe:0",
            "-an",
            "-c:v", self._encoder,
        ]
        if use_qsv:
            cmd += [
                "-preset", "veryfast",
                "-scenario", "livestreaming",
                "-low_delay_brc", "1",
                "-bitrate_limit", "1",
                "-skip_frame", "insert_dummy",
                "-bf", "0", "-async_depth", "1", "-look_ahead", "0",
                "-g", str(gop_size), "-keyint_min", str(gop_size),
            ]
        elif use_nvenc:
            cmd += ["-preset", "p4", "-tune", "ll", "-bf", "0", "-g", str(gop_size)]
        else:  # libx264 fallback
            cmd += [
                "-preset", "veryfast", "-tune", "zerolatency",
                "-x264-params", f"repeat-headers=1:keyint={gop_size}:min-keyint={gop_size}:scenecut=0",
            ]
        cmd += [
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-b:v", f"{BITRATE_K}k", "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
            # GOP is configurable via STREAM_BRIDGE_GOP (default=1 for all-IDR).
            "-f", "rtsp", "-rtsp_transport", "tcp", RTSP_URL,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            bufsize=0,  # unbuffered: each write() goes directly to the OS pipe
        )
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="encoder-writer")
        self._writer_thread.start()

    def submit(self, frame: bytes) -> bool:
        if not (self._proc and self._proc.stdin) or self._proc.poll() is not None:
            return False
        with self._pending_lock:
            if self._pending_frame is not None:
                self._metrics.record_encoder_overwrite()
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

            if frame is None:
                continue
            if not (self._proc and self._proc.stdin):
                continue

            started = time.perf_counter()
            try:
                self._proc.stdin.write(frame)
                self._metrics.record_encoder_write((time.perf_counter() - started) * 1000.0, True)
            except OSError:
                self._metrics.record_encoder_write((time.perf_counter() - started) * 1000.0, False)
                return

    @property
    def alive(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)

    def stop(self) -> None:
        self._stop_event.set()
        self._pending_event.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=2.0)
        if self._proc:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


# ─────────────────────────────────────────────────────────────
# Frame bridge
#   • Dedicated thread: reads from active SourceDecoder at 30 fps
#     and writes raw NV12 frames to StreamEncoder stdin
#   • Source switch is atomic (lock-protected pointer swap)
#   • Black frames fill gaps (startup, switch preroll, EOF)
# ─────────────────────────────────────────────────────────────
class FrameBridge:
    def __init__(self, enc: StreamEncoder, metrics: StreamMetrics) -> None:
        self._enc   = enc
        self._metrics = metrics
        self._dec: SourceDecoder | None = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sent_frame: bytes = BLACK_FRAME

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-bridge")
        self._thread.start()

    def switch(self, new_dec: SourceDecoder) -> None:
        """Swap in new decoder; stop old one in background."""
        with self._lock:
            old = self._dec
            self._dec = new_dec
        if old:
            threading.Thread(target=old.stop, daemon=True).start()

    def _loop(self) -> None:
        # Python is the sole pacer: one frame every 1/30 s.
        # The decoder runs with -re (realtime) so the queue stays at 0-4 frames
        # in steady state; the 16-frame buffer just absorbs timing jitter.
        # get() is always non-blocking: if the queue is momentarily empty it
        # returns the last decoded frame (freeze), not black, not a stall.
        pacer = FramePacer(OUTPUT_FPS, self._metrics)
        try:
            while not self._stop.is_set():
                with self._lock:
                    dec = self._dec
                if dec is not None:
                    frame, freeze = dec.get()
                else:
                    frame, freeze = BLACK_FRAME, False
                black = dec is None

                if not self._enc.submit(frame):
                    # Encoder not ready yet — backoff and reset the pacer so we
                    # don't burst-send a backlog right after recovery.
                    self._stop.wait(0.1)
                    pacer.reset()
                    continue

                self._last_sent_frame = frame
                self._metrics.record_bridge_frame(freeze=freeze, black=black)
                pacer.wait_next(self._stop)
        finally:
            pacer.close()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            dec = self._dec
            self._dec = None
        if dec:
            dec.stop()


# ─────────────────────────────────────────────────────────────
# GUI — colours (Catppuccin Mocha-inspired)
# ─────────────────────────────────────────────────────────────
C_BG     = "#1e1e2e"
C_PANEL  = "#181825"
C_CARD   = "#2a2a3e"
C_HOVER  = "#313244"
C_SEL    = "#45475a"
C_FG     = "#cdd6f4"
C_DIM    = "#6c7086"
C_ACCENT = "#cba6f7"
C_VBADGE = "#89b4fa"   # blue — video
C_IBADGE = "#a6e3a1"   # green — image

THUMB_W, THUMB_H = 64, 36   # thumbnail size (16:9 ratio)


# ─────────────────────────────────────────────────────────────
# Media card widget
# ─────────────────────────────────────────────────────────────
class MediaCard(tk.Frame):
    def __init__(self, parent: tk.Widget, item: dict, on_select, ffmpeg: Path) -> None:
        super().__init__(parent, bg=C_CARD, cursor="hand2")
        self._item     = item
        self._selected = False

        badge_color = C_VBADGE if item["type"] == "video" else C_IBADGE
        badge = tk.Label(
            self,
            text=item["type"][:3].upper(),
            bg=badge_color, fg="#11111b",
            font=("Consolas", 7, "bold"), padx=4, width=4,
        )
        badge.pack(side="left", padx=(6, 4), pady=8)

        # Thumbnail placeholder (filled asynchronously if Pillow available)
        self._thumb_label: tk.Label | None = None
        if _PILLOW:
            self._thumb_label = tk.Label(self, bg=C_CARD, width=THUMB_W, height=THUMB_H)
            self._thumb_label.pack(side="left", padx=(0, 6), pady=4)
            threading.Thread(
                target=self._load_thumb,
                args=(ffmpeg, item),
                daemon=True,
            ).start()

        name = tk.Label(
            self,
            text=item["name"],
            bg=C_CARD, fg=C_FG,
            font=("Consolas", 9), anchor="w",
            wraplength=170, justify="left",
        )
        name.pack(side="left", fill="x", expand=True, pady=8, padx=(0, 6))

        for w in (self, badge, name) + ((self._thumb_label,) if self._thumb_label else ()):
            w.bind("<Button-1>", lambda _: on_select(item))
            w.bind("<Enter>",    self._hover_in)
            w.bind("<Leave>",    self._hover_out)

    # ── thumbnail ────────────────────────────────────────────
    def _load_thumb(self, ffmpeg: Path, item: dict) -> None:
        if not _PILLOW or not self._thumb_label:
            return
        try:
            if item["type"] == "image":
                img = Image.open(item["path"]).convert("RGB")
            else:
                result = subprocess.run(
                    [
                        str(ffmpeg), "-hide_banner", "-loglevel", "error",
                        "-i", str(item["path"]),
                        "-vframes", "1", "-vf", f"scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=decrease",
                        "-f", "image2pipe", "-vcodec", "bmp", "pipe:1",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
                if not result.stdout:
                    return
                import io
                img = Image.open(io.BytesIO(result.stdout)).convert("RGB")

            img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            # Schedule GUI update on main thread
            self._thumb_label.after(0, lambda p=photo: self._set_thumb(p))
        except Exception:
            pass  # thumbnail is optional

    def _set_thumb(self, photo) -> None:
        if self._thumb_label and self._thumb_label.winfo_exists():
            self._thumb_label.config(image=photo, width=THUMB_W, height=THUMB_H)
            self._thumb_label.image = photo  # keep reference

    # ── selection state ──────────────────────────────────────
    def set_selected(self, v: bool) -> None:
        self._selected = v
        bg = C_SEL if v else C_CARD
        self.config(bg=bg)
        for w in self.winfo_children():
            if isinstance(w, tk.Label) and w.cget("bg") not in (C_VBADGE, C_IBADGE):
                w.config(bg=bg)

    def _hover_in(self, _) -> None:
        if not self._selected:
            self.config(bg=C_HOVER)

    def _hover_out(self, _) -> None:
        if not self._selected:
            self.config(bg=C_CARD)


# ─────────────────────────────────────────────────────────────
# Main application window
# ─────────────────────────────────────────────────────────────
class App:
    def __init__(
        self,
        ffmpeg:   Path,
        encoder:  str,
        mediamtx: MediaMtx,
        enc:      StreamEncoder,
        bridge:   FrameBridge,
        metrics:  StreamMetrics,
        items:    list[dict],
        profile_seconds: float | None = None,
    ) -> None:
        self._ffmpeg   = ffmpeg
        self._encoder  = encoder
        self._mediamtx = mediamtx
        self._enc      = enc
        self._bridge   = bridge
        self._metrics  = metrics
        self._items    = items
        self._profile_seconds = profile_seconds
        self._cards:   list[MediaCard] = []
        self._active:  dict | None = None
        self._cancel   = threading.Event()   # cancels in-flight preroll
        self._metrics_last_ts = time.perf_counter()
        self._metrics_prev = self._metrics.snapshot()

        self._root = tk.Tk()
        self._root.title("Stream Bridge  ▶  " + RTSP_URL)
        self._root.configure(bg=C_BG)
        self._root.geometry("900x580")
        self._root.minsize(700, 400)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Status bar (bottom)
        self._status = tk.StringVar(value="Starting…")
        tk.Label(
            self._root, textvariable=self._status,
            bg="#313244", fg=C_FG,
            font=("Consolas", 9), anchor="w", padx=10, pady=3,
        ).pack(side="bottom", fill="x")

        # Left panel — catalog
        left = tk.Frame(self._root, bg=C_PANEL, width=290)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(
            left, text=" ▶  CATALOG",
            bg=C_PANEL, fg=C_ACCENT,
            font=("Consolas", 10, "bold"), anchor="w", pady=8,
        ).pack(fill="x", padx=6)

        # Scrollable catalog
        canvas = tk.Canvas(left, bg=C_PANEL, highlightthickness=0)
        sb = tk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C_PANEL)
        cw = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(cw, width=e.width),
        )
        canvas.bind(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"),
        )

        for item in self._items:
            card = MediaCard(inner, item, self._select, self._ffmpeg)
            card.pack(fill="x", padx=4, pady=2)
            self._cards.append(card)

        if not self._items:
            tk.Label(
                inner,
                text=(
                    "No media files found.\n\n"
                    "Add files to:\n"
                    "  • windows\\ref_videos\\\n"
                    "  • windows\\ref_images\\\n"
                    "  • E:\\SequencesVideo\\"
                ),
                bg=C_PANEL, fg=C_DIM,
                font=("Consolas", 9), justify="left",
            ).pack(padx=10, pady=16)

        # Right panel — stream info
        right = tk.Frame(self._root, bg=C_BG)
        right.pack(side="right", fill="both", expand=True, padx=20, pady=14)

        tk.Label(
            right, text="STREAM INFO",
            bg=C_BG, fg=C_ACCENT,
            font=("Consolas", 11, "bold"), anchor="w",
        ).pack(fill="x", pady=(0, 12))

        ip = local_ip()
        rows = [
            ("RTSP (local)", RTSP_URL),
            ("RTSP (LAN)",   f"rtsp://{ip}:{RTSP_PORT}/{RTSP_PATH}"),
            ("Resolution",   f"{OUTPUT_W} × {OUTPUT_H} @ {OUTPUT_FPS} fps"),
            ("Encoder",      self._encoder),
            ("Bitrate",      f"{BITRATE_K:,} kbps"),
        ]
        for label, val in rows:
            row = tk.Frame(right, bg=C_BG)
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text=f"{label}:",
                bg=C_BG, fg=C_DIM,
                font=("Consolas", 9), width=15, anchor="w",
            ).pack(side="left")
            tk.Label(
                row, text=val,
                bg=C_BG, fg=C_FG,
                font=("Consolas", 9), anchor="w",
            ).pack(side="left")

        tk.Label(
            right,
            text="\nVLC  →  Media  →  Open Network Stream  →  paste RTSP URL\n"
                 "(stream is live as soon as a source is selected)",
            bg=C_BG, fg=C_DIM,
            font=("Consolas", 8), justify="left",
        ).pack(anchor="w")

        # Now-playing label
        self._now = tk.Label(
            right, text="",
            bg=C_BG, fg=C_ACCENT,
            font=("Consolas", 10), anchor="w", wraplength=520,
        )
        self._now.pack(anchor="w", pady=(20, 0))

        self._metrics_var = tk.StringVar(value="runtime metrics pending…")
        tk.Label(
            right, text="\nRUNTIME METRICS",
            bg=C_BG, fg=C_ACCENT,
            font=("Consolas", 10, "bold"), anchor="w",
        ).pack(anchor="w", pady=(18, 6))
        tk.Label(
            right,
            textvariable=self._metrics_var,
            bg=C_BG,
            fg=C_FG,
            justify="left",
            anchor="w",
            font=("Consolas", 8),
        ).pack(anchor="w")

    def _refresh_metrics(self) -> None:
        now = time.perf_counter()
        snap = self._metrics.snapshot()
        elapsed = max(0.001, now - self._metrics_last_ts)
        prev = self._metrics_prev

        decode_fps = (snap.decoder_frames - prev.decoder_frames) / elapsed
        output_fps = (snap.encoder_frames_written - prev.encoder_frames_written) / elapsed
        freeze_fps = (snap.bridge_freeze_frames - prev.bridge_freeze_frames) / elapsed
        drop_fps = (snap.decoder_drops - prev.decoder_drops) / elapsed
        overwrite_fps = (snap.encoder_overwrites - prev.encoder_overwrites) / elapsed

        write_calls_delta = snap.encoder_write_calls - prev.encoder_write_calls
        write_total_delta = snap.encoder_write_total_ms - prev.encoder_write_total_ms
        write_avg_ms = (write_total_delta / write_calls_delta) if write_calls_delta > 0 else 0.0

        pacer_waits_delta = snap.pacer_waits - prev.pacer_waits
        pacer_total_delta = snap.pacer_late_total_ms - prev.pacer_late_total_ms
        pacer_avg_ms = (pacer_total_delta / pacer_waits_delta) if pacer_waits_delta > 0 else 0.0

        self._metrics_var.set(
            "\n".join([
                f"decode fps      : {decode_fps:5.1f}    drops/s : {drop_fps:4.1f}",
                f"output fps      : {output_fps:5.1f}    freeze/s: {freeze_fps:4.1f}",
                f"queue depth     : {snap.queue_depth:2d} / {snap.queue_depth_max:2d} max",
                f"enc overwrite/s : {overwrite_fps:5.1f}",
                f"enc write avg   : {write_avg_ms:6.2f} ms   max(run): {snap.encoder_write_max_ms:6.2f} ms",
                f"enc failures    : {snap.encoder_write_failures}",
                f"pacer late avg  : {pacer_avg_ms:6.3f} ms   max(run): {snap.pacer_late_max_ms:6.3f} ms",
                f"black frames    : {snap.bridge_black_frames}   read errors: {snap.decoder_read_errors}",
            ])
        )

        self._metrics_last_ts = now
        self._metrics_prev = snap
        if self._root.winfo_exists():
            self._root.after(500, self._refresh_metrics)

    # ── source selection ──────────────────────────────────────
    def _select(self, item: dict) -> None:
        if item is self._active:
            return
        self._active = item

        for card in self._cards:
            card.set_selected(card._item is item)
        self._now.config(text=f"▶  {item['name']}")
        self._status.set(f"Loading: {item['name']} …")

        # Cancel any previous in-flight preroll
        self._cancel.set()
        self._cancel = threading.Event()
        cancel = self._cancel  # capture for closure

        def _preroll() -> None:
            dec = SourceDecoder(self._ffmpeg, item["path"], item["type"] == "image", self._metrics)
            dec.start()
            # Wait until the decoder has buffered a few frames or a hard deadline
            # is reached. This avoids an initial underrun right after switching.
            preroll_target = min(8, dec._q.maxsize)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if cancel.is_set():
                    dec.stop()
                    return
                if dec.queue_size >= preroll_target:
                    break
                time.sleep(0.015)
            if not cancel.is_set():
                self._bridge.switch(dec)
                self._root.after(0, lambda: self._status.set(f"Streaming: {item['name']}"))

        threading.Thread(target=_preroll, daemon=True).start()

    # ── lifecycle ─────────────────────────────────────────────
    def run(self) -> None:
        self._status.set("Ready — select a source from the catalog")
        self._root.after(250, self._refresh_metrics)
        if self._profile_seconds and self._profile_seconds > 0:
            self._root.after(int(self._profile_seconds * 1000), self._on_close)
        # Auto-select first item after a short delay
        if self._items:
            self._root.after(200, lambda: self._select(self._items[0]))
        self._root.mainloop()

    def _on_close(self) -> None:
        self._cancel.set()
        self._bridge.stop()
        self._enc.stop()
        self._mediamtx.stop()
        self._root.destroy()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    print("=" * 62)
    print("  Stream Bridge — QSV/NVENC hardware RTSP streaming")
    print("=" * 62)

    print("\n[1/5] Locating FFmpeg …")
    ffmpeg = find_ffmpeg()
    print(f"      {ffmpeg}")

    print("[2/5] Resolving encoder …")
    encoder = resolve_encoder(ffmpeg)
    print(f"      {encoder}")

    metrics = StreamMetrics()
    profiler: ProfileRecorder | None = None
    if args.profile:
        profile_dir = Path(args.profile_dir).expanduser().resolve() if args.profile_dir else (
            SCRIPT_DIR / "profiles" / time.strftime("%Y%m%d-%H%M%S")
        )
        profiler = ProfileRecorder(metrics, profile_dir)
        profiler.start()
        print(f"      profiling enabled -> {profile_dir}")

    print("[3/5] Starting MediaMTX relay …")
    try:
        mtx = MediaMtx(find_mediamtx())
        mtx.start()
    except Exception as exc:
        print(f"[!]   MediaMTX failed: {exc}")
        sys.exit(1)
    print(f"      RTSP relay listening on :{RTSP_PORT}")

    print("[4/5] Starting stream encoder (FFmpeg → RTSP) …")
    enc = StreamEncoder(ffmpeg, encoder, metrics)
    enc.start()
    bridge = FrameBridge(enc, metrics)
    bridge.start()

    print("[5/5] Scanning media catalog …")
    items = scan_catalog()
    print(f"      {len(items)} item(s) found")
    if not items:
        print("      [!] No media found — add files to ref_videos/ or ref_images/")

    print(f"\n  Stream URL : {RTSP_URL}")
    print(f"  LAN URL    : rtsp://{local_ip()}:{RTSP_PORT}/{RTSP_PATH}")
    print("\n  Opening GUI …\n")

    try:
        app = App(ffmpeg, encoder, mtx, enc, bridge, metrics, items, args.profile_seconds)
        app.run()
    finally:
        if profiler is not None:
            csv_path, png_path = profiler.stop()
            print(f"\n[profile] CSV : {csv_path}")
            if png_path is not None:
                print(f"[profile] PNG : {png_path}")
            else:
                print("[profile] PNG : matplotlib unavailable, CSV only")


if __name__ == "__main__":
    main()

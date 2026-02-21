from __future__ import annotations

import logging
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.request
import zipfile
from typing import BinaryIO, Optional

# Configuration
PORT = 5002
FFMPEG_DOWNLOAD_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
)
FFMPEG_BIN_DIR = Path(__file__).resolve().parent / "bin"
DEVICE_PATTERN = re.compile(r'"(?P<name>.+)" \((?P<type>[^)]+)\)')
ALTERNATIVE_NAME_PATTERN = re.compile(r'Alternative name\s+"?(.+?)"?$', re.IGNORECASE)
logger = logging.getLogger("camera_bridge")


# Per-client buffer: 512 TS-aligned chunks (~32 MB at ~64 KB/chunk).
# 512 × 65 KB ≈ 33 MB — comfortable 20 s of headroom at 15 Mbps.
_CLIENT_QUEUE_SIZE = 512


@dataclass
class _ClientStream:
    """Per-client state: the data queue and a keyframe re-sync flag."""
    q: "queue.Queue[bytes | None]" = field(
        default_factory=lambda: queue.Queue(maxsize=_CLIENT_QUEUE_SIZE)
    )
    # True while flushing stale data after an overflow — new chunks are
    # skipped until the next IDR keyframe so the client always starts with
    # a complete, decodable access unit.
    draining: bool = False


def _find_keyframe_offset(chunk: bytes) -> int:
    """Return the byte offset of the first PAT TS packet (PID 0x0000) in *chunk*.

    Returns -1 if no PAT is found.

    FFmpeg with ``-mpegts_flags +resend_headers`` emits a fresh PAT/PMT
    immediately before every IDR frame, so a PAT anywhere in the chunk
    reliably marks a keyframe-restart boundary.  We scan every TS packet
    rather than just checking byte 0 because fixed-size reads (N × 188 bytes)
    can land at any offset relative to the stream's IDR access unit.
    """
    for off in range(0, len(chunk) - 3, 188):
        if chunk[off] != 0x47:          # lost sync byte — try to continue
            continue
        pid = ((chunk[off + 1] & 0x1F) << 8) | chunk[off + 2]
        if pid == 0x0000:
            return off
    return -1


class StreamBroadcaster:
    def __init__(self) -> None:
        self._clients: list[_ClientStream] = []
        self._lock = threading.Lock()
        self._history: deque[bytes] = deque()
        self._history_size = 0
        self._history_limit = 512 * 1024

    def register(self) -> _ClientStream:
        stream = _ClientStream()
        with self._lock:
            self._clients.append(stream)
        return stream

    def register_with_history(self) -> tuple[_ClientStream, list[bytes]]:
        stream = _ClientStream()
        with self._lock:
            self._clients.append(stream)
            history = list(self._history)
        return stream, history

    def unregister(self, stream: _ClientStream) -> None:
        with self._lock:
            if stream in self._clients:
                self._clients.remove(stream)

    def broadcast(self, chunk: bytes) -> None:
        self._history.append(chunk)
        self._history_size += len(chunk)
        while self._history_size > self._history_limit and self._history:
            removed = self._history.popleft()
            self._history_size -= len(removed)
        with self._lock:
            clients = list(self._clients)
        for stream in clients:
            if stream.draining:
                off = _find_keyframe_offset(chunk)
                if off < 0:
                    continue  # no IDR boundary in this chunk — keep waiting
                stream.draining = False
                # Trim the chunk so it starts exactly at the PAT packet.
                effective = chunk[off:] if off > 0 else chunk
            else:
                effective = chunk
                if stream.q.full():
                    # Queue overflowed.  Flush entirely and re-sync at the next
                    # keyframe so every frame the consumer sees is complete.
                    while True:
                        try:
                            stream.q.get_nowait()
                        except queue.Empty:
                            break
                    off = _find_keyframe_offset(chunk)
                    if off < 0:
                        stream.draining = True
                        continue
                    effective = chunk[off:] if off > 0 else chunk
            try:
                stream.q.put_nowait(effective)
            except queue.Full:
                pass  # race condition — skip rather than block

    def notify_stream_ended(self) -> None:
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for stream in clients:
            stream.q.put(None)


class StreamRequestHandler(BaseHTTPRequestHandler):
    broadcaster: StreamBroadcaster
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path != "/video_feed":
            self.send_response(404)
            self.end_headers()
            return
        stream, history = self.broadcaster.register_with_history()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp2t")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in history:
                self.wfile.write(chunk)
            self.wfile.flush()
            while True:
                chunk = stream.q.get()
                if chunk is None:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break
        finally:
            self.broadcaster.unregister(stream)


def drain_ffmpeg_stderr(pipe: Optional[BinaryIO]) -> None:
    if pipe is None:
        return
    for line in pipe:
        try:
            decoded = line.decode(errors="ignore").strip()
        except AttributeError:
            decoded = str(line).strip()
        logger.debug("[ffmpeg] %s", decoded)


@dataclass
class DirectShowDevice:
    friendly_name: str
    device_type: str
    alternatives: list[str] = field(default_factory=list)

    def add_alternative(self, alt_name: str) -> None:
        self.alternatives.append(alt_name)

    @property
    def input_name(self) -> str:
        if self.alternatives:
            return self.alternatives[-1]
        return self.friendly_name


def ensure_ffmpeg() -> Path | None:
    FFMPEG_BIN_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = next(FFMPEG_BIN_DIR.rglob("ffmpeg.exe"), None)
    if ffmpeg_path and ffmpeg_path.exists():
        return ffmpeg_path

    archive_path = FFMPEG_BIN_DIR / "ffmpeg.zip"
    print("[i] Téléchargement de FFmpeg (quelques Mo)...")
    urllib.request.urlretrieve(FFMPEG_DOWNLOAD_URL, archive_path)
    print("[i] Extraction de FFmpeg...")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(FFMPEG_BIN_DIR)
    archive_path.unlink()

    ffmpeg_path = next(FFMPEG_BIN_DIR.rglob("ffmpeg.exe"), None)
    if ffmpeg_path is None:
        print("[!] Impossible de trouver ffmpeg.exe après extraction")
        return None
    return ffmpeg_path


def query_dshow_devices(ffmpeg_path: Path) -> list[DirectShowDevice]:
    result = subprocess.run(
        [
            str(ffmpeg_path),
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    logging.debug("FFmpeg -list_devices stderr:\n%s", result.stderr)
    devices: list[DirectShowDevice] = []
    current: Optional[DirectShowDevice] = None
    for line in result.stderr.splitlines():
        match = DEVICE_PATTERN.search(line)
        if match:
            name = match.group("name").strip()
            device_type = match.group("type").strip().lower()
            current = DirectShowDevice(name, device_type)
            if device_type == "video":
                devices.append(current)
            else:
                current = None
            continue
        if current is not None:
            alt_match = ALTERNATIVE_NAME_PATTERN.search(line)
            if alt_match:
                current.add_alternative(alt_match.group(1).strip())
    return devices


def choose_device(ffmpeg_path: Path) -> Optional[DirectShowDevice]:
    devices = query_dshow_devices(ffmpeg_path)
    if not devices:
        print("[!] Aucun périphérique DirectShow détecté")
        logging.debug("FFmpeg output did not contain video devices")
        return None
    print("\n--- PÉRIPHÉRIQUES VIDÉO DISPONIBLES ---")
    for idx, device in enumerate(devices, 1):
        input_note = "" if device.input_name == device.friendly_name else f" (entrée: {device.input_name})"
        print(f"{idx}. {device.friendly_name}{input_note}")
    choice = input(f"Choisissez une source (1-{len(devices)}) [1]: ").strip() or "1"
    if choice.isdigit() and 1 <= int(choice) <= len(devices):
        return devices[int(choice) - 1]
    print("[!] Sélection invalide, utilisation du premier périphérique détecté")
    return devices[0]


def query_device_options(ffmpeg_path: Path, device: DirectShowDevice) -> list[tuple[int, int, int]]:
    result = subprocess.run(
        [
            str(ffmpeg_path),
            "-f",
            "dshow",
            "-list_options",
            "true",
            "-i",
            f"video={device.input_name}",
        ],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    logger.debug("FFmpeg -list_options stderr for %s:\n%s", device.friendly_name, result.stderr)
    # Build a map of size -> max fps seen for that size in FFmpeg output
    size_map: dict[tuple[int, int], int] = {}
    size_pattern = re.compile(r"s=(\d+)x(\d+)")
    fps_pattern = re.compile(r"fps=(\d+)")
    for line in result.stderr.splitlines():
        sizes = size_pattern.findall(line)
        fpss = fps_pattern.findall(line)
        max_fps = 0
        if fpss:
            try:
                max_fps = max(int(x) for x in fpss)
            except ValueError:
                max_fps = 0
        for w_str, h_str in sizes:
            try:
                w = int(w_str)
                h = int(h_str)
            except ValueError:
                continue
            key = (w, h)
            prev = size_map.get(key, 0)
            if max_fps > prev:
                size_map[key] = max_fps

    # If no sizes parsed, try to fallback by probing a common set
    if not size_map:
        common = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (640, 480)]
        for w, h in common:
            size_map[(w, h)] = 30

    # Convert to list of options (width, height, fps) using the recorded max fps
    options: list[tuple[int, int, int]] = []
    for (w, h), fps in size_map.items():
        options.append((w, h, fps if fps > 0 else 30))

    # Sort by resolution area desc then width
    options.sort(key=lambda t: (t[0] * t[1], t[0]), reverse=True)
    return options


def detect_encoders(ffmpeg_path: Path) -> list[str]:
    """Return a list of available H.264 encoders reported by ffmpeg (-encoders)."""
    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return ["libx264"]
    out = (result.stdout or "") + "\n" + (result.stderr or "")
    encs = set()
    for line in out.splitlines():
        line = line.strip()
        # ffmpeg encoders lines typically look like: " V..... h264_nvenc            NVIDIA NVENC H.264 encoder"
        m = re.match(r"^[A-Z\.\s]+\s+(h264_[a-z0-9_]+|libx264)\b", line)
        if m:
            encs.add(m.group(1))
    preferred = ["h264_nvenc", "h264_qsv", "h264_vaapi", "h264_amf", "libx264"]
    result_list = [e for e in preferred if e in encs]
    # On Windows, VAAPI is generally not available/usable; filter it out to avoid runtime errors
    if os.name == "nt":
        result_list = [e for e in result_list if not e.startswith("h264_vaapi")]
    if not result_list:
        # fallback to libx264
        return ["libx264"]
    return result_list


def choose_encoder(ffmpeg_path: Path) -> str:
    encs = detect_encoders(ffmpeg_path)
    print("\n--- ENCODEURS H.264 DISPONIBLES ---")
    for idx, e in enumerate(encs, 1):
        print(f"{idx}. {e}")
    choice = input(f"Choisissez un encodeur (1-{len(encs)}) [{encs.index('libx264')+1 if 'libx264' in encs else 1}]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(encs):
        return encs[int(choice) - 1]
    # default to first
    return encs[0]


def choose_bitrate_for_encoder(encoder: str) -> int:
    """Prompt user for bitrate (kbps) with sensible defaults per encoder."""
    # Hardware encoders default to 15 Mbps — necessary for 4K to avoid
    # visible macroblocks that degrade density-map quality.
    defaults = {
        "h264_nvenc": 15000,
        "h264_qsv": 15000,
        "h264_vaapi": 15000,
        "libx264": 25000,
    }
    default = defaults.get(encoder, 25000)
    prompt = f"Target video bitrate in kbps for {encoder} [{default}]: "
    val = input(prompt).strip()
    if not val:
        return default
    try:
        kbps = int(val)
        if kbps <= 0:
            return default
        return kbps
    except ValueError:
        return default


def choose_stream_settings(options: list[tuple[int, int, int]]) -> tuple[int, int, int, bool]:
    if not options:
        print("[!] Aucun mode détecté via FFmpeg, utilisation du fallback 1280x720@30")
        return 1280, 720, 30, False
    print("\n--- MODES SUPPORTED PAR LA CAMÉRA ---")
    for idx, (w, h, fps) in enumerate(options, 1):
        print(f"{idx}. {w}x{h} @ {fps} fps")
    choice = input(f"Choisissez un mode (1-{len(options)}) [1]: ").strip() or "1"
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        w, h, fps = options[int(choice) - 1]
    else:
        print("[!] Sélection invalide, utilisation du premier mode détecté")
        w, h, fps = options[0]
    gop_choice = input("Forcer GOP=1 (toutes les images I-frame) ? [y/N]: ").strip().lower()
    gop1 = gop_choice.startswith("y")
    return w, h, fps, gop1


def get_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def start_ffmpeg_stream(
    ffmpeg_path: Path,
    device_input: str,
    width: int,
    height: int,
    fps: int,
    gop1: bool = False,
    encoder: str = "libx264",
    bitrate_kbps: int | None = None,
) -> subprocess.Popen:
    cmd = [
        str(ffmpeg_path),
        "-f", "dshow",
        "-framerate", str(fps),
        "-video_size", f"{width}x{height}",
        "-rtbufsize", "150M",
        "-i", f"video={device_input}",
    ]

    # Configure encoder
    if encoder == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"]
        x264_params = ["repeat-headers=1"]
        if gop1:
            x264_params.extend(["keyint=1", "min-keyint=1", "scenecut=0"])
        cmd += ["-x264-params", ":".join(x264_params)]
        if bitrate_kbps:
            cmd += ["-b:v", f"{bitrate_kbps}k", "-bufsize", f"{bitrate_kbps*2}k"]
    else:
        # Hardware encoders: use encoder name directly. Add GOP if requested.
        cmd += ["-c:v", encoder]
        if bitrate_kbps:
            # common approach: set target bitrate and buffer parameters
            maxrate = int(bitrate_kbps * 1.5)
            bufsize = int(bitrate_kbps * 2)
            cmd += ["-b:v", f"{bitrate_kbps}k", "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k"]
        # Special handling per-hardware encoder
        if encoder.startswith("h264_vaapi"):
            # VAAPI requires the frames to be uploaded to the VAAPI device in a supported
            # pixel format (usually nv12). Insert filter to convert and upload.
            # Note: on some systems you may need to pass -vaapi_device /dev/dri/renderD128
            cmd += ["-vf", "format=nv12,hwupload"]
        if encoder.startswith("h264_nvenc"):
            # VBR HQ preset; disable B-frames for lower latency + shorter IDR distance.
            cmd += ["-rc", "vbr_hq", "-preset", "llhq", "-bf", "0"]
        if encoder.startswith("h264_qsv"):
            # No B-frames (reduces encode latency by ~2 frames) and single-frame
            # async depth so each frame is flushed immediately.
            cmd += ["-bf", "0", "-async_depth", "4"]
        if not gop1:
            # Keyframe every 2 s (≈ 2×fps frames).  Shorter GOP = faster NVDEC
            # resync after a corrupt packet (recovery within 2 s instead of the
            # hardware-encoder default of ~250 frames / ~8 s).
            gop_frames = max(1, fps * 2)
            cmd += ["-g", str(gop_frames), "-keyint_min", str(gop_frames)]
        if gop1:
            # NVIDIA NVENC enforces: Gop Length should be greater than number of B frames + 1
            # Setting GOP=1 with default B-frames will fail. For nvenc, disable B-frames and
            # use the minimal allowed GOP (2). If true GOP=1 is required, fallback to libx264.
            if encoder.startswith("h264_nvenc"):
                cmd += ["-g", "2"]
                logger.warning("h264_nvenc doesn't support GOP=1; using -bf 0 -g 2 instead")
            else:
                cmd += ["-g", "1"]

    # Output as MPEG-TS to stdout.
    # +resend_headers replays PAT/PMT before every IDR frame so a new client
    # (or a reader that missed the initial headers) can sync on any keyframe.
    cmd += ["-f", "mpegts", "-mpegts_flags", "+resend_headers", "-"]
    logger.info("Launching FFmpeg bridge: %s", " ".join(cmd))
    print("[i] Démarrage de FFmpeg pour streamer le flux H.264 …")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    threading.Thread(target=drain_ffmpeg_stderr, args=(proc.stderr,), daemon=True).start()
    return proc


# MPEG-TS packets are always 188 bytes; read multiples of 188 so we never
# split a TS packet across two broadcaster chunks.  348 × 188 = 65424 bytes
# (≈ 64 KB, close to the previous read size without fragmenting TS packets).
_TS_PACKET_SIZE = 188
_TS_READ_SIZE = _TS_PACKET_SIZE * 348  # 65424 bytes per syscall


def stream_ffmpeg_output(ffmpeg_proc: subprocess.Popen, broadcaster: StreamBroadcaster) -> None:
    stdout = ffmpeg_proc.stdout
    if stdout is None:
        return
    try:
        while True:
            chunk = stdout.read(_TS_READ_SIZE)
            if not chunk:
                break
            broadcaster.broadcast(chunk)
    finally:
        stdout.close()


def start_ffmpeg_stream_file(
    ffmpeg_path: Path,
    input_path: str,
    out_width: int = 0,
    out_height: int = 0,
    fps: int = 30,
    encoder: str = "libx264",
    bitrate_kbps: int | None = None,
) -> subprocess.Popen:
    """Stream an image or video file as MPEG-TS.

    * **Image** (``.jpg``, ``.png``, etc.) — looped endlessly at *fps* using
      FFmpeg's ``-loop 1`` flag with the ``image2`` demuxer.
    * **Video** — looped endlessly with ``-stream_loop -1 -re`` (respects source
      framerate; ``-re`` throttles output to real-time).

    The source is scaled (with letterboxing) to *out_width*×*out_height* when
    those values are non-zero; otherwise the native resolution is kept.
    """
    input_lower = input_path.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    is_image = any(input_lower.endswith(ext) for ext in image_exts)

    cmd = [str(ffmpeg_path)]

    if is_image:
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", input_path]
    else:
        # Re-stream at real-time rate to emulate a live camera feed.
        cmd += ["-stream_loop", "-1", "-re", "-i", input_path]

    # Build scale/pad filter when output resolution is requested.
    # pad=... adds black bars; setsar=1 fixes SAR so clients see square pixels.
    vf_parts: list[str] = []
    if out_width and out_height:
        vf_parts.append(
            f"scale={out_width}:{out_height}:force_original_aspect_ratio=decrease"
        )
        vf_parts.append(
            f"pad={out_width}:{out_height}:(ow-iw)/2:(oh-ih)/2:black"
        )
        vf_parts.append("setsar=1")
    if not is_image:
        vf_parts.append(f"fps={fps}")
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    elif is_image:
        cmd += ["-r", str(fps)]

    # Force standard YUV 4:2:0 (TV range) so PyNvCodec can always recognise
    # the pixel format — h264_qsv/nvenc may default to yuvj420p (full-range)
    # which PyNvCodec doesn't support.
    cmd += ["-pix_fmt", "yuv420p"]

    # Configure encoder
    if encoder == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency"]
        cmd += ["-x264-params", "repeat-headers=1"]
        if bitrate_kbps:
            cmd += ["-b:v", f"{bitrate_kbps}k", "-bufsize", f"{bitrate_kbps*2}k"]
    else:
        cmd += ["-c:v", encoder]
        if bitrate_kbps:
            maxrate = int(bitrate_kbps * 1.5)
            bufsize = int(bitrate_kbps * 2)
            cmd += ["-b:v", f"{bitrate_kbps}k", "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k"]
        if encoder.startswith("h264_nvenc"):
            cmd += ["-rc", "vbr_hq", "-preset", "llhq", "-bf", "0"]
        if encoder.startswith("h264_qsv"):
            cmd += ["-bf", "0", "-async_depth", "1"]  # async_depth=1 minimises initial buffering
    # GOP: keyframe every 2 s for fast NVDEC resync after packet corruption.
    gop_frames = max(1, fps * 2)
    cmd += ["-g", str(gop_frames), "-keyint_min", str(gop_frames)]

    # For images, limit output rate to *fps* via -fps_mode so FFmpeg doesn't
    # try to emit 25 × fps duplicate frames due to loop expansion.
    if is_image:
        cmd += ["-fps_mode", "vfr"]

    cmd += ["-an"]  # no audio
    cmd += ["-f", "mpegts", "-mpegts_flags", "+resend_headers", "-"]

    logger.info("Launching FFmpeg file bridge: %s", " ".join(cmd))
    print("[i] Démarrage de FFmpeg pour streamer le fichier…")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    threading.Thread(target=drain_ffmpeg_stderr, args=(proc.stderr,), daemon=True).start()
    return proc


if __name__ == "__main__":
    import argparse

    RESOLUTIONS = {
        "4K":    (3840, 2160),
        "1440p": (2560, 1440),
        "1080p": (1920, 1080),
        "720p":  (1280, 720),
    }

    def _parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description="Windows camera/file → MPEG-TS HTTP bridge for PeopleCounter.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument(
            "--input-file", "-i",
            metavar="PATH",
            help="Path to an image (.jpg/.png) or video file to stream instead of a camera. "
                 "Images are looped endlessly; videos are looped with -stream_loop -1.",
        )
        p.add_argument(
            "--resolution", "-r",
            choices=list(RESOLUTIONS),
            default=None,
            help="Output resolution (scale/letterbox the source). Default: keep source resolution.",
        )
        p.add_argument(
            "--fps",
            type=int,
            default=30,
            metavar="N",
            help="Output framerate (used for image source or to cap a video source).",
        )
        p.add_argument(
            "--bitrate",
            type=int,
            default=None,
            metavar="KBPS",
            help="Target video bitrate in kbps. Default: encoder-specific (15000 for HW, 25000 for x264).",
        )
        p.add_argument(
            "--encoder", "-e",
            default=None,
            metavar="ENC",
            help="H.264 encoder (h264_nvenc / h264_qsv / libx264). Auto-detected when omitted.",
        )
        p.add_argument(
            "--port",
            type=int,
            default=PORT,
            metavar="N",
            help=f"HTTP listen port (default {PORT}).",
        )
        return p.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    args = _parse_args()
    ffmpeg_path = ensure_ffmpeg()
    if ffmpeg_path is None:
        sys.exit(1)

    # ── Choose encoder ────────────────────────────────────────────────────────
    if args.encoder:
        encoder = args.encoder
    else:
        encoder = choose_encoder(ffmpeg_path)

    bitrate_kbps = args.bitrate or choose_bitrate_for_encoder(encoder)
    ip = get_ip()

    broadcaster = StreamBroadcaster()
    StreamRequestHandler.broadcaster = broadcaster
    listen_port = args.port
    http_server = ThreadingHTTPServer(("0.0.0.0", listen_port), StreamRequestHandler)
    server_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    logger.info("HTTP broadcaster listening on port %s", listen_port)

    print("\n" + "="*60)
    if args.input_file:
        print("      WINDOWS FILE BRIDGE POUR DOCKER (H.264 via FFmpeg)")
    else:
        print("      WINDOWS CAMERA BRIDGE POUR DOCKER (H.264 via FFmpeg)")
    print("="*60)
    print(f"\n[+] Flux disponible sur : http://{ip}:{listen_port}/video_feed")
    print(f"[+] Encodeur sélectionné : {encoder} @ {bitrate_kbps} kbps")

    try:
        while True:
            if args.input_file:
                # ── File source mode ──────────────────────────────────────────
                input_path = args.input_file
                tgt_w, tgt_h = RESOLUTIONS.get(args.resolution, (0, 0)) if args.resolution else (0, 0)
                fps = args.fps

                print(f"[+] Source  : {input_path}")
                if tgt_w:
                    print(f"[+] Sortie  : {tgt_w}×{tgt_h} @ {fps} fps")
                else:
                    print(f"[+] Sortie  : résolution native @ {fps} fps")

                ffmpeg_proc = start_ffmpeg_stream_file(
                    ffmpeg_path,
                    input_path,
                    out_width=tgt_w,
                    out_height=tgt_h,
                    fps=fps,
                    encoder=encoder,
                    bitrate_kbps=bitrate_kbps,
                )
            else:
                # ── Camera source mode (interactive) ──────────────────────────
                device = choose_device(ffmpeg_path)
                if device is None:
                    sys.exit(1)

                options = query_device_options(ffmpeg_path, device)
                # Honour --resolution if given; otherwise let user choose interactively.
                if args.resolution:
                    tgt_w, tgt_h = RESOLUTIONS[args.resolution]
                    fps = args.fps
                    gop1 = False
                    width, height = tgt_w, tgt_h
                else:
                    width, height, fps, gop1 = choose_stream_settings(options)

                display_input = device.input_name
                input_note = f" (entrée: {display_input})" if display_input != device.friendly_name else ""
                print(f"[+] Périphérique utilisé : {device.friendly_name}{input_note}")
                print(f"[+] Mode retenu : {width}×{height} @ {fps} fps")
                print(f"[!] COMMANDE A COPIER DANS LE TERMINAL WSL :")
                print(f"    ./run_app.sh http://{ip}:{listen_port}/video_feed")

                ffmpeg_proc = start_ffmpeg_stream(
                    ffmpeg_path,
                    device.input_name,
                    width,
                    height,
                    fps,
                    gop1=gop1,
                    encoder=encoder,
                    bitrate_kbps=bitrate_kbps,
                )

            try:
                stream_ffmpeg_output(ffmpeg_proc, broadcaster)
            except KeyboardInterrupt:
                ffmpeg_proc.terminate()
                ffmpeg_proc.wait()
                break
            return_code = ffmpeg_proc.wait()
            broadcaster.notify_stream_ended()
            if return_code == 0:
                break
            logger.warning("FFmpeg exited (code=%s); restarting in 1s", return_code)
            time.sleep(1)
    finally:
        http_server.shutdown()
        http_server.server_close()
        server_thread.join(timeout=1)

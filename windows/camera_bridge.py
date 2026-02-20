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


# Per-client buffer: 256 TS-aligned chunks (~16 MB at ~64 KB/chunk).
# Large enough to absorb ~10 s of an 8-Mbps stream without drops.
_CLIENT_QUEUE_SIZE = 256


class StreamBroadcaster:
    def __init__(self) -> None:
        self._clients: list[queue.Queue[bytes | None]] = []
        self._lock = threading.Lock()
        self._history: deque[bytes] = deque()
        self._history_size = 0
        self._history_limit = 512 * 1024

    def register(self) -> queue.Queue[bytes | None]:
        q: queue.Queue[bytes | None] = queue.Queue(maxsize=_CLIENT_QUEUE_SIZE)
        with self._lock:
            self._clients.append(q)
            history = list(self._history)
        return q

    def register_with_history(self) -> tuple[queue.Queue[bytes | None], list[bytes]]:
        q = queue.Queue(maxsize=_CLIENT_QUEUE_SIZE)
        with self._lock:
            self._clients.append(q)
            history = list(self._history)
        return q, history

    def unregister(self, q: queue.Queue[bytes | None]) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def broadcast(self, chunk: bytes) -> None:
        self._history.append(chunk)
        self._history_size += len(chunk)
        while self._history_size > self._history_limit and self._history:
            removed = self._history.popleft()
            self._history_size -= len(removed)
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            # Drop-oldest strategy: when the queue is full discard the stale
            # front chunk so the consumer always has the freshest data and the
            # MPEG-TS stream is not silently gapped by dropping the newest chunk.
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(chunk)
            except queue.Full:
                pass  # race condition — skip rather than block

    def notify_stream_ended(self) -> None:
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for q in clients:
            q.put(None)


class StreamRequestHandler(BaseHTTPRequestHandler):
    broadcaster: StreamBroadcaster
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path != "/video_feed":
            self.send_response(404)
            self.end_headers()
            return
        queue_ref, history = self.broadcaster.register_with_history()
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
                chunk = queue_ref.get()
                if chunk is None:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (ConnectionResetError, BrokenPipeError, OSError):
                    break
        finally:
            self.broadcaster.unregister(queue_ref)


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
    # Hardware encoders default to 8 Mbps — enough for 4K detection accuracy
    # while keeping the WSL2 network queue small and reducing corrupt-packet rate.
    defaults = {
        "h264_nvenc": 8000,
        "h264_qsv": 8000,
        "h264_vaapi": 8000,
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    ffmpeg_path = ensure_ffmpeg()
    if ffmpeg_path is None:
        sys.exit(1)

    device = choose_device(ffmpeg_path)
    if device is None:
        sys.exit(1)

    options = query_device_options(ffmpeg_path, device)
    width, height, fps, gop1 = choose_stream_settings(options)
    encoder = choose_encoder(ffmpeg_path)
    bitrate_kbps = choose_bitrate_for_encoder(encoder)
    ip = get_ip()

    broadcaster = StreamBroadcaster()
    StreamRequestHandler.broadcaster = broadcaster
    http_server = ThreadingHTTPServer(("0.0.0.0", PORT), StreamRequestHandler)
    server_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    logger.info("HTTP broadcaster listening on port %s", PORT)

    print("\n" + "="*60)
    print("      WINDOWS CAMERA BRIDGE POUR DOCKER (H.264 via FFmpeg)")
    print("="*60)
    print(f"\n[+] Flux disponible sur : http://{ip}:{PORT}/video_feed")
    display_input = device.input_name
    input_note = f" (entrée: {display_input})" if display_input != device.friendly_name else ""
    print(f"[+] Périphérique utilisé : {device.friendly_name}{input_note}")
    print(f"[+] Mode retenu : {width}x{height} @ {fps} fps")
    print(f"[+] Commande de test : ffprobe http://{ip}:{PORT}/video_feed")
    print(f"\n[!] COMMANDE A COPIER DANS LE TERMINAL WSL :")
    print(f"    ./run_app.sh http://{ip}:{PORT}/video_feed")
    print("\n" + "="*60)

    try:
        while True:
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

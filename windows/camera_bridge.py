from __future__ import annotations

import logging
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


class StreamBroadcaster:
    def __init__(self) -> None:
        self._clients: list[queue.Queue[bytes | None]] = []
        self._lock = threading.Lock()
        self._history: deque[bytes] = deque()
        self._history_size = 0
        self._history_limit = 512 * 1024

    def register(self) -> queue.Queue[bytes | None]:
        q: queue.Queue[bytes | None] = queue.Queue(maxsize=64)
        with self._lock:
            self._clients.append(q)
            history = list(self._history)
        return q

    def register_with_history(self) -> tuple[queue.Queue[bytes | None], list[bytes]]:
        q = queue.Queue(maxsize=64)
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
            try:
                q.put(chunk, block=False)
            except queue.Full:
                continue

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
    pattern = re.compile(r"size=(\d+)x(\d+).*?fps=(\d+)")
    options: list[tuple[int, int, int]] = []
    seen = set()
    for line in result.stderr.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        width, height, fps = map(int, match.groups())
        key = (width, height, fps)
        if key in seen:
            continue
        seen.add(key)
        options.append(key)
    return options


def choose_stream_settings(options: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    if not options:
        print("[!] Aucun mode détecté via FFmpeg, utilisation du fallback 1280x720@30")
        return 1280, 720, 30
    print("\n--- MODES SUPPORTED PAR LA CAMÉRA ---")
    for idx, (w, h, fps) in enumerate(options, 1):
        print(f"{idx}. {w}x{h} @ {fps} fps")
    choice = input(f"Choisissez un mode (1-{len(options)}) [1]: ").strip() or "1"
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return options[int(choice) - 1]
    print("[!] Sélection invalide, utilisation du premier mode détecté")
    return options[0]


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
) -> subprocess.Popen:
    cmd = [
        str(ffmpeg_path),
        "-f", "dshow",
        "-framerate", str(fps),
        "-video_size", f"{width}x{height}",
        "-rtbufsize", "150M",
        "-i", f"video={device_input}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-f", "mpegts",
        "-",
    ]
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


def stream_ffmpeg_output(ffmpeg_proc: subprocess.Popen, broadcaster: StreamBroadcaster) -> None:
    stdout = ffmpeg_proc.stdout
    if stdout is None:
        return
    try:
        while True:
            chunk = stdout.read(64 * 1024)
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
    width, height, fps = choose_stream_settings(options)
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
            ffmpeg_proc = start_ffmpeg_stream(ffmpeg_path, device.input_name, width, height, fps)
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

from __future__ import annotations

import logging
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
import urllib.request
import zipfile
from typing import Optional

# Configuration
PORT = 5002
FFMPEG_DOWNLOAD_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
)
FFMPEG_BIN_DIR = Path(__file__).resolve().parent / "bin"
DEVICE_PATTERN = re.compile(r'"(?P<name>.+)" \((?P<type>[^)]+)\)')
ALTERNATIVE_NAME_PATTERN = re.compile(r'Alternative name\s+"?(.+?)"?$', re.IGNORECASE)
logger = logging.getLogger("camera_bridge")


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
    if len(devices) == 1:
        print(f"[+] Utilisation du périphérique vidéo : {devices[0].friendly_name}")
        return devices[0]
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
        "-listen", "1",
        f"http://0.0.0.0:{PORT}/video_feed",
    ]
    logger.info("Launching FFmpeg bridge: %s", " ".join(cmd))
    print("[i] Démarrage de FFmpeg pour streamer le flux H.264 …")
    return subprocess.Popen(cmd)


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

    ffmpeg_proc = start_ffmpeg_stream(ffmpeg_path, device.input_name, width, height, fps)
    try:
        ffmpeg_proc.wait()
    except KeyboardInterrupt:
        ffmpeg_proc.terminate()
        ffmpeg_proc.wait()

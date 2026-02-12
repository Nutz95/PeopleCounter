from __future__ import annotations

import cv2
import socket
import logging
import subprocess
import sys
from pathlib import Path
import urllib.request
import zipfile

# Configuration
PORT = 5002
CAMERA_INDEX = 0
FFMPEG_DOWNLOAD_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
)
FFMPEG_BIN_DIR = Path(__file__).resolve().parent / "bin"


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


def select_resolution() -> tuple[int, int] | None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[!] Erreur: Impossible d'ouvrir la caméra {CAMERA_INDEX}")
        return None

    candidates = [
        (3840, 2160, "4K / UltraHD"),
        (2560, 1440, "2K / QHD"),
        (1920, 1080, "1080p / FullHD"),
        (1280, 720,  "720p / HD"),
        (800, 600,   "SVGA"),
        (640, 480,   "VGA"),
    ]

    print("\n[i] Détection des formats supportés par votre caméra...")
    supported: list[tuple[int, int, str]] = []
    for w, h, name in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res_key = (actual_w, actual_h)
        if res_key not in [s[:2] for s in supported]:
            label = next((c[2] for c in candidates if c[0] == actual_w and c[1] == actual_h), f"{actual_w}x{actual_h}")
            supported.append((actual_w, actual_h, label))

    supported.sort(key=lambda x: x[0], reverse=True)
    print("\n--- FORMATS DÉTECTÉS ---")
    for i, (w, h, label) in enumerate(supported, 1):
        print(f"{i}. {label} ({w}x{h})")
    print("0. Garder le réglage actuel")
    choice = input(f"\nChoisissez une option (0-{len(supported)}) [0]: ").strip() or "0"

    if choice != "0" and choice.isdigit() and int(choice) <= len(supported):
        w, h, name = supported[int(choice)-1]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        print(f"[+] Réglage validé sur {name}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Résolution active : {actual_w}x{actual_h}")
    cap.release()
    return actual_w, actual_h


def query_dshow_devices(ffmpeg_path: Path) -> list[str]:
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
    devices: list[str] = []
    recording = False
    for line in result.stderr.splitlines():
        if "DirectShow video devices" in line:
            recording = True
            continue
        if recording:
            line = line.strip()
            if not line:
                continue
            if line.startswith("\"") and line.endswith("\""):
                devices.append(line.strip('"'))
            elif line.startswith("\"" ):
                devices.append(line.split('"')[1])
            if line.startswith("DirectShow audio devices"):
                break
    return devices


def choose_device(ffmpeg_path: Path) -> str | None:
    devices = query_dshow_devices(ffmpeg_path)
    if not devices:
        print("[!] Aucun périphérique DirectShow détecté")
        return None
    if len(devices) == 1:
        print(f"[+] Utilisation du périphérique vidéo : {devices[0]}")
        return devices[0]
    print("\n--- PÉRIPHÉRIQUES VIDÉO DISPONIBLES ---")
    for idx, name in enumerate(devices, 1):
        print(f"{idx}. {name}")
    choice = input(f"Choisissez une source (1-{len(devices)}) [1]: ").strip() or "1"
    if choice.isdigit() and 1 <= int(choice) <= len(devices):
        return devices[int(choice) - 1]
    print("[!] Sélection invalide, utilisation du premier périphérique")
    return devices[0]


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


def start_ffmpeg_stream(ffmpeg_path: Path, device: str, width: int, height: int) -> subprocess.Popen:
    cmd = [
        str(ffmpeg_path),
        "-f", "dshow",
        "-framerate", "30",
        "-video_size", f"{width}x{height}",
        "-i", f"video={device}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-f", "mpegts",
        f"http://0.0.0.0:{PORT}/video_feed",
    ]
    print("[i] Démarrage de FFmpeg pour streamer le flux H.264 …")
    return subprocess.Popen(cmd)


if __name__ == "__main__":
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    ffmpeg_path = ensure_ffmpeg()
    if ffmpeg_path is None:
        sys.exit(1)

    resolution = select_resolution()
    if resolution is None:
        sys.exit(1)

    width, height = resolution
    ip = get_ip()
    device = choose_device(ffmpeg_path)
    if device is None:
        sys.exit(1)

    print("\n" + "="*60)
    print("      WINDOWS CAMERA BRIDGE POUR DOCKER (H.264 via FFmpeg)")
    print("="*60)
    print(f"\n[+] Flux disponible sur : http://{ip}:{PORT}/video_feed")
    print(f"[+] Périphérique utilisé : {device}")
    print(f"[+] Commande de test : ffprobe http://{ip}:{PORT}/video_feed")
    print(f"\n[!] COMMANDE A COPIER DANS LE TERMINAL WSL :")
    print(f"    ./run_app.sh http://{ip}:{PORT}/video_feed")
    print("\n" + "="*60)

    ffmpeg_proc = start_ffmpeg_stream(ffmpeg_path, device, width, height)
    try:
        ffmpeg_proc.wait()
    except KeyboardInterrupt:
        ffmpeg_proc.terminate()
        ffmpeg_proc.wait()

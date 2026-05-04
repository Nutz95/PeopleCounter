from __future__ import annotations

import argparse
import logging
import re
import shutil
import socket
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PORT = 5002
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "4K": (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_BIN_DIR = Path(__file__).resolve().parent / "bin"

logger = logging.getLogger("camera_bridge")


@dataclass
class DirectShowDevice:
    friendly_name: str
    device_type: str
    alternatives: list[str] = field(default_factory=list)

    @property
    def input_name(self) -> str:
        return self.friendly_name


@dataclass(frozen=True)
class DirectShowMode:
    width: int
    height: int
    fps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows camera / file bridge that serves VLC/NVDEC-friendly MPEG-TS over HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["camera", "media", "auto"], default="auto")
    parser.add_argument("--input-file", help="Initial image/video file to stream in media mode.")
    parser.add_argument("--media-dir", help="Directory containing playable videos.")
    parser.add_argument("--image-dir", help="Directory containing playable images.")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="4K")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=50000, help="Target bitrate in kbps.")
    parser.add_argument("--encoder", default="auto", help="FFmpeg video encoder (auto, h264_qsv, h264_nvenc, libx264).")
    parser.add_argument("--port", type=int, default=PORT)
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if any((args.input_file, args.media_dir, args.image_dir)):
        return "media"
    return "camera"


def ensure_ffmpeg() -> Path | None:
    FFMPEG_BIN_DIR.mkdir(parents=True, exist_ok=True)
    local_ffmpeg = next(FFMPEG_BIN_DIR.rglob("ffmpeg.exe"), None)
    if local_ffmpeg is not None and local_ffmpeg.exists():
        return local_ffmpeg

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return Path(ffmpeg_path)

    archive_path = FFMPEG_BIN_DIR / "ffmpeg.zip"
    print("[i] Downloading FFmpeg...")
    try:
        urllib.request.urlretrieve(FFMPEG_DOWNLOAD_URL, archive_path)
        print("[i] Extracting FFmpeg...")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(FFMPEG_BIN_DIR)
        archive_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[!] FFmpeg is not available in PATH and auto-download failed: {exc}")
        return None

    local_ffmpeg = next(FFMPEG_BIN_DIR.rglob("ffmpeg.exe"), None)
    if local_ffmpeg is None:
        print("[!] ffmpeg.exe not found after extraction")
        return None
    return local_ffmpeg


def has_encoder(ffmpeg_path: Path, encoder_name: str) -> bool:
    result = subprocess.run(
        [str(ffmpeg_path), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return encoder_name in (result.stdout + result.stderr)


def resolve_encoder(ffmpeg_path: Path, requested: str) -> str:
    if requested == "auto":
        for candidate in ("h264_qsv", "h264_nvenc", "libx264"):
            if has_encoder(ffmpeg_path, candidate):
                return candidate
        return "libx264"
    if not has_encoder(ffmpeg_path, requested):
        raise RuntimeError(f"Requested encoder '{requested}' is not available in FFmpeg.")
    return requested


def get_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def query_dshow_devices(ffmpeg_path: Path) -> list[DirectShowDevice]:
    result = subprocess.run(
        [str(ffmpeg_path), "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    devices: list[DirectShowDevice] = []
    current: Optional[DirectShowDevice] = None
    for line in result.stderr.splitlines():
        match = re.search(r'"(?P<name>.+)" \((?P<type>[^)]+)\)', line)
        if match:
            current = DirectShowDevice(
                friendly_name=match.group("name").strip(),
                device_type=match.group("type").strip().lower(),
            )
            if current.device_type == "video":
                devices.append(current)
            else:
                current = None
            continue
        if current is None:
            continue
        alt_match = re.search(r'Alternative name\s+"?(.+?)"?$', line, re.IGNORECASE)
        if alt_match:
            current.alternatives.append(alt_match.group(1).strip())
    return devices


def query_device_modes(ffmpeg_path: Path, device: DirectShowDevice) -> list[DirectShowMode]:
    result = subprocess.run(
        [
            str(ffmpeg_path),
            "-f",
            "dshow",
            "-list_options",
            "true",
            "-i",
            f'video="{device.input_name}"',
        ],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    max_pattern = re.compile(r"max s=(\d+)x(\d+).*?fps=(\d+)")
    any_pattern = re.compile(r"s=(\d+)x(\d+).*?fps=(\d+)")
    seen: set[DirectShowMode] = set()
    for line in result.stderr.splitlines():
        match = max_pattern.search(line) or any_pattern.search(line)
        if not match:
            continue
        seen.add(DirectShowMode(*(int(value) for value in match.groups())))
    return sorted(seen, key=lambda item: (item.width * item.height, item.fps), reverse=True)


def pick_preferred_mode(
    modes: list[DirectShowMode],
    target_width: int,
    target_height: int,
    target_fps: int,
) -> DirectShowMode:
    if not modes:
        return DirectShowMode(width=1280, height=720, fps=30)

    def score(mode: DirectShowMode) -> tuple[int, int, int, int, int]:
        exact_resolution = int(mode.width == target_width and mode.height == target_height)
        within_target = int(mode.width <= target_width and mode.height <= target_height)
        exact_fps = int(mode.fps == target_fps)
        resolution_area = mode.width * mode.height
        fps_gap = -abs(mode.fps - target_fps)
        return exact_resolution, within_target, exact_fps, resolution_area, fps_gap

    return max(modes, key=score)


def choose_device(ffmpeg_path: Path) -> Optional[DirectShowDevice]:
    devices = query_dshow_devices(ffmpeg_path)
    if not devices:
        print("[!] No DirectShow video device detected")
        return None
    if len(devices) == 1:
        print(f"[+] Using video device: {devices[0].friendly_name}")
        return devices[0]
    print("\n--- AVAILABLE VIDEO DEVICES ---")
    for idx, device in enumerate(devices, 1):
        input_note = "" if device.input_name == device.friendly_name else f" (input: {device.input_name})"
        print(f"{idx}. {device.friendly_name}{input_note}")
    choice = input(f"Choose a source (1-{len(devices)}) [1]: ").strip() or "1"
    if choice.isdigit() and 1 <= int(choice) <= len(devices):
        return devices[int(choice) - 1]
    print("[!] Invalid selection, using first detected device")
    return devices[0]


def choose_stream_settings(options: list[DirectShowMode], resolution_label: str | None) -> tuple[int, int, int]:
    if resolution_label in RESOLUTIONS:
        width, height = RESOLUTIONS[resolution_label]
        if options:
            candidate = pick_preferred_mode(options, width, height, 30)
            return width, height, candidate.fps or 30
        return width, height, 30
    if not options:
        print("[!] No mode detected via FFmpeg, using fallback 1280x720@30")
        return 1280, 720, 30
    print("\n--- CAMERA MODES ---")
    for idx, mode in enumerate(options, 1):
        print(f"{idx}. {mode.width}x{mode.height} @ {mode.fps} fps")
    choice = input(f"Choose a mode (1-{len(options)}) [1]: ").strip() or "1"
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        selected = options[int(choice) - 1]
        return selected.width, selected.height, selected.fps
    print("[!] Invalid selection, using first detected mode")
    selected = options[0]
    return selected.width, selected.height, selected.fps


def choose_media_file(media_dir: Path | None, image_dir: Path | None, input_file: str | None) -> Path:
    if input_file:
        path = Path(input_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path
    for directory, allowed in ((media_dir, VIDEO_EXTENSIONS), (image_dir, IMAGE_EXTENSIONS)):
        if directory is None or not directory.exists():
            continue
        for path in sorted(directory.iterdir(), key=lambda entry: entry.name.lower()):
            if path.is_file() and path.suffix.lower() in allowed:
                return path.resolve()
    raise FileNotFoundError("No playable media file found in media/image directories")


def build_encoder_args(encoder: str, fps: int, bitrate_kbps: int) -> list[str]:
    maxrate = int(bitrate_kbps * 1.5)
    bufsize = int(bitrate_kbps * 2)
    target_pix_fmt = "nv12" if encoder in {"h264_qsv", "h264_nvenc"} else "yuv420p"
    args: list[str] = []
    if encoder == "h264_qsv":
        args.extend([
            "-bf", "0", "-async_depth", "4", "-look_ahead", "0",
            "-forced_idr", "1", "-idr_interval", "1", "-repeat_pps", "1", "-aud", "1",
            "-g", "1",
        ])
    elif encoder == "h264_nvenc":
        args.extend(["-preset", "p4", "-tune", "ll", "-bf", "0", "-forced-idr", "1", "-g", "1"])
    else:
        args.extend(["-preset", "veryfast", "-tune", "zerolatency", "-x264-params", "repeat-headers=1:keyint=1:min-keyint=1:scenecut=0"])
    args.extend([
        "-pix_fmt", target_pix_fmt,
        "-b:v", f"{bitrate_kbps}k",
        "-maxrate", f"{maxrate}k",
        "-bufsize", f"{bufsize}k",
        "-bsf:v", "dump_extra",
    ])
    return args


def start_ffmpeg_stream(
    ffmpeg_path: Path,
    input_args: list[str],
    output_width: int,
    output_height: int,
    fps: int,
    encoder: str,
    bitrate_kbps: int,
    port: int,
) -> subprocess.Popen[bytes]:
    uses_intel_qsv = encoder == "h264_qsv"
    vf_parts = [
        f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease",
        f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:black",
        "setsar=1",
        f"fps={fps}",
    ]
    if uses_intel_qsv:
        vf_parts.append("format=nv12")
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    if uses_intel_qsv:
        cmd.extend(["-init_hw_device", "qsv=hw"])
    cmd.extend([
        *input_args,
        "-vf",
        ",".join(vf_parts),
        "-c:v",
        encoder,
        *build_encoder_args(encoder, fps, bitrate_kbps),
        "-an",
        "-f",
        "mpegts",
        "-mpegts_flags",
        "+resend_headers",
        "-listen",
        "1",
        f"http://0.0.0.0:{port}/video_feed",
    ])
    logger.info("Launching FFmpeg bridge: %s", " ".join(cmd))
    return subprocess.Popen(cmd)


def build_camera_input_args(device_input: str, width: int, height: int, fps: int) -> list[str]:
    return [
        "-f",
        "dshow",
        "-framerate",
        str(fps),
        "-video_size",
        f"{width}x{height}",
        "-rtbufsize",
        "150M",
        "-i",
        f"video={device_input}",
    ]


def build_media_input_args(media_path: Path, fps: int) -> list[str]:
    if media_path.suffix.lower() in IMAGE_EXTENSIONS:
        return ["-loop", "1", "-framerate", str(fps), "-i", str(media_path)]
    return ["-stream_loop", "-1", "-re", "-i", str(media_path)]


def run_camera_mode(args: argparse.Namespace) -> int:
    ffmpeg_path = ensure_ffmpeg()
    if ffmpeg_path is None:
        return 1
    device = choose_device(ffmpeg_path)
    if device is None:
        return 1
    options = query_device_modes(ffmpeg_path, device)
    if args.resolution in RESOLUTIONS:
        width, height = RESOLUTIONS[args.resolution]
        capture_fps = pick_preferred_mode(options, width, height, args.fps).fps if options else args.fps
        output_fps = args.fps
    else:
        width, height, output_fps = choose_stream_settings(options, None)
        capture_fps = output_fps
    encoder = resolve_encoder(ffmpeg_path, args.encoder)
    ip = get_ip()
    print("\n" + "=" * 60)
    print("      WINDOWS CAMERA BRIDGE")
    print("=" * 60)
    print(f"\n[+] Stream URL: http://{ip}:{args.port}/video_feed")
    print(f"[+] Device: {device.friendly_name}")
    print(f"[+] Mode: {width}x{height} @ {output_fps} fps")
    print(f"[+] Encoder: {encoder}")
    print(f"[+] WSL command: ./run_app.sh http://{ip}:{args.port}/video_feed")
    print("\n" + "=" * 60)
    ffmpeg_proc = start_ffmpeg_stream(
        ffmpeg_path,
        build_camera_input_args(device.input_name, width, height, capture_fps),
        width,
        height,
        output_fps,
        encoder,
        args.bitrate,
        args.port,
    )
    try:
        ffmpeg_proc.wait()
    except KeyboardInterrupt:
        ffmpeg_proc.terminate()
        ffmpeg_proc.wait()
    return 0


def run_media_mode(args: argparse.Namespace) -> int:
    ffmpeg_path = ensure_ffmpeg()
    if ffmpeg_path is None:
        return 1
    base_dir = Path(__file__).resolve().parent
    media_dir = Path(args.media_dir).expanduser().resolve() if args.media_dir else base_dir / "ref_videos"
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else base_dir / "ref_images"
    media_path = choose_media_file(media_dir, image_dir, args.input_file)
    width, height = RESOLUTIONS[args.resolution]
    encoder = resolve_encoder(ffmpeg_path, args.encoder)
    ip = get_ip()
    print("\n" + "=" * 68)
    print("      WINDOWS MEDIA BRIDGE FOR PEOPLECOUNTER")
    print("=" * 68)
    print(f"\n[+] Stream URL      : http://{ip}:{args.port}/video_feed")
    print(f"[+] Source          : {media_path}")
    print(f"[+] Output format   : {width}x{height} @ {args.fps} fps")
    print(f"[+] Bitrate         : {args.bitrate} kbps")
    print(f"[+] Encoder         : {encoder}")
    print(f"[+] WSL command     : ./run_app.sh http://{ip}:{args.port}/video_feed")
    print("=" * 68)
    ffmpeg_proc = start_ffmpeg_stream(
        ffmpeg_path,
        build_media_input_args(media_path, args.fps),
        width,
        height,
        args.fps,
        encoder,
        args.bitrate,
        args.port,
    )
    try:
        ffmpeg_proc.wait()
    except KeyboardInterrupt:
        ffmpeg_proc.terminate()
        ffmpeg_proc.wait()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    mode = resolve_mode(args)
    if mode == "camera":
        return run_camera_mode(args)
    return run_media_mode(args)


if __name__ == "__main__":
    sys.exit(main())
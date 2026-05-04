from __future__ import annotations

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_BIN_DIR = Path(__file__).resolve().parent.parent / "bin"


def ensure_ffmpeg() -> Path | None:
    FFMPEG_BIN_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = next(FFMPEG_BIN_DIR.rglob("ffmpeg.exe"), None)
    if ffmpeg_path and ffmpeg_path.exists():
        return ffmpeg_path

    archive_path = FFMPEG_BIN_DIR / "ffmpeg.zip"
    print("[i] Downloading FFmpeg...")
    urllib.request.urlretrieve(FFMPEG_DOWNLOAD_URL, archive_path)
    print("[i] Extracting FFmpeg...")
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(FFMPEG_BIN_DIR)
    archive_path.unlink(missing_ok=True)

    ffmpeg_path = next(FFMPEG_BIN_DIR.rglob("ffmpeg.exe"), None)
    if ffmpeg_path is None:
        print("[!] ffmpeg.exe not found after extraction")
        return None
    return ffmpeg_path


def has_encoder(ffmpeg_path: Path, encoder_name: str) -> bool:
    result = subprocess.run(
        [str(ffmpeg_path), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    return encoder_name in output


def resolve_encoder(ffmpeg_path: Path, requested: str) -> str:
    if requested == "auto":
        for candidate in ("h264_qsv", "h264_nvenc", "libx264"):
            if has_encoder(ffmpeg_path, candidate):
                return candidate
        return "libx264"
    if not has_encoder(ffmpeg_path, requested):
        raise RuntimeError(
            f"Requested encoder '{requested}' is not available in FFmpeg at {ffmpeg_path}."
        )
    return requested

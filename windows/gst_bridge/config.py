from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ResolutionPreset(str, Enum):
    UHD_4K = "4K"
    QHD_1440P = "1440p"
    FHD_1080P = "1080p"
    HD_720P = "720p"

    @property
    def size(self) -> tuple[int, int]:
        return {
            ResolutionPreset.UHD_4K: (3840, 2160),
            ResolutionPreset.QHD_1440P: (2560, 1440),
            ResolutionPreset.FHD_1080P: (1920, 1080),
            ResolutionPreset.HD_720P: (1280, 720),
        }[self]


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".webm", ".ts", ".m2ts"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_GSTREAMER_VERSION = "1.28.2"
DEFAULT_GSTREAMER_INSTALLER_URL = (
    "https://gstreamer.freedesktop.org/data/pkg/windows/1.28.2/msvc/"
    "gstreamer-1.0-msvc-x86_64-1.28.2.exe"
)
DEFAULT_MEDIAMTX_VERSION = "1.11.3"
DEFAULT_MEDIAMTX_DOWNLOAD_URL = (
    "https://github.com/bluenviron/mediamtx/releases/download/v1.11.3/"
    "mediamtx_v1.11.3_windows_amd64.zip"
)

REQUIRED_GST_PLUGINS = [
    "filesrc",
    "decodebin",
    "videotestsrc",
    "videoconvert",
    "videoscale",
    "videorate",
    "capsfilter",
    "queue",
    "tee",
    "fpsdisplaysink",
    "h264parse",
    "flvmux",
    "rtmpsink",
    "qsvh264enc",
]


@dataclass(frozen=True)
class BridgeConfig:
    base_dir: Path
    media_dir: Path
    image_dir: Path
    width: int
    height: int
    fps: int
    bitrate_kbps: int
    rtsp_port: int
    rtmp_port: int
    rtsp_path: str
    standby_path: str
    profile: bool
    profile_seconds: float | None
    profile_dir: Path | None
    third_party_dir: Path

    @property
    def rtsp_url(self) -> str:
        return f"rtsp://127.0.0.1:{self.rtsp_port}/{self.rtsp_path}"

    @property
    def rtmp_publish_url(self) -> str:
        return f"rtmp://127.0.0.1:{self.rtmp_port}/{self.rtsp_path}"

    @property
    def standby_rtmp_publish_url(self) -> str:
        return f"rtmp://127.0.0.1:{self.rtmp_port}/{self.standby_path}"

    @property
    def profile_output_dir(self) -> Path | None:
        if not self.profile:
            return None
        return self.profile_dir or (self.base_dir / "profiles")

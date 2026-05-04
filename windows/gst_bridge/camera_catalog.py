from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .models import MediaItem


def list_camera_items(ffmpeg_path: Path, target_width: int, target_height: int, target_fps: int) -> list[MediaItem]:
    devices = _query_dshow_devices(ffmpeg_path)
    items: list[MediaItem] = []
    for device_name in devices:
        width, height, fps = _pick_preferred_mode(
            _query_device_modes(ffmpeg_path, device_name),
            target_width,
            target_height,
            target_fps,
        )
        items.append(
            MediaItem(
                source_id=f"camera:{device_name.lower()}",
                name=device_name,
                kind="camera",
                path=None,
                source_spec=_build_dshow_video_spec(device_name),
                thumbnail_path=None,
                details=f"camera - {width}x{height} @ {fps} fps",
            )
        )
    return items


def _build_dshow_video_spec(device_name: str) -> str:
    return f"video={device_name}"


def _query_dshow_devices(ffmpeg_path: Path) -> list[str]:
    result = subprocess.run(
        [str(ffmpeg_path), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
        check=False,
    )
    devices: list[str] = []
    for line in result.stderr.splitlines():
        match = re.search(r'"(?P<name>.+)" \((?P<type>[^)]+)\)', line)
        if match and match.group("type").strip().lower() == "video":
            devices.append(match.group("name").strip())
    return devices


def _query_device_modes(ffmpeg_path: Path, device_name: str) -> list[tuple[int, int, int]]:
    result = subprocess.run(
        [
            str(ffmpeg_path),
            "-hide_banner",
            "-f",
            "dshow",
            "-list_options",
            "true",
            "-i",
            _build_dshow_video_spec(device_name),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    patterns = [
        re.compile(r"max s=(\d+)x(\d+).*?fps=(\d+)"),
        re.compile(r"s=(\d+)x(\d+).*?fps=(\d+)"),
    ]
    modes: set[tuple[int, int, int]] = set()
    for line in result.stderr.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                modes.add(tuple(int(value) for value in match.groups()))
                break
    return sorted(modes, key=lambda mode: (mode[0] * mode[1], mode[2]), reverse=True)


def _pick_preferred_mode(
    modes: list[tuple[int, int, int]],
    target_width: int,
    target_height: int,
    target_fps: int,
) -> tuple[int, int, int]:
    if not modes:
        return target_width, target_height, target_fps

    def score(mode: tuple[int, int, int]) -> tuple[int, int, int, int, int]:
        width, height, fps = mode
        exact_resolution = int(width == target_width and height == target_height)
        within_target = int(width <= target_width and height <= target_height)
        exact_fps = int(fps == target_fps)
        resolution_area = width * height
        fps_gap = -abs(fps - target_fps)
        return exact_resolution, within_target, exact_fps, resolution_area, fps_gap

    return max(modes, key=score)

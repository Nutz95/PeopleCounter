from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .directshow_device import DirectShowDevice
from .directshow_mode import DirectShowMode

DEVICE_PATTERN = re.compile(r'"(?P<name>.+)" \((?P<type>[^)]+)\)')
ALTERNATIVE_NAME_PATTERN = re.compile(r'Alternative name\s+"?(.+?)"?$', re.IGNORECASE)


def query_dshow_devices(ffmpeg_path: Path) -> list[DirectShowDevice]:
    result = subprocess.run(
        [str(ffmpeg_path), "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    devices: list[DirectShowDevice] = []
    current: DirectShowDevice | None = None
    for line in result.stderr.splitlines():
        match = DEVICE_PATTERN.search(line)
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
        alt_match = ALTERNATIVE_NAME_PATTERN.search(line)
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
            f"video={device.input_name}",
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
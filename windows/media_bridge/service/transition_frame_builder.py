from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


class TransitionFrameBuilder:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir

    def build(self, width: int, height: int, message: str) -> Path:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.runtime_dir / "transition_frame.png"
        image = Image.new("RGB", (width, height), color="#10131a")
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((64, 64, width - 64, height - 64), outline="#f59e0b", width=8)
        title = "PeopleCounter Media Bridge"
        subtitle = message
        drawer.text((width * 0.5 - 280, height * 0.5 - 90), title, fill="#f2f4f8")
        drawer.text((width * 0.5 - 180, height * 0.5 + 10), subtitle, fill="#ffd58a")
        image.save(image_path)
        return image_path

    def build_segment(self, ffmpeg_path: Path, width: int, height: int, fps: int, message: str) -> Path:
        image_path = self.build(width=width, height=height, message=message)
        segment_path = self.runtime_dir / "transition_segment.mp4"
        command = [
            str(ffmpeg_path),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(image_path),
            "-t",
            "2",
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-g",
            str(fps * 2),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(segment_path),
        ]
        subprocess.run(command, check=True)
        return segment_path
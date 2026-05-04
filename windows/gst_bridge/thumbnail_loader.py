from __future__ import annotations

import io
import subprocess
from pathlib import Path

from .models import MediaItem

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency already declared in runtime requirements
    Image = None

THUMBNAIL_SIZE = (320, 180)


def thumbnails_enabled() -> bool:
    return Image is not None


def load_thumbnail_image(item: MediaItem, ffmpeg_path: Path, size: tuple[int, int] = THUMBNAIL_SIZE):
    if Image is None:
        return None
    if item.kind == "image" and item.path is not None:
        image = Image.open(item.path).convert("RGB")
    elif item.kind == "video" and item.path is not None:
        result = subprocess.run(
            [
                str(ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(item.path),
                "-vframes",
                "1",
                "-vf",
                f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=decrease,pad={size[0]}:{size[1]}:(ow-iw)/2:(oh-ih)/2:color=black",
                "-f",
                "image2pipe",
                "-vcodec",
                "bmp",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        if not result.stdout:
            return None
        image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    else:
        return None
    image.thumbnail(size, Image.LANCZOS)
    if image.size != size:
        canvas = Image.new("RGB", size, (11, 18, 28))
        offset = ((size[0] - image.size[0]) // 2, (size[1] - image.size[1]) // 2)
        canvas.paste(image, offset)
        image = canvas
    return image

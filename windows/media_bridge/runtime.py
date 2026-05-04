from __future__ import annotations

from pathlib import Path


def default_media_dir(base_dir: Path) -> Path:
    return base_dir / "ref_videos"


def default_image_dir(base_dir: Path) -> Path:
    return base_dir / "ref_images"

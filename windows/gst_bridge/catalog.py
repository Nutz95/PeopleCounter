from __future__ import annotations

from pathlib import Path

from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .models import MediaItem


class MediaCatalog:
    def __init__(self, media_dir: Path, image_dir: Path) -> None:
        self._media_dir = media_dir
        self._image_dir = image_dir

    def list_items(self) -> list[MediaItem]:
        items: list[MediaItem] = []
        items.extend(self._scan_directory(self._media_dir, VIDEO_EXTENSIONS, "video"))
        items.extend(self._scan_directory(self._image_dir, IMAGE_EXTENSIONS, "image"))
        return items

    def _scan_directory(self, directory: Path, allowed_extensions: set[str], kind: str) -> list[MediaItem]:
        if not directory.exists():
            return []
        return [
            MediaItem(
                source_id=f"{kind}:{path.as_posix().lower()}",
                name=path.name,
                kind=kind,
                path=path.resolve(),
                details=f"{kind} - {path.resolve()}",
            )
            for path in sorted(directory.iterdir(), key=lambda entry: entry.name.lower())
            if path.is_file() and path.suffix.lower() in allowed_extensions
        ]

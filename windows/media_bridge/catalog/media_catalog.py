from __future__ import annotations

from pathlib import Path

from ..configuration.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from ..dshow.directshow_device import DirectShowDevice
from ..dshow.directshow_discovery import pick_preferred_mode, query_device_modes, query_dshow_devices
from ..models.media_item import MediaItem


class MediaCatalog:
    def __init__(
        self,
        media_dir: Path,
        image_dir: Path,
        ffmpeg_path: Path,
        output_width: int,
        output_height: int,
        target_fps: int,
        reference_image: Path | None = None,
        initial_source: Path | None = None,
        include_cameras: bool = False,
    ) -> None:
        self.media_dir = media_dir
        self.image_dir = image_dir
        self.ffmpeg_path = ffmpeg_path
        self.output_width = output_width
        self.output_height = output_height
        self.target_fps = target_fps
        self.reference_image = reference_image
        self.initial_source = initial_source
        self.include_cameras = include_cameras

    def list_items(self) -> list[MediaItem]:
        items: list[MediaItem] = []
        items.extend(self._build_file_items(self.media_dir, VIDEO_EXTENSIONS, "video"))
        items.extend(self._build_file_items(self.image_dir, IMAGE_EXTENSIONS, "image"))
        if self.include_cameras:
            items.extend(self._build_camera_items())
        if self.reference_image is not None and self.reference_image.exists():
            items.insert(0, self._build_reference_item(self.reference_image))
        if self.initial_source is not None and self.initial_source.exists():
            resolved = self.initial_source.resolve()
            if all(item.path != resolved for item in items):
                kind = "image" if resolved.suffix.lower() in IMAGE_EXTENSIONS else "video"
                items.insert(0, self._build_file_item(resolved, kind))
        return items

    def resolve_initial_item_id(self, items: list[MediaItem]) -> str | None:
        if self.initial_source is not None:
            resolved = self.initial_source.resolve()
            for item in items:
                if item.path == resolved:
                    return item.source_id
        if self.reference_image is not None:
            resolved = self.reference_image.resolve()
            for item in items:
                if item.path == resolved:
                    return item.source_id
        return items[0].source_id if items else None

    def _build_file_items(self, directory: Path, allowed_extensions: set[str], kind: str) -> list[MediaItem]:
        if not directory.exists():
            return []
        items: list[MediaItem] = []
        for path in sorted(directory.iterdir(), key=lambda entry: entry.name.lower()):
            if not path.is_file() or path.suffix.lower() not in allowed_extensions:
                continue
            items.append(self._build_file_item(path.resolve(), kind))
        return items

    def _build_file_item(self, path: Path, kind: str) -> MediaItem:
        if kind == "image":
            input_args = ("-loop", "1", "-framerate", str(self.target_fps), "-i", str(path))
        else:
            input_args = ("-stream_loop", "-1", "-re", "-i", str(path))
        return MediaItem(
            source_id=f"{kind}:{path.as_posix().lower()}",
            name=path.name,
            kind=kind,
            details=f"{kind} - {path}",
            input_args=input_args,
            path=path,
            thumbnail_path=path,
        )

    def _build_reference_item(self, path: Path) -> MediaItem:
        return MediaItem(
            source_id=f"reference:{path.as_posix().lower()}",
            name=path.name,
            kind="reference",
            details=f"reference image - {path}",
            input_args=("-loop", "1", "-framerate", str(self.target_fps), "-i", str(path)),
            path=path,
            thumbnail_path=path,
        )

    def _build_camera_items(self) -> list[MediaItem]:
        devices = query_dshow_devices(self.ffmpeg_path)
        items: list[MediaItem] = []
        for device in devices:
            items.append(self._build_camera_item(device))
        return items

    def _build_camera_item(self, device: DirectShowDevice) -> MediaItem:
        modes = query_device_modes(self.ffmpeg_path, device)
        selected_mode = pick_preferred_mode(
            modes,
            target_width=self.output_width,
            target_height=self.output_height,
            target_fps=self.target_fps,
        )
        preview_modes = ", ".join(
            f"{mode.width}x{mode.height}@{mode.fps}" for mode in modes[:3]
        ) or "mode auto"
        input_args = (
            "-thread_queue_size",
            "2048",
            "-f",
            "dshow",
            "-rtbufsize",
            "512M",
            "-framerate",
            str(selected_mode.fps),
            "-video_size",
            f"{selected_mode.width}x{selected_mode.height}",
            "-i",
            f'video="{device.input_name}"',
        )
        return MediaItem(
            source_id=f"camera:{device.input_name.lower()}",
            name=device.friendly_name,
            kind="camera",
            details=f"camera - {selected_mode.width}x{selected_mode.height}@{selected_mode.fps} - {preview_modes}",
            input_args=input_args,
        )
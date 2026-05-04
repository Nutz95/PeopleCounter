from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MediaItem:
    source_id: str
    name: str
    kind: str
    path: Path
    details: str


@dataclass
class RuntimeMetrics:
    selected_source: str = "none"
    pipeline_state: str = "idle"
    decoder_fps: float = 0.0
    output_fps: float = 0.0
    dropped_fps: float = 0.0
    queue_fill: str = "n/a"
    warnings: int = 0
    errors: int = 0
    last_log: str = ""
    gst_version: str = "unknown"
    encoder: str = "qsvh264enc"
    publisher: str = "mediamtx (rtmp ingest)"
    last_profile_csv: str = ""
    last_profile_png: str = ""

    def as_lines(self) -> list[str]:
        return [
            f"source          : {self.selected_source}",
            f"pipeline state  : {self.pipeline_state}",
            f"decode fps      : {self.decoder_fps:5.2f}",
            f"output fps      : {self.output_fps:5.2f}",
            f"drop fps        : {self.dropped_fps:5.2f}",
            f"queue fill      : {self.queue_fill}",
            f"gst version     : {self.gst_version}",
            f"encoder         : {self.encoder}",
            f"publisher       : {self.publisher}",
            f"warnings/errors : {self.warnings}/{self.errors}",
            f"last log        : {self.last_log[:110]}",
        ]


@dataclass
class ProfileSample:
    t_s: float
    decoder_fps: float
    output_fps: float
    dropped_fps: float
    warnings: int
    errors: int


@dataclass
class ValidationResult:
    gstreamer_root: Path
    gst_launch_path: Path
    gst_inspect_path: Path
    mediamtx_path: Path
    gst_version: str
    plugins: list[str] = field(default_factory=list)

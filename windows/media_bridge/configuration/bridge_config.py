from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BridgeConfig:
    width: int
    height: int
    fps: int
    bitrate_kbps: int
    encoder: str
    port: int
    media_dir: Path
    image_dir: Path
    initial_source: Path | None = None
    reference_image: Path | None = None
    loop_forever: bool = True
    show_ui: bool = True
    interstitial_seconds: float = 0.0
    camera_warmup_seconds: float = 2.5
    rtsp_path: str = "live"
    zmq_port: int = 5555

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}@{self.fps}"
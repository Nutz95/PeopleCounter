from __future__ import annotations

import shutil
from pathlib import Path

from ..config import BridgeConfig


def find_ffmpeg(base_dir: Path) -> Path:
    search_root = base_dir / "bin"
    if search_root.exists():
        found = next(search_root.rglob("ffmpeg.exe"), None)
        if found is not None:
            return found
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return Path(ffmpeg_path)
    raise RuntimeError(
        f"ffmpeg.exe not found under {search_root} and not available in PATH. "
        "The persistent Windows bridge requires a local FFmpeg binary."
    )


def build_black_frame(width: int, height: int) -> bytes:
    luma_size = width * height
    chroma_size = luma_size // 2
    return bytes([16]) * luma_size + bytes([128]) * chroma_size


def build_scale_filter(config: BridgeConfig) -> str:
    target_aspect = config.width / config.height
    return (
        "scale="
        f"w='if(gt(dar,{target_aspect}),{config.width},trunc({config.height}*dar/2)*2)':"
        f"h='if(gt(dar,{target_aspect}),trunc({config.width}/dar/2)*2,{config.height})',"
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={config.fps},setsar=1,format=nv12"
    )

from __future__ import annotations

from pathlib import Path

from .config import BridgeConfig
from .models import MediaItem


class PipelineBuilder:
    def __init__(self, config: BridgeConfig) -> None:
        self._config = config

    def build(self, item: MediaItem) -> list[str]:
        args: list[str] = ["-e", "-m"]

        args.extend(["uridecodebin", f"uri={item.path.resolve().as_uri()}"])

        if item.kind == "image":
            args.extend(["!", "imagefreeze", "is-live=true"])

        args.extend(
            [
                "!",
                "queue",
                "!",
                "videoconvert",
                "!",
                "videoscale",
                "!",
                "videorate",
                "!",
                self._raw_caps,
                "!",
                "tee",
                "name=t",
                "t.",
                "!",
                "queue",
                "leaky=downstream",
                "max-size-buffers=2",
                "!",
                "fpsdisplaysink",
                "fps-update-interval=1000",
                "text-overlay=false",
                "signal-fps-measurements=false",
                "video-sink=fakesink",
                "sync=false",
                "t.",
                "!",
                "queue",
                "!",
                "qsvh264enc",
                f"bitrate={self._config.bitrate_kbps}",
                f"gop-size={self._config.fps}",
                "!",
                "h264parse",
                "config-interval=-1",
                "!",
                "video/x-h264,stream-format=avc",
                "!",
                "flvmux",
                "streamable=true",
                "!",
                "rtmpsink",
                f"location={self._config.rtmp_publish_url}",
            ]
        )
        return args

    def build_standby(self) -> list[str]:
        return [
            "-e",
            "-m",
            "videotestsrc",
            "is-live=true",
            "pattern=black",
            "do-timestamp=true",
            "!",
            "queue",
            "!",
            "videoconvert",
            "!",
            "videoscale",
            "!",
            "videorate",
            "!",
            self._raw_caps,
            "!",
            "qsvh264enc",
            f"bitrate={self._config.bitrate_kbps}",
            f"gop-size={self._config.fps}",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "video/x-h264,stream-format=avc",
            "!",
            "flvmux",
            "streamable=true",
            "!",
            "queue",
            "!",
            "rtmpsink",
            f"location={self._config.standby_rtmp_publish_url}",
        ]

    @property
    def _raw_caps(self) -> str:
        return (
            f"video/x-raw,format=NV12,width={self._config.width},height={self._config.height},"
            f"framerate={self._config.fps}/1"
        )

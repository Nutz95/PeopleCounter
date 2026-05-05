from __future__ import annotations

from pathlib import Path

from windows.gst_bridge.config import BridgeConfig
from windows.gst_bridge.persistent.stream_encoder import StreamEncoder


def _make_config() -> BridgeConfig:
    base_dir = Path("/tmp/peoplecounter")
    return BridgeConfig(
        base_dir=base_dir,
        media_dir=base_dir / "media",
        image_dir=base_dir / "images",
        width=3840,
        height=2160,
        fps=30,
        bitrate_kbps=50000,
        rtsp_port=5002,
        rtmp_port=1935,
        rtsp_path="live",
        standby_path="standby",
        profile=False,
        profile_seconds=None,
        profile_dir=None,
        third_party_dir=base_dir / "third_party",
    )


def test_stream_encoder_build_command_uses_decoder_friendly_h264_settings() -> None:
    config = _make_config()
    encoder = StreamEncoder(Path("ffmpeg.exe"), config)

    cmd = encoder._build_command()

    assert "-forced_idr" in cmd
    assert cmd[cmd.index("-forced_idr") + 1] == "1"
    assert "-idr_interval" in cmd
    assert cmd[cmd.index("-idr_interval") + 1] == "1"
    assert "-repeat_pps" in cmd
    assert cmd[cmd.index("-repeat_pps") + 1] == "1"
    assert "-aud" in cmd
    assert cmd[cmd.index("-aud") + 1] == "1"
    assert "-skip_frame" not in cmd
    assert cmd[cmd.index("-g") + 1] == "30"
    assert cmd[cmd.index("-keyint_min") + 1] == "30"
    assert cmd[cmd.index("-f") + 1] == "rawvideo"
    assert cmd[-2] == "flv"
    assert cmd[-1] == config.rtmp_publish_url

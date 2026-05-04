from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bootstrap import ensure_gstreamer, ensure_mediamtx
from .config import BridgeConfig, ResolutionPreset
from .service import GstBridgeService
from .ui import GstBridgeApp
from .validation import validate_runtime


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="PeopleCounter GStreamer + MediaMTX bridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--media-dir", default=str(base_dir / "ref_videos"))
    parser.add_argument("--image-dir", default=str(base_dir / "ref_images"))
    parser.add_argument("--resolution", choices=[preset.value for preset in ResolutionPreset], default=ResolutionPreset.UHD_4K.value)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=50000, help="Target bitrate in kbps.")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--rtmp-port", type=int, default=1935)
    parser.add_argument("--rtsp-path", default="live")
    parser.add_argument("--standby-path", default="standby")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-seconds", type=float)
    parser.add_argument("--profile-dir")
    parser.add_argument("--third-party-dir", default=str(base_dir / "third_party"))
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BridgeConfig:
    preset = ResolutionPreset(args.resolution)
    width, height = preset.size
    base_dir = Path(__file__).resolve().parent.parent
    return BridgeConfig(
        base_dir=base_dir,
        media_dir=Path(args.media_dir).expanduser().resolve(),
        image_dir=Path(args.image_dir).expanduser().resolve(),
        width=width,
        height=height,
        fps=args.fps,
        bitrate_kbps=args.bitrate,
        rtsp_port=args.port,
        rtmp_port=args.rtmp_port,
        rtsp_path=args.rtsp_path,
        standby_path=args.standby_path,
        profile=args.profile,
        profile_seconds=args.profile_seconds,
        profile_dir=Path(args.profile_dir).expanduser().resolve() if args.profile_dir else None,
        third_party_dir=Path(args.third_party_dir).expanduser().resolve(),
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)

    print("=" * 64)
    print("  PeopleCounter GStreamer + MediaMTX bridge")
    print("=" * 64)

    print("\n[1/4] Ensuring local dependencies …")
    gstreamer_root = ensure_gstreamer(config.third_party_dir)
    mediamtx_path = ensure_mediamtx(config.third_party_dir)

    print("[2/4] Validating local runtime …")
    validation = validate_runtime(gstreamer_root, mediamtx_path)
    print(f"      {validation.gst_version}")
    print(f"      encoder plugin: qsvh264enc")

    print("[3/4] Starting services …")
    service = GstBridgeService(config, validation)
    service.start()

    print("[4/4] Opening UI …")
    print(f"      Stream URL: {service.get_stream_url()}")

    try:
        app = GstBridgeApp(service, auto_close_seconds=config.profile_seconds)
        app.run()
    except Exception as exc:
        service.stop()
        print(f"[!] Bridge failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

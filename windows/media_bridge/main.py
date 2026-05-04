from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import BridgeConfig
from .runtime import default_image_dir, default_media_dir
from .service import MediaBridgeService
from .ui import MediaBridgeApp


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Static media bridge demo for PeopleCounter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", help="Initial media file to highlight when the GUI starts.")
    parser.add_argument("--media-dir", help="Directory containing playable videos.")
    parser.add_argument("--image-dir", help="Directory containing playable images.")
    parser.add_argument("--resolution", choices=["4K", "1440p", "1080p", "720p"], default="4K")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=50000, help="Target bitrate in kbps.")
    parser.add_argument("--encoder", default="auto", help="FFmpeg video encoder (auto, h264_qsv, h264_nvenc, libx264).")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--rtsp-path", default="live", help="RTSP publication path.")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ control port used by FFmpeg.")
    parser.add_argument(
        "--reference-image",
        help="Optional reference image displayed first if present.",
    )
    parser.set_defaults(base_dir=base_dir)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BridgeConfig:
    base_dir: Path = args.base_dir
    media_dir = Path(args.media_dir).expanduser().resolve() if args.media_dir else default_media_dir(base_dir)
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else default_image_dir(base_dir)
    input_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
    reference_image = Path(args.reference_image).expanduser().resolve() if args.reference_image else None
    width, height = {
        "4K": (3840, 2160),
        "1440p": (2560, 1440),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
    }[args.resolution]
    return BridgeConfig(
        width=width,
        height=height,
        fps=args.fps,
        bitrate_kbps=args.bitrate,
        encoder=args.encoder,
        port=args.port,
        media_dir=media_dir,
        image_dir=image_dir,
        initial_source=input_file,
        reference_image=reference_image,
        show_ui=True,
        rtsp_path=args.rtsp_path,
        zmq_port=args.zmq_port,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    service = MediaBridgeService(config)
    service.start()
    try:
        app = MediaBridgeApp(service)
        app.run()
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

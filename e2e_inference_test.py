#!/usr/bin/env python3
"""End-to-end visual inference test for PeopleCounter.

Streams a calibration image (or video) through the full NVDEC → CUDA preprocess
→ YOLO (global and/or tiling) pipeline, then saves an annotated PNG showing
the decoded frame with bounding boxes and segmentation masks overlaid.

Usage (inside the Docker container):
    python e2e_inference_test.py --image /workspace/people_walking.jpg
    python e2e_inference_test.py --image /workspace/people_walking.jpg --model both --resolution 1080p
    python e2e_inference_test.py --image /workspace/people_walking.jpg --frames 3 --bitrate 20000
    python e2e_inference_test.py --video /workspace/clip.mp4 --output /workspace/result.png

Results are written to:
  * <output>.png  — annotated frame (bboxes + seg mask)
  * <output>.json — detection results + metrics

The script is self-contained: it starts an FFmpeg HTTP server locally on a random
free port, feeds it to the pipeline, and tears everything down when done.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "4K":    (3840, 2160),
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
    "720p":  (1280,  720),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2E inference test: stream image/video → NVDEC → YOLO → annotated PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", metavar="FILE", help="Image file to loop as a stream.")
    src.add_argument("--video", metavar="FILE", help="Video file to loop.")
    p.add_argument(
        "--model",
        choices=["global", "tiles", "both"],
        default="both",
        help="Which YOLO model(s) to run.",
    )
    p.add_argument(
        "--resolution", "-r",
        choices=list(RESOLUTIONS),
        default=None,
        help="Target resolution for the stream (scale + letterbox).",
    )
    p.add_argument("--fps",      type=int, default=30, help="Stream framerate.")
    p.add_argument("--bitrate",  type=int, default=15000, metavar="KBPS", help="H.264 bitrate in kbps.")
    p.add_argument("--frames",   type=int, default=1,  help="Number of frames to process.")
    p.add_argument(
        "--output", "-o",
        default="/app/app_v2/tests/integration/pipeline/artifacts/e2e_result",
        metavar="PATH",
        help="Output path prefix (no extension). .png and .json will be appended. "
             "Inside Docker the workspace is mounted at /app.",
    )
    p.add_argument(
        "--conf", type=float, default=0.25, help="YOLO confidence threshold."
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Mini HTTP broadcaster (same logic as camera_bridge.py, self-contained)
# ──────────────────────────────────────────────────────────────────────────────

_TS_PACKET_SIZE = 188
_TS_READ_SIZE   = _TS_PACKET_SIZE * 348  # ~64 KB per read


class _StreamQueue:
    def __init__(self) -> None:
        import queue
        self.q: queue.Queue[bytes | None] = queue.Queue(maxsize=256)


class _Broadcaster:
    def __init__(self) -> None:
        self._clients: list[_StreamQueue] = []
        self._lock = threading.Lock()

    def register(self) -> _StreamQueue:
        s = _StreamQueue()
        with self._lock:
            self._clients.append(s)
        return s

    def unregister(self, s: _StreamQueue) -> None:
        with self._lock:
            try:
                self._clients.remove(s)
            except ValueError:
                pass

    def broadcast(self, chunk: bytes) -> None:
        with self._lock:
            clients = list(self._clients)
        for s in clients:
            try:
                s.q.put_nowait(chunk)
            except Exception:
                pass

    def stop_all(self) -> None:
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for s in clients:
            try:
                s.q.put_nowait(None)
            except Exception:
                pass


_broadcaster: _Broadcaster = _Broadcaster()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # silence access log
        pass

    def do_GET(self) -> None:
        if self.path != "/video_feed":
            self.send_response(404); self.end_headers(); return
        stream = _broadcaster.register()
        self.send_response(200)
        self.send_header("Content-Type",  "video/mp2t")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection",    "close")
        self.end_headers()
        try:
            while True:
                chunk = stream.q.get()
                if chunk is None:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except OSError:
                    break
        finally:
            _broadcaster.unregister(stream)


# ──────────────────────────────────────────────────────────────────────────────
# FFmpeg helpers
# ──────────────────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_ffmpeg_cmd(
    input_file: str,
    is_image: bool,
    fps: int,
    out_w: int,
    out_h: int,
    bitrate_kbps: int,
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg]
    if is_image:
        cmd += ["-loop", "1", "-framerate", str(fps), "-i", input_file]
    else:
        cmd += ["-stream_loop", "-1", "-re", "-i", input_file]

    vf: list[str] = []
    if out_w and out_h:
        vf.append(f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease")
        vf.append(f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black")
        vf.append("setsar=1")
    if not is_image:
        vf.append(f"fps={fps}")
    if vf:
        cmd += ["-vf", ",".join(vf)]

    # Prefer hardware encode if available, else libx264
    enc = "libx264"
    for candidate in ("h264_nvenc", "h264_qsv", "libx264"):
        try:
            r = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=False, timeout=5,
            )
            if candidate in (r.stdout + r.stderr):
                enc = candidate
                break
        except Exception:
            pass

    if enc == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-x264-params", "repeat-headers=1"]
    else:
        cmd += ["-c:v", enc]
        if enc.startswith("h264_nvenc"):
            cmd += ["-rc", "vbr_hq", "-preset", "llhq", "-bf", "0"]
        if enc.startswith("h264_qsv"):
            cmd += ["-bf", "0", "-async_depth", "4"]

    maxrate = int(bitrate_kbps * 1.5)
    bufsize = int(bitrate_kbps * 2)
    cmd += [
        "-b:v", f"{bitrate_kbps}k",
        "-maxrate", f"{maxrate}k",
        "-bufsize", f"{bufsize}k",
        "-g", str(fps * 2),
        "-keyint_min", str(fps * 2),
        "-an",
        "-f", "mpegts",
        "-mpegts_flags", "+resend_headers",
        "-",
    ]
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Annotation helpers (PIL / numpy)
# ──────────────────────────────────────────────────────────────────────────────

def _annotate_frame(
    frame_rgb: Any,        # HxWx3 numpy uint8
    results: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
) -> Any:
    """Draw bboxes and seg masks onto *frame_rgb* and return annotated PIL Image."""
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError:
        print("[WARN] PIL/numpy not available — skipping annotation.")
        return None

    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img, "RGBA")

    for r in results:
        model_name = r.get("model", "?")
        seg_raw  = r.get("seg_mask_raw")
        seg_w    = r.get("seg_mask_w", 0)
        seg_h    = r.get("seg_mask_h", 0)

        # ── Segmentation mask ──────────────────────────────────────────────
        if seg_raw and seg_w and seg_h:
            mask_bytes = base64.b64decode(seg_raw)
            # mask_bytes is H×W uint8 (0 = bg, 255 = person) in YOLO letterbox space
            import numpy as np  # type: ignore[import]
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(seg_h, seg_w)

            # Remove letterbox padding to get content region
            model_sz = 640
            m_scale  = min(model_sz / frame_w, model_sz / frame_h)
            prep_w   = int(frame_w * m_scale)
            prep_h   = int(frame_h * m_scale)
            pad_x    = (model_sz - prep_w) // 2
            pad_y    = (model_sz - prep_h) // 2
            mpx = int(pad_x * seg_w / model_sz)
            mpy = int(pad_y * seg_h / model_sz)
            mcw = int(prep_w * seg_w / model_sz)
            mch = int(prep_h * seg_h / model_sz)
            # Crop and resize to frame
            cropped = mask[mpy:mpy + mch, mpx:mpx + mcw]
            seg_img = Image.fromarray(cropped).resize((frame_w, frame_h), Image.NEAREST)
            seg_arr = np.array(seg_img)
            # Create RGBA overlay: teal where person
            overlay = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
            overlay[seg_arr > 128] = [42, 223, 165, 90]  # teal, 35% alpha
            img = Image.alpha_composite(img.convert("RGBA"), Image.fromarray(overlay))
            draw = ImageDraw.Draw(img, "RGBA")

        # ── Bounding boxes ─────────────────────────────────────────────────
        for det in r.get("detections", []):
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            gx1, gy1, gx2, gy2 = bbox
            bx1 = int(gx1 * frame_w)
            by1 = int(gy1 * frame_h)
            bx2 = int(gx2 * frame_w)
            by2 = int(gy2 * frame_h)
            color = (42, 223, 165, 220)
            draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
            label = f"{model_name} {det.get('conf', 0):.0%}"
            draw.text((bx1 + 3, by1 - 15), label, fill=(230, 255, 250, 220))

    return img.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    input_file = args.image or args.video
    is_image   = args.image is not None

    # ── Validate input ─────────────────────────────────────────────────────
    if not Path(input_file).exists():
        print(f"[ERROR] Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # ── Build output resolution ────────────────────────────────────────────
    out_w, out_h = 0, 0
    if args.resolution:
        out_w, out_h = RESOLUTIONS[args.resolution]

    # ── Start local HTTP server ────────────────────────────────────────────
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stream_url = f"http://127.0.0.1:{port}/video_feed"
    print(f"[INFO] Local stream: {stream_url}")

    # ── Start FFmpeg ───────────────────────────────────────────────────────
    cmd = _build_ffmpeg_cmd(input_file, is_image, args.fps, out_w, out_h, args.bitrate)
    print(f"[INFO] FFmpeg: {' '.join(cmd[:8])}…")
    ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _pipe():
        assert ffmpeg_proc.stdout
        try:
            while True:
                chunk = ffmpeg_proc.stdout.read(_TS_READ_SIZE)
                if not chunk:
                    break
                _broadcaster.broadcast(chunk)
        finally:
            _broadcaster.stop_all()

    threading.Thread(target=_pipe, daemon=True, name="ffmpeg-pipe").start()
    threading.Thread(
        target=lambda: [l for l in ffmpeg_proc.stderr or []],
        daemon=True, name="ffmpeg-stderr",
    ).start()

    # Wait briefly for FFmpeg to start and HTTP server to get first bytes
    time.sleep(2.0)

    # ── Import pipeline components (must be available in Docker env) ───────
    try:
        from app_v2.infrastructure.nvdec_decoder import NvdecDecoder
        from app_v2.infrastructure.cuda_preprocessor import CudaPreprocessor
        from app_v2.application.model_builder import ModelBuilder
        from app_v2.application.inference_stream_controller import InferenceStreamController
        from app_v2.config import load_pipeline_config
        from app_v2.infrastructure.stream_pool import SimpleStreamPool
        from app_v2.infrastructure.rtsp_frame_source import RTSPFrameSource
    except ImportError as exc:
        print(f"[ERROR] Pipeline components not importable: {exc}", file=sys.stderr)
        ffmpeg_proc.terminate(); server.shutdown(); sys.exit(1)

    # ── Build models (only the ones requested) ─────────────────────────────
    config     = load_pipeline_config()
    stream_pool = SimpleStreamPool()
    inf_ctrl    = InferenceStreamController(config)
    model_bldr  = ModelBuilder(config, inf_ctrl, stream_pool)
    all_models  = model_bldr.build_models()

    def _model_filter(m: Any) -> bool:
        name = getattr(m, "name", "")
        if args.model == "global":
            return "global" in name
        if args.model == "tiles":
            return "tile" in name
        return True  # "both"

    models = [m for m in all_models if _model_filter(m)]
    if not models:
        print(f"[WARN] No models matched filter '{args.model}'; running all.")
        models = all_models
    print(f"[INFO] Running models: {[m.name for m in models]}")

    # ── Connect frame source ───────────────────────────────────────────────
    frame_source = RTSPFrameSource(stream_url=stream_url)
    try:
        frame_source.connect()
    except Exception as exc:
        print(f"[ERROR] Could not connect to stream: {exc}", file=sys.stderr)
        ffmpeg_proc.terminate(); server.shutdown(); sys.exit(1)

    preprocessor = CudaPreprocessor()
    preprocessor.configure(config)

    # ── Process N frames ───────────────────────────────────────────────────
    collected_results: list[dict[str, Any]] = []
    last_frame_rgb: Any = None
    frame_w, frame_h = 0, 0

    for frame_idx in range(args.frames):
        print(f"[INFO] Processing frame {frame_idx + 1}/{args.frames}…")
        try:
            frame = frame_source.next_frame(frame_idx)
        except Exception as exc:
            print(f"[WARN] Could not decode frame {frame_idx}: {exc}")
            continue

        frame_w = getattr(frame, "width",  0) or frame_w
        frame_h = getattr(frame, "height", 0) or frame_h

        # Capture RGB for annotation (NV12 → RGB via CUDA)
        if last_frame_rgb is None or frame_idx == args.frames - 1:
            try:
                import torch
                from app_v2.kernels.nv12_cuda_bridge import nv12_to_rgb_hwc_resized_cuda
                with torch.cuda.stream(torch.cuda.Stream()):
                    rgb_t = nv12_to_rgb_hwc_resized_cuda(frame, frame_h, frame_w, stream_id=98)
                last_frame_rgb = rgb_t.cpu().numpy()
            except Exception as exc:
                print(f"[WARN] RGB conversion failed: {exc}")

        # Run inference
        output = preprocessor.build_output(frame_idx, frame)
        frame_results: list[dict[str, Any]] = []
        for model in models:
            processed  = output.flatten_inputs(model.name)
            tile_plan  = output.plans.get(model.name)
            t0 = time.perf_counter_ns()
            try:
                prediction = model.infer(
                    frame_idx,
                    processed,
                    preprocess_events=list(output.cuda_events.values()),
                    tile_plan=tile_plan,
                )
            except Exception as exc:
                print(f"[WARN] Model {model.name} inference failed: {exc}")
                continue
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

            n_det = len(prediction.get("detections", []))
            has_seg = bool(prediction.get("seg_mask_raw"))
            print(
                f"  [{model.name}] {n_det} detections "
                f"{'+ seg mask ' if has_seg else ''}"
                f"in {elapsed_ms:.1f} ms"
            )
            frame_results.append(prediction)

        output.release_all()
        collected_results.extend(frame_results)

    frame_source.disconnect()
    ffmpeg_proc.terminate()
    server.shutdown()

    # ── Build summary report ───────────────────────────────────────────────
    total_dets = sum(len(r.get("detections", [])) for r in collected_results)
    report = {
        "input_file":     input_file,
        "resolution":     args.resolution or "native",
        "stream_url":     stream_url,
        "frames_processed": args.frames,
        "models_used":    [m.name for m in models],
        "total_detections": total_dets,
        "results":        [
            {
                "model":      r.get("model"),
                "frame_id":   r.get("frame_id"),
                "detections": r.get("detections", []),
                "inference_ms": r.get("inference_ms", 0.0),
                "has_seg_mask": bool(r.get("seg_mask_raw")),
                "seg_mask_dims": (r.get("seg_mask_w", 0), r.get("seg_mask_h", 0)),
            }
            for r in collected_results
        ],
    }

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[OK] Report written to {json_path}")
    print(f"     Total detections across {args.frames} frame(s): {total_dets}")

    # ── Annotate and save PNG ──────────────────────────────────────────────
    if last_frame_rgb is not None and total_dets >= 0:
        annotated = _annotate_frame(last_frame_rgb, collected_results, frame_w, frame_h)
        if annotated is not None:
            png_path = output_prefix.with_suffix(".png")
            annotated.save(str(png_path))
            print(f"[OK] Annotated frame saved to {png_path}")
    else:
        print("[WARN] No decoded frame available for annotation.")

    # ── Print quick summary to console ────────────────────────────────────
    print("\n" + "=" * 60)
    print("E2E INFERENCE TEST SUMMARY")
    print("=" * 60)
    for r in report["results"]:
        n = len(r["detections"])
        print(f"  {r['model']:20s}  frame {r['frame_id']:4d}  "
              f"{n:3d} det  {r['inference_ms']:6.1f} ms  "
              f"seg={'yes' if r['has_seg_mask'] else 'no ':>3}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torchvision.io as tvio

from app_v2.kernels.gpu_hotspot_renderer import GpuHotspotRenderer


@dataclass
class BenchResult:
    quality: int
    with_hotspots: bool
    hotspots_count: int
    with_inference: bool
    kernel_ms_median: float
    kernel_ms_p95: float
    draw_ms_median: float
    draw_ms_p95: float
    infer_ms_median: float
    infer_ms_p95: float
    copy_ms_median: float
    copy_ms_p95: float
    total_ms_median: float
    total_ms_p95: float
    fps_effective: float
    jpeg_kb_median: float


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    rank = (len(values_sorted) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(values_sorted) - 1)
    frac = rank - lo
    return values_sorted[lo] * (1.0 - frac) + values_sorted[hi] * frac


def _make_hotspots_tensor(n_points: int, *, device: torch.device) -> torch.Tensor:
    if n_points <= 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    # normalized coordinates + confidence in [0,1]
    x = torch.rand((n_points,), dtype=torch.float32, device=device)
    y = torch.rand((n_points,), dtype=torch.float32, device=device)
    w = 0.5 + 0.5 * torch.rand((n_points,), dtype=torch.float32, device=device)
    return torch.stack((x, y, w), dim=1)


class _P2PLikeInference:
    """Synthetic GPU workload approximating a dense point-based inference tail."""

    def __init__(self, *, device: torch.device, width: int, height: int) -> None:
        self._device = device
        self._conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        self._conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        self._head_logits = torch.nn.Conv2d(64, 2, kernel_size=1, bias=True).to(device)
        self._head_pts = torch.nn.Conv2d(64, 2, kernel_size=1, bias=True).to(device)
        self._input = torch.rand((1, 3, height, width), dtype=torch.float32, device=device)

    def run(self) -> None:
        x = torch.relu(self._conv1(self._input))
        x = torch.relu(self._conv2(x))
        logits = self._head_logits(x)
        pts = self._head_pts(x)
        logits_flat = logits.flatten(2).transpose(1, 2).contiguous()  # [1, N, 2]
        pts_flat = pts.flatten(2).transpose(1, 2).contiguous()        # [1, N, 2]
        scores = torch.softmax(logits_flat[0], dim=-1)[:, 1]
        _ = pts_flat[0][scores > 0.5]


def _load_base_frame(*, image_path: str | None, width: int, height: int, device: torch.device) -> torch.Tensor:
    if image_path:
        frame = tvio.read_image(image_path)
        if frame.dtype != torch.uint8:
            frame = frame.to(torch.uint8)
        frame = frame.to(device=device, non_blocking=True)
        if frame.shape[0] != 3:
            raise RuntimeError(f"Expected RGB image with 3 channels, got shape={tuple(frame.shape)}")
        if int(frame.shape[1]) != height or int(frame.shape[2]) != width:
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0).float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).clamp(0.0, 255.0).to(torch.uint8)
        return frame.contiguous()
    return torch.randint(0, 256, (3, height, width), dtype=torch.uint8, device=device)


def run_bench(
    *,
    image_path: str | None,
    width: int,
    height: int,
    quality: int,
    warmup: int,
    iters: int,
    with_hotspots: bool,
    hotspots_count: int,
    with_inference: bool,
    hotspot_radius: int,
) -> BenchResult:
    device = torch.device("cuda")
    encode_stream = torch.cuda.Stream(device=device)
    infer_stream = torch.cuda.Stream(device=device)
    renderer = GpuHotspotRenderer(
        circle_radius_px=max(1, int(hotspot_radius)),
        color_red=220,
        color_green=30,
        color_blue=30,
    )

    frame = _load_base_frame(image_path=image_path, width=width, height=height, device=device)
    hotspots_tensor = _make_hotspots_tensor(hotspots_count, device=device) if with_hotspots else None
    infer_sim = _P2PLikeInference(device=device, width=width, height=height) if with_inference else None

    kernel_ms: list[float] = []
    draw_ms: list[float] = []
    infer_ms: list[float] = []
    copy_ms: list[float] = []
    total_ms: list[float] = []
    jpeg_kb: list[float] = []
    prev_infer_end_evt: torch.cuda.Event | None = None

    # Warmup
    for _ in range(max(1, warmup)):
        if infer_sim is not None:
            with torch.cuda.stream(infer_stream):
                infer_sim.run()

        with torch.cuda.stream(encode_stream):
            bench_frame = frame.clone()
            if hotspots_tensor is not None:
                bench_frame = renderer.draw_hotspots_on_frame(bench_frame, hotspots_tensor)
            buf = tvio.encode_jpeg(bench_frame, quality=quality)
        encode_stream.synchronize()
        infer_stream.synchronize()
        _ = buf.cpu()

    # Bench
    for _ in range(max(1, iters)):
        if prev_infer_end_evt is not None:
            prev_infer_end_evt.synchronize()

        infer_start_evt: torch.cuda.Event | None = None
        infer_end_evt: torch.cuda.Event | None = None
        if infer_sim is not None:
            infer_start_evt = torch.cuda.Event(enable_timing=True)
            infer_end_evt = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(infer_stream):
                infer_start_evt.record(infer_stream)
                infer_sim.run()
                infer_end_evt.record(infer_stream)

        t0 = time.perf_counter_ns()
        draw_start_evt = torch.cuda.Event(enable_timing=True)
        draw_end_evt = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(encode_stream):
            bench_frame = frame.clone()
            draw_start_evt.record(encode_stream)
            if hotspots_tensor is not None:
                bench_frame = renderer.draw_hotspots_on_frame(bench_frame, hotspots_tensor)
            draw_end_evt.record(encode_stream)
            buf = tvio.encode_jpeg(bench_frame, quality=quality)
        encode_stream.synchronize()
        t1 = time.perf_counter_ns()
        host = buf.cpu()
        t2 = time.perf_counter_ns()

        if infer_start_evt is not None and infer_end_evt is not None:
            infer_end_evt.synchronize()
            infer_ms.append(float(infer_start_evt.elapsed_time(infer_end_evt)))
            prev_infer_end_evt = infer_end_evt
        else:
            prev_infer_end_evt = None

        draw_end_evt.synchronize()
        draw_ms.append(float(draw_start_evt.elapsed_time(draw_end_evt)))

        k_ms = (t1 - t0) / 1_000_000.0
        c_ms = (t2 - t1) / 1_000_000.0
        tot_ms = (t2 - t0) / 1_000_000.0

        kernel_ms.append(k_ms)
        copy_ms.append(c_ms)
        total_ms.append(tot_ms)
        jpeg_kb.append(float(host.numel()) / 1024.0)

    total_med = statistics.median(total_ms)
    fps = 1000.0 / total_med if total_med > 0.0 else 0.0

    return BenchResult(
        quality=quality,
        with_hotspots=with_hotspots,
        hotspots_count=hotspots_count if with_hotspots else 0,
        with_inference=with_inference,
        kernel_ms_median=statistics.median(kernel_ms),
        kernel_ms_p95=_percentile(kernel_ms, 95.0),
        draw_ms_median=statistics.median(draw_ms) if draw_ms else 0.0,
        draw_ms_p95=_percentile(draw_ms, 95.0),
        infer_ms_median=statistics.median(infer_ms) if infer_ms else 0.0,
        infer_ms_p95=_percentile(infer_ms, 95.0),
        copy_ms_median=statistics.median(copy_ms),
        copy_ms_p95=_percentile(copy_ms, 95.0),
        total_ms_median=total_med,
        total_ms_p95=_percentile(total_ms, 95.0),
        fps_effective=fps,
        jpeg_kb_median=statistics.median(jpeg_kb),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NVJPEG throughput at 1080p (or custom WxH), with optional hotspot rendering + p2p-like inference load.")
    parser.add_argument("--image", type=str, default=None, help="Optional image path used as source frame (will be resized to WxH).")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--qualities", type=str, default="75,85,95")
    parser.add_argument("--with-hotspots", action="store_true")
    parser.add_argument("--hotspots-count", type=int, default=3000)
    parser.add_argument("--hotspots-sweep", type=str, default=None, help="Comma list, e.g. 3000,12000,27000")
    parser.add_argument("--simulate-inference", action="store_true", help="Run a synthetic p2p-like inference workload on a parallel CUDA stream.")
    parser.add_argument("--radius", type=int, default=3, help="Hotspot circle radius in pixels for renderer tuning.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; NVJPEG benchmark requires GPU.")

    qualities = [int(x.strip()) for x in args.qualities.split(",") if x.strip()]
    hotspot_counts = [int(args.hotspots_count)]
    if args.hotspots_sweep:
        hotspot_counts = [int(x.strip()) for x in args.hotspots_sweep.split(",") if x.strip()]
    if any(v < 0 for v in hotspot_counts):
        raise SystemExit("hotspot counts must be >= 0")

    print(
        f"[bench] device={torch.cuda.get_device_name(0)} res={args.width}x{args.height} "
        f"warmup={args.warmup} iters={args.iters} qualities={qualities} "
        f"with_hotspots={args.with_hotspots} hotspots={hotspot_counts} "
        f"simulate_inference={args.simulate_inference} image={args.image or 'random'} radius={args.radius}"
    )

    print(
        "quality,with_hotspots,hotspots_count,with_inference,kernel_med_ms,kernel_p95_ms,"
        "draw_med_ms,draw_p95_ms,infer_med_ms,infer_p95_ms,copy_med_ms,copy_p95_ms,"
        "total_med_ms,total_p95_ms,fps_eff,jpeg_kb_med"
    )

    for q in qualities:
        for hs in hotspot_counts:
            with_hotspots = args.with_hotspots or hs > 0
            result = run_bench(
                image_path=args.image,
                width=args.width,
                height=args.height,
                quality=q,
                warmup=args.warmup,
                iters=args.iters,
                with_hotspots=with_hotspots,
                hotspots_count=hs,
                with_inference=args.simulate_inference,
                hotspot_radius=args.radius,
            )
            print(
                f"{result.quality},{int(result.with_hotspots)},{result.hotspots_count},{int(result.with_inference)},"
                f"{result.kernel_ms_median:.3f},{result.kernel_ms_p95:.3f},"
                f"{result.draw_ms_median:.3f},{result.draw_ms_p95:.3f},"
                f"{result.infer_ms_median:.3f},{result.infer_ms_p95:.3f},"
                f"{result.copy_ms_median:.3f},{result.copy_ms_p95:.3f},"
                f"{result.total_ms_median:.3f},{result.total_ms_p95:.3f},"
                f"{result.fps_effective:.2f},{result.jpeg_kb_median:.1f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

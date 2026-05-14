#!/usr/bin/env python3
"""
Benchmark GPU hotspot rendering with GPU EVENTS (not CPU timing).

GPU events measure actual GPU execution time, not CPU enqueue time.
This should reveal the real cost of draw operations.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

import torch
import torchvision.io as tvio

# Add app paths for Docker
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/app_v2')

try:
    from app_v2.kernels.gpu_hotspot_renderer import GpuHotspotRenderer
except ImportError:
    try:
        from kernels.gpu_hotspot_renderer import GpuHotspotRenderer
    except ImportError:
        GpuHotspotRenderer = None


@dataclass
class BenchResult:
    strategy: str
    point_count: int
    iters: int
    draw_ms_median: float
    draw_ms_p95: float
    draw_ms_p99: float
    draw_ms_max: float
    encode_ms_median: float
    encode_ms_p95: float
    total_ms_median: float
    total_ms_p95: float
    fps_effective: float
    speedup_vs_current: float = 1.0

    def __str__(self) -> str:
        return (
            f"{self.strategy:15s} | pts={self.point_count:5d} | "
            f"draw={self.draw_ms_median:7.2f}ms (p95={self.draw_ms_p95:7.2f}, max={self.draw_ms_max:7.2f}) | "
            f"encode={self.encode_ms_median:7.2f}ms | "
            f"total={self.total_ms_median:7.2f}ms | fps={self.fps_effective:6.1f} | "
            f"speedup={self.speedup_vs_current:5.2f}x"
        )


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile."""
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
    """Generate random normalized hotspots."""
    if n_points <= 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    x = torch.rand((n_points,), dtype=torch.float32, device=device)
    y = torch.rand((n_points,), dtype=torch.float32, device=device)
    w = 0.5 + 0.5 * torch.rand((n_points,), dtype=torch.float32, device=device)
    return torch.stack((x, y, w), dim=1)


def run_bench_gpu_events(
    *,
    strategy_name: str,
    frame_base: torch.Tensor,
    hotspots: torch.Tensor,
    hotspots_count: int,
    iters: int,
    quality: int = 95,
) -> BenchResult:
    """
    Benchmark using GPU events to measure actual GPU execution time.
    
    Measures:
    - draw_ms: GPU time for hotspot rendering
    - encode_ms: GPU time for NVJPEG
    - total_ms: sum of both
    """
    device = frame_base.device
    encode_stream = torch.cuda.Stream(device=device)

    draw_times = []
    encode_times = []

    # Choose render strategy
    if strategy_name == "THINNED-2x":
        hotspots_to_use = hotspots[::2]
    elif strategy_name == "THINNED-5x":
        hotspots_to_use = hotspots[::5]
    else:
        hotspots_to_use = hotspots

    # Warmup
    for _ in range(20):
        torch.cuda.synchronize()
        with torch.cuda.stream(encode_stream):
            frame = frame_base.clone()
            renderer = GpuHotspotRenderer(circle_radius_px=3, color_red=220, color_green=30, color_blue=30)
            frame = renderer.draw_hotspots_on_frame(frame, hotspots_to_use)
            buf = tvio.encode_jpeg(frame, quality=quality)
        encode_stream.synchronize()

    # Actual benchmark with GPU events
    for _ in range(iters):
        draw_start = torch.cuda.Event(enable_timing=True)
        draw_end = torch.cuda.Event(enable_timing=True)
        encode_start = torch.cuda.Event(enable_timing=True)
        encode_end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(encode_stream):
            frame = frame_base.clone()

            draw_start.record(encode_stream)
            renderer = GpuHotspotRenderer(circle_radius_px=3, color_red=220, color_green=30, color_blue=30)
            frame = renderer.draw_hotspots_on_frame(frame, hotspots_to_use)
            draw_end.record(encode_stream)

            encode_start.record(encode_stream)
            buf = tvio.encode_jpeg(frame, quality=quality)
            encode_end.record(encode_stream)

        encode_stream.synchronize()

        draw_ms = float(draw_start.elapsed_time(draw_end))
        encode_ms = float(encode_start.elapsed_time(encode_end))

        draw_times.append(draw_ms)
        encode_times.append(encode_ms)

    total_times = [d + e for d, e in zip(draw_times, encode_times)]
    total_ms_median = statistics.median(total_times)

    return BenchResult(
        strategy=strategy_name,
        point_count=hotspots_count,
        iters=iters,
        draw_ms_median=statistics.median(draw_times),
        draw_ms_p95=_percentile(draw_times, 95.0),
        draw_ms_p99=_percentile(draw_times, 99.0),
        draw_ms_max=max(draw_times),
        encode_ms_median=statistics.median(encode_times),
        encode_ms_p95=_percentile(encode_times, 95.0),
        total_ms_median=total_ms_median,
        total_ms_p95=_percentile(total_times, 95.0),
        fps_effective=1000.0 / total_ms_median if total_ms_median > 0 else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU hotspot rendering using GPU events for accurate timing"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    frame = torch.randint(0, 256, (3, args.height, args.width), dtype=torch.uint8, device=device)

    print(f"\n{'Resolution':20s} {args.width}x{args.height}")
    print(f"{'Iters':20s} {args.iters}")
    print(f"{'JPEG Quality':20s} {args.quality}")
    print(f"{'Device':20s} {torch.cuda.get_device_name(0)}")

    point_counts = [20000, 50000]
    all_results = []

    for pt_count in point_counts:
        print(f"\n{'=' * 140}")
        print(f"🎯 Testing with {pt_count} points")
        print(f"{'=' * 140}")

        hotspots = _make_hotspots_tensor(pt_count, device=device)

        strategies = [
            ("CURRENT", pt_count, hotspots),
            ("THINNED-2x", pt_count // 2, hotspots),
            ("THINNED-5x", pt_count // 5, hotspots),
        ]

        baseline = None
        for i, (strat_name, effective_count, _) in enumerate(strategies, 1):
            print(f"\n[{i}/{len(strategies)}] {strat_name} ({effective_count} points)...")
            result = run_bench_gpu_events(
                strategy_name=strat_name,
                frame_base=frame,
                hotspots=hotspots,
                hotspots_count=effective_count,
                iters=args.iters,
                quality=args.quality,
            )
            if baseline is None:
                baseline = result
            else:
                result.speedup_vs_current = baseline.draw_ms_median / result.draw_ms_median
            print(result)
            all_results.append(result)

    print(f"\n{'=' * 140}")
    print("📊 SUMMARY TABLE")
    print(f"{'=' * 140}")
    for result in all_results:
        print(result)

    print(f"\n{'=' * 140}")
    print("🎯 PERFORMANCE ANALYSIS")
    print(f"{'=' * 140}")

    for pt_count in point_counts:
        results = [r for r in all_results if r.point_count == pt_count or (pt_count == 20000 and r.point_count in [10000, 4000]) or (pt_count == 50000 and r.point_count in [25000, 10000])]
        current = next((r for r in results if r.strategy == "CURRENT"), None)
        if not current:
            continue

        print(f"\n{pt_count} points → {len([r for r in results if 'CURRENT' in r.strategy or True])} strategies:")
        print(f"  Baseline (CURRENT): draw={current.draw_ms_median:.2f}ms, encode={current.encode_ms_median:.2f}ms, total={current.total_ms_median:.2f}ms")

        for result in results:
            if result.strategy == "CURRENT":
                continue
            saved_draw = current.draw_ms_median - result.draw_ms_median
            speedup = current.draw_ms_median / result.draw_ms_median if result.draw_ms_median > 0 else 999
            print(f"  {result.strategy:12s}: draw={result.draw_ms_median:7.2f}ms "
                  f"(save {saved_draw:6.2f}ms, {speedup:5.2f}x faster), "
                  f"total={result.total_ms_median:7.2f}ms")

    return 0


if __name__ == "__main__":
    exit(main())

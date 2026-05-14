#!/usr/bin/env python3
"""
Benchmark GPU hotspot rendering following the exact orchestrator path.

This matches what happens in _encode_and_push_nvjpeg:
1. Clone frame
2. Wait for frame ready event
3. Draw hotspots on GPU (enqueued)
4. JPEG encode (enqueued)
5. Synchronize stream
6. CPU copy
7. Push
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torchvision.io as tvio

try:
    from app_v2.kernels.gpu_hotspot_renderer import GpuHotspotRenderer
except ImportError:
    GpuHotspotRenderer = None


@dataclass
class BenchResult:
    strategy: str
    point_count: int
    iters: int
    draw_only_ms_median: float
    draw_only_ms_p95: float
    draw_only_ms_p99: float
    draw_only_ms_max: float
    full_cycle_ms_median: float
    full_cycle_ms_p95: float
    full_cycle_ms_p99: float
    full_cycle_ms_max: float
    speedup_vs_current: float = 1.0

    def __str__(self) -> str:
        return (
            f"{self.strategy:15s} | pts={self.point_count:5d} | "
            f"draw_only={self.draw_only_ms_median:7.2f}ms (p95={self.draw_only_ms_p95:7.2f}) | "
            f"full_sync={self.full_cycle_ms_median:7.2f}ms (p95={self.full_cycle_ms_p95:7.2f}) | "
            f"speedup={self.speedup_vs_current:5.2f}x"
        )


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile of a list."""
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
    """Generate random normalized hotspots [0, 1]²."""
    if n_points <= 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    x = torch.rand((n_points,), dtype=torch.float32, device=device)
    y = torch.rand((n_points,), dtype=torch.float32, device=device)
    w = 0.5 + 0.5 * torch.rand((n_points,), dtype=torch.float32, device=device)
    return torch.stack((x, y, w), dim=1)


def strategy_current(frame: torch.Tensor, hotspots: torch.Tensor) -> torch.Tensor:
    """Current vectorized PyTorch strategy."""
    if GpuHotspotRenderer is None:
        return frame
    renderer = GpuHotspotRenderer(circle_radius_px=3, color_red=220, color_green=30, color_blue=30)
    return renderer.draw_hotspots_on_frame(frame, hotspots)


def strategy_thinned(frame: torch.Tensor, hotspots: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """Downsample points by stride, then render."""
    if GpuHotspotRenderer is None:
        return frame
    if hotspots.numel() == 0:
        return frame
    thinned = hotspots[::stride]
    renderer = GpuHotspotRenderer(circle_radius_px=3, color_red=220, color_green=30, color_blue=30)
    return renderer.draw_hotspots_on_frame(frame, thinned)


def run_bench_orchestrator_style(
    *,
    strategy_name: str,
    strategy_fn: Callable,
    frame: torch.Tensor,
    hotspots: torch.Tensor,
    hotspots_count: int,
    iters: int,
    quality: int = 95,
    **strategy_kwargs,
) -> BenchResult:
    """
    Benchmark following orchestrator path:
    1. Clone frame
    2. Wait for event (simulated with synchronize())
    3. Draw hotspots + JPEG encode (both enqueued on stream)
    4. Synchronize stream
    5. CPU copy + push
    
    Measures:
    - draw_only: from after wait to after draw call (GPU work enqueued)
    - full_cycle: from clone to after CPU copy (including stream sync)
    """
    device = frame.device
    encode_stream = torch.cuda.Stream(device=device)

    draw_only_times = []
    full_cycle_times = []

    # Warmup
    for _ in range(20):
        torch.cuda.synchronize()
        with torch.cuda.stream(encode_stream):
            bench_frame = frame.clone()
            bench_frame = strategy_fn(bench_frame, hotspots, **strategy_kwargs)
            buf = tvio.encode_jpeg(bench_frame, quality=quality)
        encode_stream.synchronize()

    # Actual benchmark
    for _ in range(iters):
        # Simulate frame ready event
        evt = torch.cuda.Event()
        evt.record()

        # Measure full cycle (clone → draw → encode → sync → copy)
        t_cycle_start = time.perf_counter_ns()

        with torch.cuda.stream(encode_stream):
            # Clone frame
            bench_frame = frame.clone()

            # Wait for frame ready (simulated)
            evt.synchronize()
            t_draw_start = time.perf_counter_ns()

            # Draw hotspots (enqueued on stream)
            bench_frame = strategy_fn(bench_frame, hotspots, **strategy_kwargs)
            t_draw_end = time.perf_counter_ns()

            # Encode JPEG (enqueued on stream)
            buf = tvio.encode_jpeg(bench_frame, quality=quality)

        # Synchronize stream (this is where GPU work completes)
        encode_stream.synchronize()
        t_cycle_end = time.perf_counter_ns()

        # CPU copy
        jpeg_bytes = buf.cpu().numpy().tobytes()
        t_cpu_copy_done = time.perf_counter_ns()

        draw_only_ms = (t_draw_end - t_draw_start) / 1_000_000.0
        full_cycle_ms = (t_cpu_copy_done - t_cycle_start) / 1_000_000.0

        draw_only_times.append(draw_only_ms)
        full_cycle_times.append(full_cycle_ms)

    return BenchResult(
        strategy=strategy_name,
        point_count=hotspots_count,
        iters=iters,
        draw_only_ms_median=statistics.median(draw_only_times),
        draw_only_ms_p95=_percentile(draw_only_times, 95.0),
        draw_only_ms_p99=_percentile(draw_only_times, 99.0),
        draw_only_ms_max=max(draw_only_times),
        full_cycle_ms_median=statistics.median(full_cycle_times),
        full_cycle_ms_p95=_percentile(full_cycle_times, 95.0),
        full_cycle_ms_p99=_percentile(full_cycle_times, 99.0),
        full_cycle_ms_max=max(full_cycle_times),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU hotspot rendering following orchestrator path"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")

    # Create base frame
    frame = torch.randint(0, 256, (3, args.height, args.width), dtype=torch.uint8, device=device)

    print(f"\n{'Resolution':20s} {args.width}x{args.height}")
    print(f"{'Iters':20s} {args.iters}")
    print(f"{'JPEG Quality':20s} {args.quality}")
    print(f"{'Device':20s} {torch.cuda.get_device_name(0)}")

    point_counts = [20000, 50000]
    all_results = []

    for pt_count in point_counts:
        print(f"\n{'=' * 130}")
        print(f"Testing with {pt_count} points")
        print(f"{'=' * 130}")

        hotspots = _make_hotspots_tensor(pt_count, device=device)
        baseline_result = None

        # 1. Current strategy
        print(f"\n[1/3] Testing CURRENT (vectorized PyTorch)...")
        result_current = run_bench_orchestrator_style(
            strategy_name="CURRENT",
            strategy_fn=strategy_current,
            frame=frame,
            hotspots=hotspots,
            hotspots_count=pt_count,
            iters=args.iters,
            quality=args.quality,
        )
        print(result_current)
        all_results.append(result_current)
        baseline_result = result_current

        # 2. Thinned (stride=2)
        print(f"\n[2/3] Testing THINNED (stride=2, {pt_count//2} points)...")
        result_thinned_2 = run_bench_orchestrator_style(
            strategy_name="THINNED-2x",
            strategy_fn=strategy_thinned,
            frame=frame,
            hotspots=hotspots,
            hotspots_count=pt_count // 2,
            iters=args.iters,
            quality=args.quality,
            stride=2,
        )
        result_thinned_2.speedup_vs_current = baseline_result.draw_only_ms_median / result_thinned_2.draw_only_ms_median
        print(result_thinned_2)
        all_results.append(result_thinned_2)

        # 3. Thinned (stride=5)
        print(f"\n[3/3] Testing THINNED (stride=5, {pt_count//5} points)...")
        result_thinned_5 = run_bench_orchestrator_style(
            strategy_name="THINNED-5x",
            strategy_fn=strategy_thinned,
            frame=frame,
            hotspots=hotspots,
            hotspots_count=pt_count // 5,
            iters=args.iters,
            quality=args.quality,
            stride=5,
        )
        result_thinned_5.speedup_vs_current = baseline_result.draw_only_ms_median / result_thinned_5.draw_only_ms_median
        print(result_thinned_5)
        all_results.append(result_thinned_5)

    print(f"\n{'=' * 130}")
    print("SUMMARY")
    print(f"{'=' * 130}")
    for result in all_results:
        print(result)

    # Compute recommendations
    print(f"\n{'=' * 130}")
    print("PERFORMANCE RECOMMENDATIONS")
    print(f"{'=' * 130}")

    for pt_count in point_counts:
        results_for_count = [r for r in all_results if r.point_count == pt_count]
        if not results_for_count:
            continue

        current = next((r for r in results_for_count if r.strategy == "CURRENT"), None)
        if not current:
            continue

        print(f"\n📊 For {pt_count} points:")
        print(f"  Current draw time: {current.draw_only_ms_median:.2f}ms (full cycle: {current.full_cycle_ms_median:.2f}ms)")

        for result in results_for_count:
            if result.strategy == "CURRENT":
                continue
            speedup_draw = current.draw_only_ms_median / result.draw_only_ms_median
            saved_draw_ms = current.draw_only_ms_median - result.draw_only_ms_median
            saved_cycle_ms = current.full_cycle_ms_median - result.full_cycle_ms_median
            print(f"  {result.strategy:15s}: draw={result.draw_only_ms_median:7.2f}ms ({speedup_draw:5.2f}x), "
                  f"save {saved_draw_ms:6.2f}ms draw, {saved_cycle_ms:6.2f}ms total")

    return 0


if __name__ == "__main__":
    exit(main())

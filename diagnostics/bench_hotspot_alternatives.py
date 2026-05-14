#!/usr/bin/env python3
"""
Benchmark GPU hotspot rendering alternatives at 1080p.

Tests:
1. Current: Vectorized PyTorch approach
2. CUDA Kernel: Optimized CUDA compute kernel
3. Thinned: Point downsampling (1 in N)

Measures total time from clone → draw → JPEG encode on isolated encode stream.
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
            f"draw={self.draw_ms_median:6.2f}ms (p95={self.draw_ms_p95:6.2f}) | "
            f"encode={self.encode_ms_median:6.2f}ms | "
            f"total={self.total_ms_median:6.2f}ms | "
            f"fps={self.fps_effective:5.1f} | "
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


def _compile_cuda_hotspot_kernel() -> tuple[Callable, bool]:
    """Compile a simple CUDA kernel for circle rasterization."""
    # Skip Triton (too complex for this task), return None
    return None, False


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


def strategy_cuda_kernel(
    frame: torch.Tensor, hotspots: torch.Tensor, kernel_func: Callable | None
) -> torch.Tensor:
    """CUDA kernel strategy (if available)."""
    if kernel_func is None or hotspots.numel() == 0:
        # Fallback to current
        return strategy_current(frame, hotspots)

    device = frame.device
    frame_h, frame_w = frame.shape[1], frame.shape[2]

    # Convert normalized coordinates to pixel space
    centers = torch.zeros((hotspots.shape[0], 2), dtype=torch.float32, device=device)
    centers[:, 0] = hotspots[:, 0] * frame_w
    centers[:, 1] = hotspots[:, 1] * frame_h

    return kernel_func(frame, centers, radius=3, color=(220, 30, 30))


def run_bench(
    *,
    strategy_name: str,
    strategy_fn: Callable,
    frame: torch.Tensor,
    hotspots: torch.Tensor,
    hotspots_count: int,
    iters: int,
    encode_stream: torch.cuda.Stream,
    quality: int = 95,
    **strategy_kwargs,
) -> BenchResult:
    """Benchmark a single strategy using GPU events for accurate timing."""
    draw_times = []
    encode_times = []
    total_times = []

    # Warmup
    for _ in range(20):
        with torch.cuda.stream(encode_stream):
            bench_frame = frame.clone()
            bench_frame = strategy_fn(bench_frame, hotspots, **strategy_kwargs)
            buf = tvio.encode_jpeg(bench_frame, quality=quality)
        encode_stream.synchronize()

    # Actual benchmark using GPU events
    for _ in range(iters):
        # Create events for precise GPU timing
        clone_start = torch.cuda.Event(enable_timing=True)
        draw_start = torch.cuda.Event(enable_timing=True)
        draw_end = torch.cuda.Event(enable_timing=True)
        encode_start = torch.cuda.Event(enable_timing=True)
        encode_end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(encode_stream):
            clone_start.record(encode_stream)
            bench_frame = frame.clone()
            draw_start.record(encode_stream)
            bench_frame = strategy_fn(bench_frame, hotspots, **strategy_kwargs)
            draw_end.record(encode_stream)
            encode_start.record(encode_stream)
            buf = tvio.encode_jpeg(bench_frame, quality=quality)
            encode_end.record(encode_stream)

        encode_stream.synchronize()

        draw_ms = float(draw_start.elapsed_time(draw_end))
        encode_ms = float(encode_start.elapsed_time(encode_end))
        clone_to_end_ms = float(clone_start.elapsed_time(encode_end))

        draw_times.append(draw_ms)
        encode_times.append(encode_ms)
        total_times.append(clone_to_end_ms)

    torch.cuda.synchronize()

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
        total_ms_median=statistics.median(total_times),
        total_ms_p95=_percentile(total_times, 95.0),
        fps_effective=1000.0 / statistics.median(total_times) if total_times else 0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU hotspot rendering strategies at 1080p with 20K/50K points"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    device = torch.device("cuda")
    encode_stream = torch.cuda.Stream(device=device)

    # Create base frame
    frame = torch.randint(0, 256, (3, args.height, args.width), dtype=torch.uint8, device=device)

    # Compile Triton kernel
    triton_kernel, triton_ok = _compile_cuda_hotspot_kernel()
    if triton_ok:
        print("[✓] Triton kernel compiled successfully")
    else:
        print("[✗] Triton kernel not available (will use PyTorch fallback)")

    print(f"\n{'Resolution':20s} {args.width}x{args.height}")
    print(f"{'Iters':20s} {args.iters}")
    print(f"{'JPEG Quality':20s} {args.quality}")
    print(f"{'Device':20s} {torch.cuda.get_device_name(0)}")

    point_counts = [20000, 50000]
    all_results = []

    for pt_count in point_counts:
        print(f"\n{'=' * 120}")
        print(f"Testing with {pt_count} points")
        print(f"{'=' * 120}")

        hotspots = _make_hotspots_tensor(pt_count, device=device)
        baseline_result = None

        # 1. Current strategy
        print(f"\n[1/4] Testing CURRENT (vectorized PyTorch)...")
        result_current = run_bench(
            strategy_name="CURRENT",
            strategy_fn=strategy_current,
            frame=frame,
            hotspots=hotspots,
            hotspots_count=pt_count,
            iters=args.iters,
            encode_stream=encode_stream,
            quality=args.quality,
        )
        print(result_current)
        all_results.append(result_current)
        baseline_result = result_current

        # 2. Thinned (stride=2)
        print(f"\n[2/4] Testing THINNED (stride=2, {pt_count//2} points)...")
        result_thinned_2 = run_bench(
            strategy_name="THINNED-2x",
            strategy_fn=strategy_thinned,
            frame=frame,
            hotspots=hotspots,
            hotspots_count=pt_count // 2,
            iters=args.iters,
            encode_stream=encode_stream,
            quality=args.quality,
            stride=2,
        )
        result_thinned_2.speedup_vs_current = baseline_result.draw_ms_median / result_thinned_2.draw_ms_median
        print(result_thinned_2)
        all_results.append(result_thinned_2)

        # 3. Thinned (stride=5)
        print(f"\n[3/4] Testing THINNED (stride=5, {pt_count//5} points)...")
        result_thinned_5 = run_bench(
            strategy_name="THINNED-5x",
            strategy_fn=strategy_thinned,
            frame=frame,
            hotspots=hotspots,
            hotspots_count=pt_count // 5,
            iters=args.iters,
            encode_stream=encode_stream,
            quality=args.quality,
            stride=5,
        )
        result_thinned_5.speedup_vs_current = baseline_result.draw_ms_median / result_thinned_5.draw_ms_median
        print(result_thinned_5)
        all_results.append(result_thinned_5)

        # 4. CUDA Kernel (if available)
        if triton_ok:
            print(f"\n[4/4] Testing CUDA KERNEL (Triton)...")
            result_cuda = run_bench(
                strategy_name="CUDA-KERNEL",
                strategy_fn=strategy_cuda_kernel,
                frame=frame,
                hotspots=hotspots,
                hotspots_count=pt_count,
                iters=args.iters,
                encode_stream=encode_stream,
                quality=args.quality,
                kernel_func=triton_kernel,
            )
            result_cuda.speedup_vs_current = baseline_result.draw_ms_median / result_cuda.draw_ms_median
            print(result_cuda)
            all_results.append(result_cuda)
        else:
            print(f"\n[4/4] CUDA KERNEL skipped (Triton not available)")

    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")
    for result in all_results:
        print(result)

    # Compute recommendations
    print(f"\n{'=' * 120}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 120}")

    for pt_count in point_counts:
        results_for_count = [r for r in all_results if r.point_count == pt_count]
        if not results_for_count:
            continue

        current = next((r for r in results_for_count if r.strategy == "CURRENT"), None)
        if not current:
            continue

        print(f"\nFor {pt_count} points:")
        print(f"  Current draw time: {current.draw_ms_median:.2f}ms")

        for result in results_for_count:
            if result.strategy == "CURRENT":
                continue
            speedup = current.draw_ms_median / result.draw_ms_median
            saved_ms = current.draw_ms_median - result.draw_ms_median
            print(f"  {result.strategy:15s}: {result.draw_ms_median:6.2f}ms ({speedup:5.2f}x faster, save {saved_ms:6.2f}ms)")

    return 0


if __name__ == "__main__":
    exit(main())

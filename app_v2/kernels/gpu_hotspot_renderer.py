"""GPU-resident hotspot rendering kernel.

Renders P2PNet/Density hotspots directly onto RGB frames before JPEG encoding.
Ultra-simple: just fill circles with a fixed color (red for P2PNet/Density).
No alpha blending, no fancy stuff — pure parallel performance.
"""

from __future__ import annotations

import time
from typing import Any

import torch

class GpuHotspotRenderer:
    """Ultra-fast GPU hotspot circle renderer for dense scenes."""

    def __init__(
        self,
        circle_radius_px: int = 3,
        color_red: int = 220,
        color_green: int = 30,
        color_blue: int = 30,
    ):
        """
        Args:
            circle_radius_px: Circle radius in pixels (3–5 typical for dense scenes)
            color_red, color_green, color_blue: RGB color values
        """
        self.circle_radius_px = int(circle_radius_px)
        self.color_red = int(color_red) & 0xFF
        self.color_green = int(color_green) & 0xFF
        self.color_blue = int(color_blue) & 0xFF
        self._disk_offsets_cache: dict[tuple[int, str], torch.Tensor] = {}

    def draw_hotspots_on_frame(
        self,
        frame_chw_uint8: torch.Tensor,
        hotspots_list: list[tuple[float, float, float]] | torch.Tensor | None,
        frame_id: int | None = None,
        metrics_callback: Any | None = None,
    ) -> torch.Tensor:
        """
        Draw hotspots directly onto an RGB frame (GPU-resident, ultra-fast).

        Uses PyTorch tensor operations for parallel circle rasterization.
        No CPU-GPU sync, entirely on GPU device.

        Args:
            frame_chw_uint8: [3, H, W] uint8 CUDA tensor (CHW RGB)
            hotspots_list: list of (x_norm, y_norm, w_confidence) tuples
            frame_id: optional frame ID for caching/debugging
            metrics_callback: optional callback for timing metrics

        Returns:
            Modified frame_chw_uint8 (same tensor, modified in-place on GPU)
        """
        perf_start_ns = time.perf_counter_ns()

        if hotspots_list is None:
            return frame_chw_uint8

        # Validate frame tensor
        if not isinstance(frame_chw_uint8, torch.Tensor):
            return frame_chw_uint8
        if frame_chw_uint8.dim() != 3 or frame_chw_uint8.shape[0] != 3:
            return frame_chw_uint8
        if frame_chw_uint8.dtype != torch.uint8 or not frame_chw_uint8.is_cuda:
            return frame_chw_uint8

        frame_h = int(frame_chw_uint8.shape[1])
        frame_w = int(frame_chw_uint8.shape[2])
        device = frame_chw_uint8.device

        try:
            # Create coordinate grids for the frame
            # yy, xx = torch.meshgrid(torch.arange(frame_h, dtype=torch.float32, device=device),
            #                         torch.arange(frame_w, dtype=torch.float32, device=device),
            #                         indexing='ij')

            # Vectorized pipeline:
            # 1) Build hotspot centers tensor on GPU
            # 2) Expand with precomputed disk offsets (radius)
            # 3) Clamp to frame bounds and write RGB in one pass per channel
            if isinstance(hotspots_list, torch.Tensor):
                hotspot_tensor_gpu = hotspots_list
                if hotspot_tensor_gpu.device != device:
                    hotspot_tensor_gpu = hotspot_tensor_gpu.to(device=device, non_blocking=True)
                hotspot_tensor_gpu = hotspot_tensor_gpu.float()
            else:
                if len(hotspots_list) == 0:
                    return frame_chw_uint8
                hotspot_tensor_cpu = torch.tensor(hotspots_list, dtype=torch.float32)
                hotspot_tensor_gpu = hotspot_tensor_cpu.to(device=device, non_blocking=True)

            if hotspot_tensor_gpu.numel() == 0:
                return frame_chw_uint8

            center_x = (hotspot_tensor_gpu[:, 0] * frame_w).round().to(torch.int64)
            center_y = (hotspot_tensor_gpu[:, 1] * frame_h).round().to(torch.int64)

            hotspot_count = int(hotspot_tensor_gpu.shape[0])
            effective_radius = self.circle_radius_px
            # Dense scenes: shrink circles to stabilize NVJPEG+render latency.
            # (radius is in pixels, so 1 => ~2 px diameter points)
            if hotspot_count >= 8000:
                effective_radius = 1
            elif hotspot_count >= 3000:
                effective_radius = min(effective_radius, 2)

            cache_key = (effective_radius, str(device))
            disk_offsets = self._disk_offsets_cache.get(cache_key)
            if disk_offsets is None:
                radius = effective_radius
                offset_range = torch.arange(-radius, radius + 1, device=device, dtype=torch.int64)
                grid_y, grid_x = torch.meshgrid(offset_range, offset_range, indexing="ij")
                disk_mask = (grid_x * grid_x + grid_y * grid_y) <= (radius * radius)
                disk_offsets = torch.stack((grid_x[disk_mask], grid_y[disk_mask]), dim=1)
                self._disk_offsets_cache[cache_key] = disk_offsets

            all_x = center_x[:, None] + disk_offsets[:, 0][None, :]
            all_y = center_y[:, None] + disk_offsets[:, 1][None, :]

            valid_mask = (all_x >= 0) & (all_x < frame_w) & (all_y >= 0) & (all_y < frame_h)
            valid_x = all_x[valid_mask]
            valid_y = all_y[valid_mask]

            if valid_x.numel() == 0:
                return frame_chw_uint8

            # Keep duplicates: assigning the same color multiple times is harmless
            # and avoids an expensive global unique() that causes latency spikes
            # on very dense scenes.
            linear_indices = (valid_y * frame_w + valid_x)
            flat_red = frame_chw_uint8[0].view(-1)
            flat_green = frame_chw_uint8[1].view(-1)
            flat_blue = frame_chw_uint8[2].view(-1)
            flat_red[linear_indices] = self.color_red
            flat_green[linear_indices] = self.color_green
            flat_blue[linear_indices] = self.color_blue

        except Exception as exc:
            import sys

            print(f"[GpuHotspotRenderer] draw failed: {exc}", file=sys.stderr, flush=True)
            return frame_chw_uint8

        perf_done_ns = time.perf_counter_ns()

        if callable(metrics_callback):
            metrics_callback(
                {
                    "video_gpu_hotspot_render_ms": (perf_done_ns - perf_start_ns) / 1_000_000.0,
                    "video_gpu_hotspot_count": int(hotspot_tensor_gpu.shape[0]),
                }
            )

        return frame_chw_uint8

    # Renderer is stateless regarding frames; hotspot cache is managed by server.py

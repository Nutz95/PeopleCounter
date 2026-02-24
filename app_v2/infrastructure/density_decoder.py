from __future__ import annotations

import base64
from typing import Any

from app_v2.core.postprocessor import Postprocessor


class DensityDecoder(Postprocessor):
    """Transforms DM-Count tile outputs into a stitched heatmap and count.

    DM-Count outputs a [B, 1, H_out, W_out] density map where summing all
    pixels gives the crowd count.  For 4K tiling (6 cols × 3 rows = 18 tiles,
    no overlap, each 640×720 input): H_out ≈ 45, W_out ≈ 40, global heatmap
    ≈ 270×240 pixels before resizing.
    """

    def process(self, frame_id: int, outputs: dict[str, Any]) -> dict[str, Any]:
        """Stitch per-tile density maps, return count + base64-encoded heatmap.

        Expected ``outputs`` keys (from ``DensityTRT.infer()``):
            density_tiles  – list[Tensor[1, H_out, W_out]] fp32
            density_count  – float total people count across all tiles
            tile_plan      – PreprocessPlan (for spatial reconstruction)
        """
        density_tiles = outputs.get("density_tiles", [])
        total_count   = float(outputs.get("density_count", 0.0))
        tile_plan     = outputs.get("tile_plan")

        heatmap_b64: str | None = None
        heatmap_w = 0
        heatmap_h = 0

        if density_tiles and tile_plan is not None:
            heatmap_b64, heatmap_w, heatmap_h = self._stitch_heatmap(density_tiles, tile_plan)

        return {
            "frame_id": frame_id,
            "model": "density",
            "density_count": total_count,
            "heatmap_raw": heatmap_b64,
            "heatmap_w": heatmap_w,
            "heatmap_h": heatmap_h,
        }

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _stitch_heatmap(
        density_tiles: list[Any],
        tile_plan: Any,
    ) -> tuple[str | None, int, int]:
        """Composite per-tile density maps into a global frame heatmap.

        The output is a quarter-resolution canvas (frame_w//4 × frame_h//4) to
        keep memory and bandwidth reasonable.  Values are scaled to [0, 255]
        uint8 and base64-encoded, matching the seg-mask contract.

        Returns (base64_str, width, height) or (None, 0, 0) on error.
        """
        try:
            import torch
            import numpy as np
        except ModuleNotFoundError:
            return None, 0, 0

        fw = getattr(tile_plan, "frame_width", 0)
        fh = getattr(tile_plan, "frame_height", 0)
        tasks = getattr(tile_plan, "tasks", ())
        if fw <= 0 or fh <= 0 or not tasks:
            return None, 0, 0

        # Quarter-resolution global canvas
        canvas_w = max(1, fw // 4)
        canvas_h = max(1, fh // 4)
        canvas = torch.zeros(canvas_h, canvas_w, dtype=torch.float32)

        for i, task in enumerate(tasks):
            if i >= len(density_tiles):
                break
            tile_map = density_tiles[i]  # [1, H_out, W_out] or [H_out, W_out]
            if tile_map is None:
                continue
            if isinstance(tile_map, torch.Tensor):
                tile_map = tile_map.squeeze()  # → [H_out, W_out]
                if tile_map.dim() == 0:
                    continue
                tile_map = tile_map.float().cpu()
            else:
                try:
                    tile_map = torch.as_tensor(tile_map, dtype=torch.float32).squeeze()
                except Exception:
                    continue

            # Destination region in canvas coords
            dst_x1 = int(round(task.source_x / fw * canvas_w))
            dst_y1 = int(round(task.source_y / fh * canvas_h))
            dst_x2 = int(round((task.source_x + task.source_width)  / fw * canvas_w))
            dst_y2 = int(round((task.source_y + task.source_height) / fh * canvas_h))
            dst_x2 = min(dst_x2, canvas_w)
            dst_y2 = min(dst_y2, canvas_h)
            dst_w = max(1, dst_x2 - dst_x1)
            dst_h = max(1, dst_y2 - dst_y1)

            # Resize tile density map to the canvas region
            resized = torch.nn.functional.interpolate(
                tile_map.unsqueeze(0).unsqueeze(0),
                size=(dst_h, dst_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()  # [dst_h, dst_w]

            canvas[dst_y1:dst_y1 + dst_h, dst_x1:dst_x1 + dst_w] += resized

        # Normalise to [0, 255] uint8 for transport
        c_max = float(canvas.max().item())
        if c_max > 0:
            canvas = canvas / c_max
        arr = (canvas.numpy() * 255).astype("uint8")
        b64 = base64.b64encode(arr.tobytes()).decode("ascii")
        return b64, canvas_w, canvas_h

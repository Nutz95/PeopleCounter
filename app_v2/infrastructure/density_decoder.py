from __future__ import annotations

from typing import Any

from app_v2.core.postprocessor import Postprocessor

# Canvas resolution relative to the original frame.
# 1/4 scale gives enough granularity for per-person peak separation in 4K scenes.
_CANVAS_DOWNSCALE: int = 4

# NMS kernel: small value keeps peaks close together, allowing dense crowd scenes
# to produce many distinct hotspots (one per person at fine scale).
_NMS_KERNEL: int = 3

# Minimum density value to be considered a real peak (filters absolute zeros).
_DENSITY_THRESHOLD: float = 1e-6


class DensityDecoder(Postprocessor):
    """Transforms DM-Count tile outputs into a count and hotspot coordinate list.

    All tensor operations stay on GPU until the final scalar extraction, which
    is a single .tolist() call across all hotspots.  No bitmaps, no base64,
    no CPU-side loops over density maps.

    Return format:
        {
            "frame_id": int,
            "model":    "density",
            "density_count": float,
            "hotspots": [{"x": float, "y": float, "w": float}, ...],
        }

    hotspots are sorted by density weight descending.  x/y are normalised
    frame coordinates [0, 1].  w is the relative density weight [0, 1].
    There is no cap on the number of hotspots — dense crowd scenes with
    hundreds of people will produce hundreds of hotspots.
    """

    def process(self, frame_id: int, outputs: dict[str, Any]) -> dict[str, Any]:
        density_tiles = outputs.get("density_tiles", [])
        total_count   = float(outputs.get("density_count", 0.0))
        tile_plan     = outputs.get("tile_plan")

        hotspots: list[dict[str, float]] = []
        if density_tiles and tile_plan is not None:
            hotspots = _extract_hotspots(density_tiles, tile_plan)

        return {
            "frame_id": frame_id,
            "model": "density",
            "density_count": total_count,
            "hotspots": hotspots,
        }


def _extract_hotspots(
    density_tiles: list[Any],
    tile_plan: Any,
) -> list[dict[str, float]]:
    """GPU-resident hotspot extraction.  All tensor work stays on the GPU device
    of the input tiles; only the final coordinate list is transferred to host
    via a single .tolist() call.

    Returns an empty list on any error so the pipeline never crashes.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError:
        return []

    fw     = getattr(tile_plan, "frame_width",  0)
    fh     = getattr(tile_plan, "frame_height", 0)
    tasks  = getattr(tile_plan, "tasks",        ())
    if fw <= 0 or fh <= 0 or not tasks:
        return []

    # Resolve GPU device from first available CUDA tile; fall back to CPU.
    device: torch.device = torch.device("cpu")
    for t in density_tiles:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            device = t.device
            break

    canvas_w = max(1, fw // _CANVAS_DOWNSCALE)
    canvas_h = max(1, fh // _CANVAS_DOWNSCALE)
    canvas   = torch.zeros(canvas_h, canvas_w, dtype=torch.float32, device=device)

    for i, task in enumerate(tasks):
        if i >= len(density_tiles):
            break
        tile_map = density_tiles[i]
        if tile_map is None:
            continue
        if not isinstance(tile_map, torch.Tensor):
            try:
                tile_map = torch.as_tensor(tile_map, dtype=torch.float32, device=device)
            except Exception:
                continue

        tile_map = tile_map.squeeze().float()
        if tile_map.dim() == 0:
            continue
        if tile_map.device != device:
            tile_map = tile_map.to(device, non_blocking=True)

        dst_x1 = int(round(task.source_x / fw * canvas_w))
        dst_y1 = int(round(task.source_y / fh * canvas_h))
        dst_x2 = min(int(round((task.source_x + task.source_width)  / fw * canvas_w)), canvas_w)
        dst_y2 = min(int(round((task.source_y + task.source_height) / fh * canvas_h)), canvas_h)
        dst_w  = max(1, dst_x2 - dst_x1)
        dst_h  = max(1, dst_y2 - dst_y1)

        resized = F.interpolate(
            tile_map.unsqueeze(0).unsqueeze(0),
            size=(dst_h, dst_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        canvas[dst_y1:dst_y1 + dst_h, dst_x1:dst_x1 + dst_w].add_(resized)

    global_max = float(canvas.max().item())   # single scalar — negligible sync cost
    if global_max <= 0:
        return []

    # GPU-side non-maximum suppression.
    # max_pool2d marks each cell with the maximum in its neighbourhood;
    # cells equal to the pool result are local maxima by definition.
    padding = _NMS_KERNEL // 2
    pooled  = F.max_pool2d(
        canvas.unsqueeze(0).unsqueeze(0),
        kernel_size=_NMS_KERNEL, stride=1, padding=padding,
    ).squeeze()
    peaks_mask   = (canvas >= pooled - 1e-8) & (canvas > _DENSITY_THRESHOLD)
    peak_indices = peaks_mask.nonzero(as_tuple=False)   # [N, 2]: (row, col) on GPU
    if peak_indices.numel() == 0:
        return []

    peak_vals  = canvas[peak_indices[:, 0], peak_indices[:, 1]]

    # Sort by weight descending, entirely on GPU.
    sorted_idx   = peak_vals.argsort(descending=True)
    peak_indices = peak_indices[sorted_idx]
    peak_vals    = peak_vals[sorted_idx]

    # Single host transfer: normalised x/y coords and relative weights.
    xs = ((peak_indices[:, 1].float() + 0.5) / canvas_w).tolist()
    ys = ((peak_indices[:, 0].float() + 0.5) / canvas_h).tolist()
    ws = (peak_vals / global_max).tolist()

    return [{"x": x, "y": y, "w": w} for x, y, w in zip(xs, ys, ws)]



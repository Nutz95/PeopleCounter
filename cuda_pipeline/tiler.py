from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from tiling_manager import TilingManager


class CudaTiler:
    """Performs tiling on a CUDA tensor so downstream engines never touch the CPU."""

    def __init__(self, tile_size: int = 640, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for CudaTiler.")
        self.device = torch.device(device)
        self.tile_size = tile_size
        self._tiling_manager = TilingManager(tile_size=self.tile_size)

    def tile_frame(
        self,
        frame_tensor: torch.Tensor,
        use_tiling: bool = True,
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int, float, float, int, int]]]:
        """Return (tiles, metadata) for the provided CUDA tensor.

        Metadata entries use the same format as the CPU tiler: (x, y, scale_w, scale_h, width, height).
        """
        if frame_tensor.device != self.device:
            frame_tensor = frame_tensor.to(self.device)
        if frame_tensor.ndim != 3 or frame_tensor.shape[0] != 3:
            raise ValueError("Expected frame tensor with shape (3, H, W) on the CUDA device.")

        _, h, w = frame_tensor.shape
        tiles: List[torch.Tensor] = []
        metadata: List[Tuple[int, int, float, float, int, int]] = []

        if not use_tiling or (h <= self.tile_size and w <= self.tile_size):
            tiles.append(frame_tensor.clone())
            metadata.append((0, 0, 1.0, 1.0, w, h))
            return tiles, metadata

        tiles.append(self._resize_tensor(frame_tensor, self.tile_size, self.tile_size))
        metadata.append((0, 0, w / self.tile_size, h / self.tile_size, w, h))

        for x1, y1, x2, y2 in self._tiling_manager.get_tiles((h, w, 3)):
            tw, th = x2 - x1, y2 - y1
            crop = frame_tensor[:, y1:y2, x1:x2]
            if crop.shape[1:] != (self.tile_size, self.tile_size):
                crop = self._resize_tensor(crop, self.tile_size, self.tile_size)
            tiles.append(crop.contiguous())
            metadata.append((x1, y1, tw / self.tile_size, th / self.tile_size, tw, th))

        return tiles, metadata

    def _resize_tensor(self, tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3:
            pass
        else:
            raise ValueError("Expected a (C, H, W) tensor for resizing.")
        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return resized.contiguous()

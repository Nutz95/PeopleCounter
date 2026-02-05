from __future__ import annotations

from typing import Sequence, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class CudaPreprocessor:
    """Normalize and resize CUDA tensors before TensorRT inference."""

    def __init__(self, target_size: Tuple[int, int], device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for CudaPreprocessor.")
        self.device = torch.device(device)
        self.target_height, self.target_width = target_size

    def preprocess(self, tiles: Sequence[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
        batch = len(tiles)
        if batch == 0:
            return torch.empty((0, 3, self.target_height, self.target_width), dtype=torch.float32, device=self.device)

        tensors = []
        for tile in tiles:
            if torch.is_tensor(tile):
                tensor = tile.to(dtype=torch.float32, device=self.device)
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
            else:
                arr = np.asarray(tile, dtype=np.float32)
                tensor = torch.from_numpy(arr).to(self.device)
            tensor = tensor.clamp(min=0, max=255)
            tensor = self._ensure_chw(tensor)
            tensor = tensor * (1.0 / 255.0)
            if tensor.shape[1:] != (self.target_height, self.target_width):
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(self.target_height, self.target_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            tensors.append(tensor.contiguous())

        return torch.stack(tensors, dim=0)

    def _ensure_chw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            return tensor
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            return tensor.permute(2, 0, 1)
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        if tensor.ndim == 4:
            if tensor.shape[0] == 1 and tensor.shape[1] in (1, 3):
                return tensor.squeeze(0)
            if tensor.shape[-1] in (1, 3):
                return tensor.permute(0, 3, 1, 2).squeeze(0)
        raise RuntimeError("Unable to reshape tensor into (C, H, W) for CUDA preprocessing.")

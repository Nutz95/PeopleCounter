from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class BasePreprocessor(ABC):
    def __init__(self, target_size: Tuple[int, int]):
        self.target_height, self.target_width = target_size

    @abstractmethod
    def preprocess(self, tiles: Sequence[np.ndarray]) -> torch.Tensor:
        """Return a tensor of shape (B, 3, H, W) containing normalized values in [0, 1]."""
        raise NotImplementedError()


class CpuPreprocessor(BasePreprocessor):
    def preprocess(self, tiles: Sequence[np.ndarray]) -> torch.Tensor:
        batch = len(tiles)
        if batch == 0:
            return torch.empty((0, 3, self.target_height, self.target_width), dtype=torch.float32)

        if cv2 is None:
            raise RuntimeError("cv2 is required for CPU preprocessing but is not available.")

        result = torch.empty((batch, 3, self.target_height, self.target_width), dtype=torch.float32)
        for idx, tile in enumerate(tiles):
            resized = cv2.resize(tile, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized.astype(np.float32) * (1.0 / 255.0))
            result[idx] = tensor.permute(2, 0, 1)

        return result


class GpuPreprocessor(BasePreprocessor):
    def __init__(self, target_size: Tuple[int, int], device: str = 'cuda'):
        super().__init__(target_size)
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available for GPU preprocessing.")
        self.device = torch.device(device)

    def preprocess(self, tiles: Sequence[np.ndarray]) -> torch.Tensor:
        batch = len(tiles)
        if batch == 0:
            return torch.empty((0, 3, self.target_height, self.target_width), dtype=torch.float32, device=self.device)

        tensors = []
        for tile in tiles:
            tensor = torch.from_numpy(tile.astype(np.float32) * (1.0 / 255.0)).permute(2, 0, 1).to(self.device)
            if tensor.shape[1:] != (self.target_height, self.target_width):
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=(self.target_height, self.target_width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            tensors.append(tensor.contiguous())

        return torch.stack(tensors, dim=0)

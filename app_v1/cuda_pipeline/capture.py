from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class GpuFrameStager:
    """Utility to keep a single RGB frame resident on CUDA before downstream processing."""

    def __init__(self, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for GpuFrameStager.")
        self.device = torch.device(device)
        self._staged: Optional[torch.Tensor] = None

    def stage(self, frame: np.ndarray) -> torch.Tensor:
        """Copy the latest CPU frame to a device tensor (C, H, W) on CUDA."""
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("Expected an RGB(A) frame with 3 or 4 channels.")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        tensor = torch.from_numpy(frame).to(self.device)
        if tensor.shape[2] == 4:
            tensor = tensor[:, :, :3]
        tensor = tensor.permute(2, 0, 1).contiguous()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        self._staged = tensor
        return tensor

    def latest(self) -> Optional[torch.Tensor]:
        """Return the most recently staged tensor (or None)."""
        return self._staged

    def release(self) -> None:
        """Drop the reference and free CUDA memory if requested."""
        if self._staged is not None:
            del self._staged
            self._staged = None
            torch.cuda.empty_cache()

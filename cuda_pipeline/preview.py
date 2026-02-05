from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


@dataclass
class CudaPreviewResult:
    tensor: torch.Tensor
    frame: np.ndarray
    encoded: Optional[bytes]

    def to_payload(self) -> dict:
        return {
            "frame": self.frame,
            "encoded": self.encoded,
        }


class CudaPreviewComposer:
    def __init__(self, jpeg_quality: int = 80) -> None:
        if torch is None or F is None or not torch.cuda.is_available():
            raise EnvironmentError("CUDA preview compositor requires torch with CUDA support.")
        self.device = torch.device("cuda")
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))

    def compose(self, overlay_tensor: torch.Tensor, target_size: Tuple[int, int]) -> CudaPreviewResult:
        if torch is None or F is None:  # pragma: no cover
            raise RuntimeError("CUDA preview requires the torch functional API.")
        target_h, target_w = target_size
        if target_h <= 0 or target_w <= 0:
            raise ValueError("Target size must be positive.")

        tensor = overlay_tensor.to(device=self.device, dtype=torch.float32, non_blocking=True)
        tensor = self._normalize_axes(tensor)

        src_h, src_w = tensor.shape[1], tensor.shape[2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        padded = torch.zeros((3, target_h, target_w), dtype=resized.dtype, device=self.device)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        padded[:, y_off:y_off+new_h, x_off:x_off+new_w] = resized
        padded = torch.clamp(padded, 0, 255).to(dtype=torch.uint8)

        frame_cpu = padded.permute(1, 2, 0).contiguous().cpu().numpy()
        ret, jpeg = cv2.imencode(
            ".jpg",
            frame_cpu,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        encoded = jpeg.tobytes() if ret else None
        return CudaPreviewResult(tensor=padded, frame=frame_cpu, encoded=encoded)

    def _normalize_axes(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            return tensor
        if tensor.ndim == 3 and tensor.shape[2] in (1, 3):
            return tensor.permute(2, 0, 1)
        raise ValueError("Overlay tensor must be 3-channel with shape (C,H,W) or (H,W,C).")

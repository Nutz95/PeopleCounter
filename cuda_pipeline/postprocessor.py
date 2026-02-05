from __future__ import annotations

from typing import Optional

import torch


class CudaMaskPostprocessor:
    """Rebuilds mask fragments on CUDA directly from the YOLO prototypes."""

    def __init__(self, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for CudaMaskPostprocessor.")
        self.device = torch.device(device)

    def build_mask_fragment(
        self,
        coeffs,
        protos,
        ix1: float,
        iy1: float,
        ix2: float,
        iy2: float,
        target_w: int,
        target_h: int,
    ) -> Optional[torch.Tensor]:
        if coeffs is None or protos is None:
            return None

        if isinstance(coeffs, torch.Tensor):
            coeffs_tensor = coeffs.to(dtype=torch.float32, device=self.device).flatten()
        else:
            coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32, device=self.device).flatten()

        if coeffs_tensor.numel() == 0:
            return None

        proto_tensor = torch.tensor(protos, dtype=torch.float32, device=self.device)
        if proto_tensor.ndim != 3:
            raise ValueError("Prototype tensor must have shape (C, H, W).")

        ph, pw = proto_tensor.shape[1], proto_tensor.shape[2]
        scale_w = pw / target_w
        scale_h = ph / target_h

        mx1 = max(0, min(pw, int(ix1 * scale_w)))
        my1 = max(0, min(ph, int(iy1 * scale_h)))
        mx2 = max(0, min(pw, int(ix2 * scale_w)))
        my2 = max(0, min(ph, int(iy2 * scale_h)))

        if mx2 <= mx1 or my2 <= my1:
            return None

        crop = proto_tensor[:, my1:my2, mx1:mx2]
        if crop.numel() == 0:
            return None

        area = crop.shape[1] * crop.shape[2]
        proto_flat = crop.reshape(proto_tensor.shape[0], -1)
        logits = torch.matmul(coeffs_tensor.unsqueeze(0), proto_flat)
        mask = torch.sigmoid(logits).reshape(my2 - my1, mx2 - mx1)
        return mask

from __future__ import annotations

from enum import StrEnum


class TensorMemoryFormat(StrEnum):
    """Canonical memory layouts for preprocess output tensors."""

    NV12 = "NV12"
    RGB_HWC_U8 = "RGB_HWC_U8"
    RGB_NCHW_FP16 = "RGB_NCHW_FP16"

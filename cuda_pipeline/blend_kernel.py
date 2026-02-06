from __future__ import annotations

import logging
import math
import os

import torch
from torch.utils.cpp_extension import load_inline

_BLEND_MODULE = None
_TORCH_ARCH_ENV_KEY = "TORCH_CUDA_ARCH_LIST"
_DEFAULT_CUDA_ARCH = os.environ.get("PEOPLE_COUNTER_CUDA_ARCH_LIST", "8.6")

_KERNEL_SOURCE = r"""
#include <cuda_runtime.h>

extern "C" __global__ void blend_roi(
    float* image,
    const unsigned char* mask,
    int channel_stride,
    int width,
    int y_offset,
    int x_offset,
    int roi_h,
    int roi_w,
    float alpha,
    float beta,
    const float* color
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= roi_w || py >= roi_h) {
        return;
    }
    int mask_idx = py * roi_w + px;
    if (mask[mask_idx] == 0) {
        return;
    }
    int y = y_offset + py;
    int x = x_offset + px;
    int base_idx = y * width + x;
    for (int channel = 0; channel < 3; ++channel) {
        int idx = channel * channel_stride + base_idx;
        float orig = image[idx];
        float blended = orig * alpha + color[channel] * beta;
        image[idx] = blended;
    }
}
"""


def _ensure_cuda_arch_list() -> None:
    if _TORCH_ARCH_ENV_KEY in os.environ and os.environ[_TORCH_ARCH_ENV_KEY].strip():
        return
    os.environ[_TORCH_ARCH_ENV_KEY] = _DEFAULT_CUDA_ARCH


def _get_blend_module():
    global _BLEND_MODULE
    if _BLEND_MODULE is not None:
        return _BLEND_MODULE
    _ensure_cuda_arch_list()
    logger = logging.getLogger(__name__)
    torch_logger = logging.getLogger("torch")
    prev_level = torch_logger.level
    torch_logger.setLevel(logging.ERROR)
    try:
        module = load_inline(
            name="people_counter_blend",
            cpp_sources="",
            cuda_sources=[_KERNEL_SOURCE],
            functions=["blend_roi"],
            extra_cuda_cflags=["-O3"],
        )
        _BLEND_MODULE = module
        return module
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("CUDA blend module unavailable: %s", exc)
        return None
    finally:
        torch_logger.setLevel(prev_level)


def blend_roi_on_cuda(image: torch.Tensor, mask_slice: torch.Tensor, y_offset: int, x_offset: int, overlay_color: torch.Tensor) -> None:
    if not torch.cuda.is_available() or image.device.type != "cuda":
        return
    if mask_slice.numel() == 0:
        return
    roi_h, roi_w = mask_slice.shape
    mask_slice = mask_slice.contiguous()
    overlay_color = overlay_color.to(dtype=torch.float32)
    module = _get_blend_module()
    channel_stride = image.shape[1] * image.shape[2]
    width = image.shape[2]
    threads = (16, 16, 1)
    blocks = (
        math.ceil(roi_w / threads[0]),
        math.ceil(roi_h / threads[1]),
        1,
    )
    if module is None:
        return
    module.blend_roi(
        image,
        mask_slice,
        channel_stride,
        width,
        y_offset,
        x_offset,
        roi_h,
        roi_w,
        0.6,
        0.4,
        overlay_color,
        grid=blocks,
        block=threads,
    )

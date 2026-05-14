"""
Optimized CUDA kernel for hotspot circle rasterization using CUDA C++ / cupy (optional).

This is a fallback in case Triton is not available.
"""

from __future__ import annotations

from typing import Callable

import torch


def _get_cuda_circle_kernel() -> Callable | None:
    """
    Try to compile a PyTorch custom CUDA kernel for circle rasterization.
    Returns a callable or None if compilation fails.
    """
    try:
        from torch.utils.cpp_extension import load_inline

        cuda_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void draw_circles_kernel(
            uint8_t* frame,
            int frame_h,
            int frame_w,
            const float* centers,
            int n_centers,
            int radius,
            uint8_t color_r,
            uint8_t color_g,
            uint8_t color_b
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n_centers) return;

            float cx_norm = centers[idx * 3];
            float cy_norm = centers[idx * 3 + 1];
            int cx = (int)(cx_norm * frame_w + 0.5f);
            int cy = (int)(cy_norm * frame_h + 0.5f);

            int r_sq = radius * radius;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int dist_sq = dx * dx + dy * dy;
                    if (dist_sq <= r_sq) {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < frame_w && py >= 0 && py < frame_h) {
                            int pixel_offset = (py * frame_w + px);
                            int r_offset = pixel_offset;
                            int g_offset = pixel_offset + frame_h * frame_w;
                            int b_offset = pixel_offset + 2 * frame_h * frame_w;
                            frame[r_offset] = color_r;
                            frame[g_offset] = color_g;
                            frame[b_offset] = color_b;
                        }
                    }
                }
            }
        }

        torch::Tensor draw_circles_cuda(
            torch::Tensor frame,
            torch::Tensor centers,
            int radius,
            int color_r,
            int color_g,
            int color_b
        ) {
            int n_centers = centers.size(0);
            int frame_h = frame.size(1);
            int frame_w = frame.size(2);

            int threads = 256;
            int blocks = (n_centers + threads - 1) / threads;

            draw_circles_kernel<<<blocks, threads>>>(
                frame.data_ptr<uint8_t>(),
                frame_h,
                frame_w,
                centers.data_ptr<float>(),
                n_centers,
                radius,
                (uint8_t)color_r,
                (uint8_t)color_g,
                (uint8_t)color_b
            );

            cudaDeviceSynchronize();
            return frame;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("draw_circles", &draw_circles_cuda, "Draw circles on GPU frame");
        }
        """

        cpp_source = """
        #include <torch/extension.h>
        torch::Tensor draw_circles_cuda(
            torch::Tensor frame,
            torch::Tensor centers,
            int radius,
            int color_r,
            int color_g,
            int color_b
        );
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("draw_circles", &draw_circles_cuda, "Draw circles on GPU frame");
        }
        """

        module = load_inline(
            name="draw_circles_cuda",
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=["draw_circles"],
            verbose=False,
            build_directory="/tmp/torch_cuda_hotspot",
        )

        def wrapper(frame: torch.Tensor, centers: torch.Tensor, radius: int = 3) -> torch.Tensor:
            """Wrapper for CUDA circle kernel."""
            if centers.numel() == 0:
                return frame
            # Ensure contiguous
            frame = frame.contiguous()
            centers_normalized = centers.float().contiguous()
            return module.draw_circles(frame, centers_normalized, radius, 220, 30, 30)

        return wrapper
    except Exception as e:
        # Fallback: return None to use PyTorch strategy
        return None

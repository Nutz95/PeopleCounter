import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tiling_manager import TilingManager
from engines.yolo.tensorrt_engine import YoloTensorRTEngine
from engines.yolo.openvino_engine import YoloOpenVINOEngine
from engines.yolo.ultralytics_engine import YoloUltralyticsEngine
from engines.yolo.renderers import CpuMaskRenderer, GpuMaskRenderer

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

class YoloSegPeopleCounter:
    def __init__(self, model_path, confidence_threshold=0.25, backend='tensorrt_native', device='cuda'):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        self.device = device
        self.person_class_id = 0
        self.tiler = TilingManager(tile_size=640)
        # On augmente le nombre de workers pour couvrir les 28 tuiles du mode 4K en parallèle
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.last_perf = {}
        self.last_internal = {}
        self.last_device = "Unknown"
        self.debug_mode = False
        self.pipeline_mode = os.environ.get('YOLO_PIPELINE_MODE', 'auto').lower()
        self._gpu_render_flag = os.environ.get('YOLO_USE_GPU_RENDER', '0') == '1'
        self._gpu_available = bool(torch is not None and torch.cuda.is_available())
        self.preprocessor_mode = 'cpu'
        self.renderer_mode = 'cpu'
        self.renderer = CpuMaskRenderer()
        self.active_pipeline_mode = 'cpu'

        if self.backend == 'tensorrt_native':
            use_gpu_pre = os.environ.get('YOLO_USE_GPU_PREPROC', '0') == '1'
            use_gpu_post = os.environ.get('YOLO_USE_GPU_POST', '0') == '1'
            self.engine = YoloTensorRTEngine(model_path, confidence_threshold, device, self.pool, use_gpu_preproc=use_gpu_pre, use_gpu_post=use_gpu_post)
            self.last_device = "GPU (TensorRT)"
        elif self.backend == 'openvino_native':
            self.engine = YoloOpenVINOEngine(model_path, confidence_threshold, device)
            self.last_device = getattr(self.engine, 'device_name', "OpenVINO (GPU)")
        else:
            self.engine = YoloUltralyticsEngine(model_path, confidence_threshold, device)
            self.last_device = "PyTorch (CPU/GPU)"

        self.configure_pipeline(self.pipeline_mode)

    def count_people(self, image, tile_size=640, draw_boxes=True, use_tiling=True, draw_tiles=False):
        t0_total = time.time()
        image_rgb = image[..., ::-1].copy()
        h, w = image_rgb.shape[:2]

        # --- Crop / tiles ---
        t_crop_start = time.time()
        tiles, metadata = self._prepare_tiles(image_rgb, use_tiling)
        t_crop = time.time() - t_crop_start

        # --- Inference ---
        t_inf_start = time.time()
        results = self.engine.infer(tiles, metadata)
        if len(results) == 4:
            all_boxes, all_scores, all_masks, all_visible_boxes = results
        else:
            all_boxes, all_scores, all_masks = results
            all_visible_boxes = all_boxes
        t_inf = time.time() - t_inf_start

        self.last_perf = self.engine.get_perf()

        if all_boxes.size == 0:
            final_img = image.copy()
            return (0, final_img) if draw_boxes else 0

        # --- Fusion ---
        t_fuse_start = time.time()
        clusters = self.tiler.fuse_results(all_boxes, all_scores, all_masks, iou_threshold=0.3)
        total_count = len(clusters)
        t_fuse = time.time() - t_fuse_start

        # --- Draw (optimized combined mask) ---
        if self.debug_mode:
            print(f"[DEBUG] YOLO pipeline tiles={len(tiles)} preproc={self.preprocessor_mode} renderer={self.renderer_mode}")
        t_draw_start = time.time()
        if draw_boxes:
            image_out = self.renderer.render(
                image.copy(),
                clusters,
                all_boxes,
                all_masks,
                all_visible_boxes,
                metadata,
                draw_boxes,
                self.debug_mode,
            )
        else:
            image_out = image.copy()
        t_draw = time.time() - t_draw_start

        # On log le détail si on est en décalage
        total_internal = t_crop + t_inf + t_fuse + t_draw
        self.last_internal = {
            't_crop': t_crop,
            't_inf': t_inf,
            't_fuse': t_fuse,
            't_draw': t_draw,
            'total_internal': total_internal
        }
        print(f"  [YOLO DETAILED] Crop={t_crop*1000:.1f}ms | Infer={t_inf*1000:.1f}ms | Fuse={t_fuse*1000:.1f}ms | Draw={t_draw*1000:.1f}ms | Total={total_internal*1000:.1f}ms")

        return int(total_count), image_out if draw_boxes else int(total_count)

    def set_debug_mode(self, enabled):
        self.debug_mode = bool(enabled)

    def configure_pipeline(self, mode='auto'):
        normalized = (mode or 'auto').lower()
        resolved = self._resolve_mode(normalized)
        self.pipeline_mode = normalized
        preproc_mode = 'gpu' if resolved in ('gpu', 'gpu_full') else 'cpu'
        if hasattr(self.engine, 'set_preprocessor_mode'):
            self.engine.set_preprocessor_mode(preproc_mode)
        self.preprocessor_mode = preproc_mode
        renderer_mode = 'gpu' if resolved in ('gpu', 'gpu_full') else 'cpu'
        self._set_renderer_mode(renderer_mode)
        self.active_pipeline_mode = resolved
        return resolved

    def _resolve_mode(self, mode):
        if mode == 'gpu_full' and self._gpu_available:
            return 'gpu_full'
        if mode == 'gpu' and self._gpu_available:
            return 'gpu'
        if mode == 'auto':
            if self._gpu_render_flag and self._gpu_available:
                return 'gpu'
            gpu_pre = getattr(self.engine, 'use_gpu_preproc', False)
            if gpu_pre and self._gpu_available:
                return 'gpu'
            return 'cpu'
        if mode == 'cpu':
            return 'cpu'
        return 'cpu'

    def _set_renderer_mode(self, mode):
        if mode == 'gpu' and self._gpu_available:
            try:
                self.renderer = GpuMaskRenderer()
                self.renderer_mode = 'gpu'
                return
            except EnvironmentError:
                pass
        self.renderer = CpuMaskRenderer()
        self.renderer_mode = 'cpu'

    def _prepare_tiles(self, image_rgb, use_tiling):
        if self._is_full_gpu_pipeline():
            try:
                return self._prepare_tiles_gpu(image_rgb, use_tiling)
            except Exception as exc:
                if self.debug_mode:
                    print(f"[WARN] Full GPU tiling failed, falling back to CPU tiles: {exc}")
        return self._prepare_tiles_cpu(image_rgb, use_tiling)

    def _prepare_tiles_cpu(self, image_rgb, use_tiling):
        tiles = []
        metadata = []  # (x, y, sw, sh, orig_w, orig_h)
        h, w = image_rgb.shape[:2]
        tile_size = self.tiler.tile_size

        if not use_tiling or (h <= tile_size and w <= tile_size):
            tiles.append(image_rgb)
            metadata.append((0, 0, 1.0, 1.0, w, h))
            return tiles, metadata

        tiles.append(cv2.resize(image_rgb, (tile_size, tile_size)))
        metadata.append((0, 0, w / tile_size, h / tile_size, w, h))

        tiles_coords = self.tiler.get_tiles(image_rgb.shape)
        for x1, y1, x2, y2 in tiles_coords:
            tw, th = x2 - x1, y2 - y1
            tiles.append(image_rgb[y1:y2, x1:x2])
            metadata.append((x1, y1, tw / tile_size, th / tile_size, tw, th))

        return tiles, metadata

    def _prepare_tiles_gpu(self, image_rgb, use_tiling):
        if torch is None or F is None or not torch.cuda.is_available():
            raise EnvironmentError("CUDA resources unavailable for full GPU tiling.")
        tile_size = self.tiler.tile_size
        h, w = image_rgb.shape[:2]
        device = torch.device('cuda')
        frame_tensor = torch.from_numpy(image_rgb.astype(np.float32)).permute(2, 0, 1).contiguous().to(device)
        tiles = []
        metadata = []

        if not use_tiling or (h <= tile_size and w <= tile_size):
            tiles.append(frame_tensor.clone())
            metadata.append((0, 0, 1.0, 1.0, w, h))
            return tiles, metadata

        tiles.append(self._resize_tensor(frame_tensor, tile_size, tile_size))
        metadata.append((0, 0, w / tile_size, h / tile_size, w, h))

        tiles_coords = self.tiler.get_tiles(image_rgb.shape)
        for x1, y1, x2, y2 in tiles_coords:
            crop = frame_tensor[:, y1:y2, x1:x2]
            if crop.shape[1:] != (tile_size, tile_size):
                crop = self._resize_tensor(crop, tile_size, tile_size)
            tiles.append(crop.contiguous())
            tw, th = x2 - x1, y2 - y1
            metadata.append((x1, y1, tw / tile_size, th / tile_size, tw, th))

        return tiles, metadata

    def _resize_tensor(self, tensor, height, width):
        if F is None:
            raise RuntimeError("Torch functional API is required to resize CUDA tensors.")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        resized = F.interpolate(
            tensor,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        return resized.contiguous()

    def _is_full_gpu_pipeline(self):
        return self.active_pipeline_mode == 'gpu_full' and self._gpu_available


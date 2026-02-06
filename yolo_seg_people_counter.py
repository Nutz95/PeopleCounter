import base64
import copy
import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tiling_manager import TilingManager
from engines.yolo.tensorrt_engine import YoloTensorRTEngine
from engines.yolo.openvino_engine import YoloOpenVINOEngine
from engines.yolo.ultralytics_engine import YoloUltralyticsEngine

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from cuda_pipeline.capture import GpuFrameStager
    from cuda_pipeline.tiler import CudaTiler
except Exception:
    GpuFrameStager = None
    CudaTiler = None

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
        self.active_pipeline_mode = 'cpu'
        self._cuda_helpers_initialized = False
        self._cuda_frame_stager = None
        self._cuda_tiler = None
        self.last_pipeline_report = {}
        self.last_detection_payload = {'clusters': [], 'detections': []}
        self.mask_downscale = max(1, int(os.environ.get('YOLO_MASK_DOWNSCALE', '4')))
        self.mask_format = os.environ.get('YOLO_MASK_FORMAT', 'webp').lower()
        self.mask_quality = min(100, max(10, int(os.environ.get('YOLO_MASK_QUALITY', '60'))))
        self.last_mask_payload = None

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

    def count_people(self, image, tile_size=640, draw_boxes=True, use_tiling=True, draw_tiles=False, global_tile_only=False):
        t0_total = time.time()
        image_rgb = image[..., ::-1].copy()
        h, w = image_rgb.shape[:2]

        # --- Crop / tiles ---
        t_crop_start = time.time()
        tiles, metadata = self._prepare_tiles(image_rgb, use_tiling, global_tile_only)
        t_crop = time.time() - t_crop_start

        # --- Inference ---
        t_inf_start = time.time()
        results = self.engine.infer(tiles, metadata)
        detection_metadata = None
        if len(results) == 5:
            all_boxes, all_scores, all_masks, all_visible_boxes, detection_metadata = results
        elif len(results) == 4:
            all_boxes, all_scores, all_masks, all_visible_boxes = results
        else:
            all_boxes, all_scores, all_masks = results[:3]
            all_visible_boxes = all_boxes
        t_inf = time.time() - t_inf_start

        self.last_perf = self.engine.get_perf()

        # --- Fusion ---
        t_fuse_start = time.time()
        clusters = self.tiler.fuse_results(all_boxes, all_scores, all_masks, iou_threshold=0.3)
        total_count = len(clusters)
        self._capture_detection_payload(clusters, all_boxes, all_scores, all_masks, all_visible_boxes)
        image_out = image.copy()
        self.last_mask_payload = self._build_mask_payload(all_boxes, all_masks, all_visible_boxes, (h, w))
        t_fuse = time.time() - t_fuse_start

        # --- Draw (optimized combined mask) ---
        if self.debug_mode:
            print(f"[DEBUG] YOLO pipeline tiles={len(tiles)} preproc={self.preprocessor_mode}")
        t_draw_start = time.time()
        image_out = image.copy()
        if draw_tiles:
            image_out = self._apply_debug_tiling_overlay(
                image_out,
                draw_tiles,
                metadata,
                detection_metadata,
                all_masks,
                all_visible_boxes,
            )
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

        return int(total_count), image_out

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
        self.renderer_mode = 'cpu'
        self.active_pipeline_mode = resolved
        self.last_pipeline_report = self._build_pipeline_report()
        return resolved

    def _resolve_mode(self, mode):
        if mode == 'gpu_full' and self._has_full_cuda_support():
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

    def _has_full_cuda_support(self):
        return (
            bool(GpuFrameStager and CudaTiler)
            and self._gpu_available
        )

    def _ensure_cuda_helpers(self):
        if self._cuda_helpers_initialized:
            return True
        if not self._has_full_cuda_support():
            return False
        try:
            self._cuda_frame_stager = GpuFrameStager()
            self._cuda_tiler = CudaTiler(tile_size=self.tiler.tile_size)
        except EnvironmentError as exc:
            if self.debug_mode:
                print(f"[WARN] CUDA helpers unavailable: {exc}")
            return False
        self._cuda_helpers_initialized = True
        return True


    def get_detection_payload(self):
        return copy.deepcopy(self.last_detection_payload)

    def _capture_detection_payload(self, clusters, all_boxes, all_scores, all_masks, all_visible_boxes):
        if not len(all_boxes):
            self.last_detection_payload = {'clusters': [], 'detections': []}
            return

        detections = []
        for idx in range(len(all_boxes)):
            bbox = [float(v) for v in all_boxes[idx]]
            visible_box = [float(v) for v in all_visible_boxes[idx]]
            mask_present = all_masks[idx] is not None
            detections.append({
                'index': int(idx),
                'confidence': float(all_scores[idx]) if idx < len(all_scores) else 0.0,
                'box': bbox,
                'visible_box': visible_box,
                'has_mask': bool(mask_present),
            })

        clusters_summary = []
        for cluster_idx, cluster in enumerate(clusters):
            if not cluster:
                continue
            cluster_boxes = [all_boxes[i] for i in cluster if i < len(all_boxes)]
            if not cluster_boxes:
                continue
            min_x = min(box[0] for box in cluster_boxes)
            min_y = min(box[1] for box in cluster_boxes)
            max_x = max(box[2] for box in cluster_boxes)
            max_y = max(box[3] for box in cluster_boxes)
            mask_count = int(sum(1 for i in cluster if i < len(all_masks) and all_masks[i] is not None))
            clusters_summary.append({
                'id': int(cluster_idx),
                'members': [int(i) for i in cluster],
                'bbox': [float(min_x), float(min_y), float(max_x), float(max_y)],
                'mask_members': mask_count,
            })

        self.last_detection_payload = {
            'clusters': clusters_summary,
            'detections': detections,
        }

    def get_mask_payload(self):
        if not self.last_mask_payload:
            return None
        return dict(self.last_mask_payload)

    def _build_mask_payload(self, all_boxes, all_masks, all_visible_boxes, frame_shape):
        mask_full = self._compose_full_mask(all_boxes, all_masks, all_visible_boxes, frame_shape)
        if mask_full is None or mask_full.sum() == 0:
            return None
        mask_small = self._downscale_mask(mask_full)
        encoded = self._encode_mask_blob(mask_small)
        if encoded is None:
            return None
        payload = {
            'blob': base64.b64encode(encoded).decode('ascii'),
            'format': self.mask_format,
            'width': mask_small.shape[1],
            'height': mask_small.shape[0],
            'scale_x': frame_shape[1] / mask_small.shape[1] if mask_small.shape[1] else 1.0,
            'scale_y': frame_shape[0] / mask_small.shape[0] if mask_small.shape[0] else 1.0,
            'downscale': self.mask_downscale,
        }
        return payload

    def _compose_full_mask(self, all_boxes, all_masks, all_visible_boxes, frame_shape):
        h, w = frame_shape
        if h <= 0 or w <= 0:
            return None
        mask_full = np.zeros((h, w), dtype=np.uint8)
        if not all_masks:
            return mask_full
        use_visible = len(all_visible_boxes) == len(all_masks)
        fallback_boxes = all_boxes if len(all_boxes) == len(all_masks) else None
        for idx, mask_fragment in enumerate(all_masks):
            normalized = self._normalize_mask_fragment(mask_fragment)
            if normalized is None:
                continue
            bbox = all_visible_boxes[idx] if use_visible else (fallback_boxes[idx] if fallback_boxes else None)
            if not bbox:
                continue
            vx1, vy1, vx2, vy2 = [float(v) for v in bbox]
            vx1, vy1, vx2, vy2 = self._clip_bbox(vx1, vy1, vx2, vy2, frame_shape)
            roi_w = int(round(vx2 - vx1))
            roi_h = int(round(vy2 - vy1))
            if roi_w <= 0 or roi_h <= 0:
                continue
            mask_resized = cv2.resize(normalized, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
            intersect_x1 = int(max(0, vx1))
            intersect_y1 = int(max(0, vy1))
            intersect_x2 = int(min(w, vx2))
            intersect_y2 = int(min(h, vy2))
            if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
                continue
            mask_shape_y, mask_shape_x = mask_bin.shape
            mask_x1 = max(0, int(intersect_x1 - vx1))
            mask_y1 = max(0, int(intersect_y1 - vy1))
            if mask_x1 >= mask_shape_x or mask_y1 >= mask_shape_y:
                continue
            width_target = intersect_x2 - intersect_x1
            height_target = intersect_y2 - intersect_y1
            available_w = mask_shape_x - mask_x1
            available_h = mask_shape_y - mask_y1
            draw_w = min(width_target, available_w)
            draw_h = min(height_target, available_h)
            if draw_w <= 0 or draw_h <= 0:
                continue
            mask_x2 = mask_x1 + draw_w
            mask_y2 = mask_y1 + draw_h
            intersect_x2 = intersect_x1 + draw_w
            intersect_y2 = intersect_y1 + draw_h
            mask_slice = mask_bin[mask_y1:mask_y2, mask_x1:mask_x2]
            if mask_slice.size == 0:
                continue
            target = mask_full[intersect_y1:intersect_y2, intersect_x1:intersect_x2]
            mask_full[intersect_y1:intersect_y2, intersect_x1:intersect_x2] = np.maximum(target, mask_slice)
        return mask_full

    def _normalize_mask_fragment(self, fragment):
        if fragment is None:
            return None
        if torch is not None and torch.is_tensor(fragment):
            if fragment.numel() == 0:
                return None
            array = fragment.detach().cpu().numpy()
        else:
            array = np.asarray(fragment)
        if array.ndim > 2:
            if array.shape[0] in (1,) and array.ndim == 3:
                array = array.squeeze(0)
            elif array.shape[-1] == 1:
                array = array[..., 0]
        if array.size == 0:
            return None
        return array.astype(np.float32, copy=False)

    def _downscale_mask(self, mask_full):
        if self.mask_downscale <= 1:
            return mask_full
        target_h = max(1, mask_full.shape[0] // self.mask_downscale)
        target_w = max(1, mask_full.shape[1] // self.mask_downscale)
        return cv2.resize(mask_full, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def _encode_mask_blob(self, mask_small, fmt_override=None):
        fmt = fmt_override if fmt_override in ('webp', 'png') else (self.mask_format if self.mask_format in ('webp', 'png') else 'png')
        params = []
        if fmt == 'webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, self.mask_quality]
        elif fmt == 'png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        success, encoded = cv2.imencode(f'.{fmt}', mask_small, params)
        if not success:
            if fmt != 'png':
                return self._encode_mask_blob(mask_small, fmt_override='png')
            return None
        return encoded.tobytes()

    def _clip_bbox(self, x1, y1, x2, y2, frame_shape):
        h, w = frame_shape
        return (
            max(0, min(w, x1)),
            max(0, min(h, y1)),
            max(0, min(w, x2)),
            max(0, min(h, y2)),
        )

    def _prepare_tiles(self, image_rgb, use_tiling, global_tile_only=False):
        if self._is_full_gpu_pipeline():
            try:
                return self._prepare_tiles_gpu(image_rgb, use_tiling, global_tile_only)
            except Exception as exc:
                if self.debug_mode:
                    print(f"[WARN] Full GPU tiling failed, falling back to CPU tiles: {exc}")
                self.last_pipeline_report = self._build_pipeline_report(last_gpu_full_success=False, last_gpu_full_error=str(exc))
        return self._prepare_tiles_cpu(image_rgb, use_tiling, global_tile_only)

    def _prepare_tiles_cpu(self, image_rgb, use_tiling, global_tile_only=False):
        tiles = []
        metadata = []  # (x, y, sw, sh, orig_w, orig_h)
        h, w = image_rgb.shape[:2]
        tile_size = self.tiler.tile_size

        if global_tile_only:
            tiles.append(cv2.resize(image_rgb, (tile_size, tile_size)))
            metadata.append((0, 0, w / tile_size, h / tile_size, w, h, True))
            return tiles, metadata

        if not use_tiling or (h <= tile_size and w <= tile_size):
            tiles.append(image_rgb)
            metadata.append((0, 0, 1.0, 1.0, w, h, True))
            return tiles, metadata

        tiles.append(cv2.resize(image_rgb, (tile_size, tile_size)))
        metadata.append((0, 0, w / tile_size, h / tile_size, w, h, True))

        tiles_coords = self.tiler.get_tiles(image_rgb.shape)
        for x1, y1, x2, y2 in tiles_coords:
            tw, th = x2 - x1, y2 - y1
            tiles.append(image_rgb[y1:y2, x1:x2])
            metadata.append((x1, y1, tw / tile_size, th / tile_size, tw, th, False))

        return tiles, metadata

    def _prepare_tiles_gpu(self, image_rgb, use_tiling, global_tile_only=False):
        if not self._ensure_cuda_helpers():
            raise EnvironmentError("CUDA tiling helpers could not be initialized.")

        frame_tensor = self._cuda_frame_stager.stage(image_rgb)
        try:
            tiles, metadata = self._cuda_tiler.tile_frame(frame_tensor, use_tiling, global_tile_only)
            self.last_pipeline_report = self._build_pipeline_report(last_gpu_full_success=True)
            return tiles, metadata
        finally:
            self._cuda_frame_stager.release()

    def get_pipeline_report(self):
        return dict(self.last_pipeline_report)

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

    def _apply_debug_tiling_overlay(self, image, draw_tiles, tile_metadata, detection_metadata, all_masks, all_visible_boxes):
        if not draw_tiles or image is None:
            return image

        if torch is not None and torch.is_tensor(image):
            return self._draw_cuda_debug_overlay(image, tile_metadata, detection_metadata, all_masks, all_visible_boxes)

        def _draw(canvas):
            self._draw_tile_grid(canvas, tile_metadata)
            self._draw_main_tile_segments(canvas, all_masks, all_visible_boxes, detection_metadata)

        return self._render_debug_canvas(image, _draw)

    def _render_debug_canvas(self, image, draw_callback):
        if image is None:
            return image
        using_tensor = torch is not None and torch.is_tensor(image)
        if using_tensor:
            device = image.device
            canvas = image.detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8, copy=True)
        else:
            canvas = image.copy()
            if canvas.dtype != np.uint8:
                canvas = canvas.astype(np.uint8, copy=False)
        draw_callback(canvas)
        canvas = np.ascontiguousarray(canvas)
        if using_tensor:
            return torch.from_numpy(canvas).permute(2, 0, 1).to(device)
        return canvas

    def _draw_tile_grid(self, canvas, metadata):
        if not metadata:
            return
        h_img, w_img = canvas.shape[:2]
        color = (0, 165, 255)
        thickness = 2
        for entry in metadata:
            if not entry or len(entry) < 6:
                continue
            x, y, _, _, tw, th = entry[:6]
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + tw))
            y2 = int(round(y + th))
            x1 = max(0, min(x1, w_img))
            y1 = max(0, min(y1, h_img))
            x2 = max(0, min(x2, w_img))
            y2 = max(0, min(y2, h_img))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def _draw_main_tile_segments(self, canvas, all_masks, all_visible_boxes, detection_metadata):
        if not all_masks or detection_metadata is None or not all_visible_boxes:
            return
        h_img, w_img = canvas.shape[:2]
        for idx, mask_fragment in enumerate(all_masks):
            if idx >= len(detection_metadata):
                break
            meta = detection_metadata[idx]
            if not meta or len(meta) < 7 or not meta[6]:
                continue
            if idx >= len(all_visible_boxes):
                continue
            vx1, vy1, vx2, vy2 = [int(round(v)) for v in all_visible_boxes[idx]]
            vx1 = max(0, min(vx1, w_img))
            vy1 = max(0, min(vy1, h_img))
            vx2 = max(0, min(vx2, w_img))
            vy2 = max(0, min(vy2, h_img))
            fw = vx2 - vx1
            fh = vy2 - vy1
            if fw <= 0 or fh <= 0:
                continue
            mask_arr = mask_fragment
            if torch is not None and torch.is_tensor(mask_arr):
                if mask_arr.numel() == 0:
                    continue
                mask_arr = mask_arr.detach().cpu().numpy()
            else:
                if mask_arr is None:
                    continue
                mask_arr = np.asarray(mask_arr)
            if mask_arr.size == 0:
                continue
            mask_arr = mask_arr.astype(np.float32, copy=False)
            try:
                mask_resized = cv2.resize(mask_arr, (fw, fh), interpolation=cv2.INTER_LINEAR)
            except Exception:
                continue
            mask_bool = mask_resized > 0.5
            if not mask_bool.any():
                continue
            roi = canvas[vy1:vy2, vx1:vx2]
            overlay = roi.copy()
            overlay[mask_bool] = (0, 0, 255)
            blended = cv2.addWeighted(overlay, 0.4, roi, 0.6, 0)
            roi[mask_bool] = blended[mask_bool]
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(roi, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_cuda_debug_overlay(self, tensor, tile_metadata, detection_metadata, all_masks, all_visible_boxes):
        if tensor is None or tensor.ndim not in (3, 4):
            return tensor
        canvas = tensor
        squeeze_first = False
        if canvas.ndim == 4:
            if canvas.shape[0] == 1:
                canvas = canvas[0]
                squeeze_first = True
            else:
                canvas = canvas.squeeze(0)
        if canvas.ndim != 3:
            return tensor
        device = canvas.device
        dtype = canvas.dtype
        self._cuda_draw_tile_grid(canvas, tile_metadata)
        self._cuda_draw_global_tile_segments(canvas, detection_metadata, all_masks, all_visible_boxes)
        if squeeze_first:
            return canvas.unsqueeze(0)
        if tensor.ndim == 4:
            return canvas.unsqueeze(0)
        return canvas

    def _cuda_draw_tile_grid(self, canvas, metadata):
        if not metadata or canvas.ndim != 3:
            return
        h_img, w_img = canvas.shape[1], canvas.shape[2]
        color_vec = torch.tensor((0, 165, 255), dtype=canvas.dtype, device=canvas.device)
        thickness = 2
        for entry in metadata:
            if not entry or len(entry) < 6:
                continue
            x, y, _, _, tw, th = entry[:6]
            x1 = max(0, min(int(round(x)), w_img))
            y1 = max(0, min(int(round(y)), h_img))
            x2 = max(0, min(int(round(x + tw)), w_img))
            y2 = max(0, min(int(round(y + th)), h_img))
            if x2 <= x1 or y2 <= y1:
                continue
            for t in range(thickness):
                y_top = y1 + t
                y_bot = y2 - 1 - t
                if 0 <= y_top < h_img:
                    canvas[:, y_top, x1:x2] = color_vec[:, None]
                if 0 <= y_bot < h_img and y_bot != y_top:
                    canvas[:, y_bot, x1:x2] = color_vec[:, None]
                x_left = x1 + t
                x_right = x2 - 1 - t
                if 0 <= x_left < w_img:
                    canvas[:, y1:y2, x_left] = color_vec[:, None]
                if 0 <= x_right < w_img and x_right != x_left:
                    canvas[:, y1:y2, x_right] = color_vec[:, None]

    def _cuda_draw_global_tile_segments(self, canvas, detection_metadata, all_masks, all_visible_boxes):
        if (not detection_metadata) or (not all_masks) or (not all_visible_boxes):
            return
        device = canvas.device
        dtype = canvas.dtype
        alpha = 0.4
        color = torch.tensor((0.0, 0.0, 255.0), dtype=torch.float32, device=device).view(3, 1, 1)
        mask_overlay = torch.zeros((canvas.shape[1], canvas.shape[2]), dtype=torch.bool, device=device)
        for idx, mask_fragment in enumerate(all_masks):
            if idx >= len(detection_metadata) or idx >= len(all_visible_boxes):
                break
            meta = detection_metadata[idx]
            if not meta or len(meta) < 7 or not meta[6]:
                continue
            vx1, vy1, vx2, vy2 = [int(round(v)) for v in all_visible_boxes[idx]]
            vx1 = max(0, min(vx1, canvas.shape[2]))
            vy1 = max(0, min(vy1, canvas.shape[1]))
            vx2 = max(0, min(vx2, canvas.shape[2]))
            vy2 = max(0, min(vy2, canvas.shape[1]))
            fw = vx2 - vx1
            fh = vy2 - vy1
            if fw <= 0 or fh <= 0:
                continue
            mask_tensor = self._mask_fragment_to_tensor(mask_fragment, device)
            if mask_tensor is None:
                continue
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            if mask_tensor.shape[-2] != fh or mask_tensor.shape[-1] != fw:
                mask_tensor = F.interpolate(mask_tensor, size=(fh, fw), mode='bilinear', align_corners=False)
            mask_tensor = mask_tensor.squeeze(0).squeeze(0)
            if not mask_tensor.numel() or (mask_tensor <= 0.5).all():
                continue
            mask_bool = mask_tensor > 0.5
            if not mask_bool.any():
                continue
            mask_overlay[vy1:vy2, vx1:vx2] |= mask_bool
        if mask_overlay.any():
            region = canvas.to(dtype=torch.float32)
            mask_bool = mask_overlay.unsqueeze(0)
            alpha_map = mask_bool.float() * alpha
            blend = region * (1 - alpha_map) + color * alpha_map
            canvas[:] = blend.clamp(0, 255).to(dtype=dtype)
            if self.debug_mode:
                self._draw_cuda_contours(canvas, mask_overlay)

    def _mask_fragment_to_tensor(self, mask_fragment, device):
        if mask_fragment is None:
            return None
        if torch.is_tensor(mask_fragment):
            return mask_fragment.to(device=device, dtype=torch.float32)
        try:
            arr = np.asarray(mask_fragment, dtype=np.float32)
        except Exception:
            return None
        if arr.size == 0:
            return None
        return torch.from_numpy(arr).to(device=device)

    def _draw_cuda_contours(self, canvas, mask_overlay):
        if mask_overlay is None or not mask_overlay.any():
            return
        try:
            mask_cpu = (mask_overlay.to(torch.uint8).cpu().numpy() * 255).astype('uint8')
            contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return
            canvas_cpu = canvas.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            cv2.drawContours(canvas_cpu, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
            canvas.copy_(torch.from_numpy(canvas_cpu).permute(2, 0, 1).to(canvas.device, dtype=canvas.dtype))
        except Exception:
            pass

    def _build_pipeline_report(self, *, last_gpu_full_success=None, last_gpu_full_error=None):
        return {
            'requested_mode': self.pipeline_mode,
            'active_mode': self.active_pipeline_mode,
            'helpers_available': self._cuda_helpers_initialized,
            'last_gpu_full_success': last_gpu_full_success,
            'last_gpu_full_error': last_gpu_full_error,
        }


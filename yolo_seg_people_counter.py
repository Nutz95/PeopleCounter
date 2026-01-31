import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tiling_manager import TilingManager
from engines.yolo.tensorrt_engine import YoloTensorRTEngine
from engines.yolo.openvino_engine import YoloOpenVINOEngine
from engines.yolo.ultralytics_engine import YoloUltralyticsEngine

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
        self.last_device = "Unknown"

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

    def count_people(self, image, tile_size=640, draw_boxes=True, use_tiling=True, draw_tiles=False):
        t0_total = time.time()
        image_rgb = image[..., ::-1].copy()
        h, w = image_rgb.shape[:2]

        # --- Crop / tiles ---
        t_crop_start = time.time()
        tiles = []
        metadata = []  # (x, y, sw, sh, orig_w, orig_h)

        if not use_tiling or (h <= 640 and w <= 640):
            tiles.append(image_rgb)
            metadata.append((0, 0, 1.0, 1.0, w, h))
        else:
            # 1. global anchor
            global_tile = cv2.resize(image_rgb, (640, 640))
            tiles.append(global_tile)
            metadata.append((0, 0, w / 640.0, h / 640.0, w, h))

            # 2. high-res tiles from tiler
            tiles_coords = self.tiler.get_tiles(image_rgb.shape)
            for x1, y1, x2, y2 in tiles_coords:
                tw, th = x2 - x1, y2 - y1
                tiles.append(image_rgb[y1:y2, x1:x2])
                metadata.append((x1, y1, tw / 640.0, th / 640.0, tw, th))

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
        t_draw_start = time.time()
        image_out = image.copy() if draw_boxes else None
        if draw_boxes:
            h_img, w_img = image_out.shape[:2]
            debug_mode = os.environ.get("DEBUG_TILING", "0") == "1" or os.environ.get("EXTREME_DEBUG", "0") == "1" or getattr(self, 'debug', False)

            combined_mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
            debug_rects = []

            for cluster in clusters:
                cvx1, cvy1, cvx2, cvy2 = 999999, 999999, -999999, -999999
                for idx in cluster:
                    bx1, by1, bx2, by2 = all_boxes[idx]
                    cvx1, cvy1 = min(cvx1, bx1), min(cvy1, by1)
                    cvx2, cvy2 = max(cvx2, bx2), max(cvy2, by2)

                cvx1, cvy1 = max(0, int(cvx1)), max(0, int(cvy1))
                cvx2, cvy2 = min(w_img, int(cvx2)), min(h_img, int(cvy2))
                cw, ch = cvx2 - cvx1, cvy2 - cvy1

                if cw > 0 and ch > 0:
                    person_mask = np.zeros((ch, cw), dtype=np.float32)
                    has_any_mask = False
                    for idx in cluster:
                        mask_fragment = all_masks[idx]
                        if mask_fragment is not None and mask_fragment.size > 0:
                            has_any_mask = True
                            vx1, vy1, vx2, vy2 = [int(v) for v in all_visible_boxes[idx]]
                            fx1, fy1 = max(cvx1, vx1), max(cvy1, vy1)
                            fx2, fy2 = min(cvx2, vx2), min(cvy2, vy2)
                            fw, fh = fx2 - fx1, fy2 - fy1
                            if fw > 0 and fh > 0:
                                try:
                                    m_resized = cv2.resize(mask_fragment, (fw, fh), interpolation=cv2.INTER_LINEAR)
                                    px, py = fx1 - cvx1, fy1 - cvy1
                                    target_roi = person_mask[py:py+fh, px:px+fw]
                                    if target_roi.shape == m_resized.shape:
                                        np.maximum(target_roi, m_resized, out=target_roi)
                                except Exception:
                                    pass

                    if has_any_mask:
                        mask_bin = (person_mask > 0.5).astype(np.uint8)
                        # Accumulate masks as a boolean union into the full-frame mask
                        roi_target = combined_mask_full[cvy1:cvy2, cvx1:cvx2]
                        if roi_target.shape == mask_bin.shape:
                            np.maximum(roi_target, mask_bin, out=roi_target)

            # Single blended overlay from the union of all cluster masks (cheaper and avoids additive darkening)
            if combined_mask_full.any():
                ys, xs = np.where(combined_mask_full > 0)
                miny, maxy = max(0, int(ys.min()) - 2), min(h_img, int(ys.max()) + 2)
                minx, maxx = max(0, int(xs.min()) - 2), min(w_img, int(xs.max()) + 2)
                mask_roi = combined_mask_full[miny:maxy, minx:maxx]
                img_roi = image_out[miny:maxy, minx:maxx]
                overlay_roi = img_roi.copy()
                overlay_roi[mask_roi > 0] = (0, 255, 0)
                blended_roi = cv2.addWeighted(overlay_roi, 0.4, img_roi, 0.6, 0)
                # mask_roi is 2D; use 2D boolean indexing to copy RGB pixels
                mask_bool = mask_roi > 0
                img_roi[mask_bool] = blended_roi[mask_bool]
                contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # draw contours in-place on the ROI
                    cv2.drawContours(image_out[miny:maxy, minx:maxx], contours, -1, (0, 100, 0), 2)

            if debug_mode:
                for (tx1, ty1, tx2, ty2) in debug_rects:
                    cv2.rectangle(image_out, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

            if os.environ.get("DEBUG_TILING", "0") == "1":
                for info in metadata:
                    if len(info) >= 6:
                        tx1, ty1, tw, th = info[0], info[1], info[4], info[5]
                        cv2.rectangle(image_out, (int(tx1), int(ty1)), (int(tx1+tw), int(ty1+th)), (0, 255, 255), 1)
                        if mask_fragment is not None and mask_fragment.size > 0:
                            has_any_mask = True
                            vx1, vy1, vx2, vy2 = [int(v) for v in all_visible_boxes[idx]]
                            # Intersection avec le cadre du cluster
                            fx1, fy1 = max(cvx1, vx1), max(cvy1, vy1)
                            fx2, fy2 = min(cvx2, vx2), min(cvy2, vy2)
                            fw, fh = fx2 - fx1, fy2 - fy1
                            if fw > 0 and fh > 0:
                                try:
                                    m_resized = cv2.resize(mask_fragment, (fw, fh), interpolation=cv2.INTER_LINEAR)
                                    px, py = fx1 - cvx1, fy1 - cvy1
                                    target_roi = person_mask[py:py+fh, px:px+fw]
                                    if target_roi.shape == m_resized.shape:
                                        np.maximum(target_roi, m_resized, out=target_roi)
                                except Exception: pass
                    
                    if has_any_mask:
                        # --- OPTIMISATION : BLENDING LOCAL ---
                        # On binarise le masque fusionné du cluster
                        mask_bin = (person_mask > 0.5).astype(np.uint8)
                        # On travaille directement sur la zone de l'image (pas de copie full 4K overlay)
                        roi_img = image_out[cvy1:cvy2, cvx1:cvx2]
                        if roi_img.shape[:2] == mask_bin.shape[:2]:
                            # Créer une version verte de la ROI
                            green_roi = np.zeros_like(roi_img)
                            green_roi[:] = (0, 255, 0)
                            # Alpha blending local (0.4 vert, 0.6 image)
                            blended = cv2.addWeighted(roi_img, 0.6, green_roi, 0.4, 0)
                            # Appliquer uniquement là où le masque est présent
                            image_out[cvy1:cvy2, cvx1:cvx2] = np.where(mask_bin[:,:,None] > 0, blended, roi_img)
                
                best_idx = cluster[0]
                tx1, ty1, tx2, ty2 = all_boxes[best_idx]
                cv2.rectangle(image_out, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (0, 255, 0), 2)

            # Rétablir le Debug Tiling si demandé
            if os.environ.get("DEBUG_TILING", "0") == "1":
                for info in metadata:
                    # metadata tuple order: (x1, y1, scale_x, scale_y, tw, th)
                    if len(info) >= 6:
                        tx1, ty1, tw, th = info[0], info[1], info[4], info[5]
                        cv2.rectangle(image_out, (int(tx1), int(ty1)), (int(tx1+tw), int(ty1+th)), (0, 255, 255), 1)

        t_draw = time.time() - t_draw_start

        # On log le détail si on est en décalage
        total_internal = t_crop + t_inf + t_fuse + t_draw
        print(f"  [YOLO DETAILED] Crop={t_crop*1000:.1f}ms | Infer={t_inf*1000:.1f}ms | Fuse={t_fuse*1000:.1f}ms | Draw={t_draw*1000:.1f}ms | Total={total_internal*1000:.1f}ms")

        return int(total_count), image_out if draw_boxes else int(total_count)


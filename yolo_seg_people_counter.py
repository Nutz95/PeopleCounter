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
            self.engine = YoloTensorRTEngine(model_path, confidence_threshold, device, self.pool)
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
        
        t_crop_start = time.time()
        tiles = []
        metadata = [] # (x, y, sw, sh, orig_w, orig_h)

        if not use_tiling or (h <= 640 and w <= 640):
            tiles.append(image_rgb)
            metadata.append((0, 0, 1.0, 1.0, w, h))
        else:
            # 1. TUILE GLOBALE (L'ancre pour la cohérence)
            global_tile = cv2.resize(image_rgb, (640, 640))
            tiles.append(global_tile)
            metadata.append((0, 0, w/640.0, h/640.0, w, h))

            # 2. TUILES HAUTE RÉSOLUTION
            tiles_coords = self.tiler.get_tiles(image_rgb.shape)
            for x1, y1, x2, y2 in tiles_coords:
                tw, th = x2 - x1, y2 - y1
                tiles.append(image_rgb[y1:y2, x1:x2])
                metadata.append((x1, y1, tw/640.0, th/640.0, tw, th))
        t_crop = time.time() - t_crop_start

        # Inférence via l'engine délégué
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

        # Fusion des détections via global NMS
        t_fuse_start = time.time()
        clusters = self.tiler.fuse_results(all_boxes, all_scores, all_masks, iou_threshold=0.3)
        total_count = len(clusters)
        t_fuse = time.time() - t_fuse_start

        t_draw_start = time.time()
        image_out = image.copy() if draw_boxes else None
        if draw_boxes:
            overlay = image_out.copy()
            h_img, w_img = image_out.shape[:2]

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
                                except Exception: pass
                    
                    if has_any_mask:
                        mask_bin = (person_mask > 0.5).astype(np.uint8)
                        roi_img = overlay[cvy1:cvy2, cvx1:cvx2]
                        if roi_img.shape[:2] == mask_bin.shape[:2]:
                            roi_img[mask_bin > 0] = (0, 255, 0)
                
                best_idx = cluster[0]
                tx1, ty1, tx2, ty2 = all_boxes[best_idx]
                cv2.rectangle(image_out, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (0, 255, 0), 2)
            
            cv2.addWeighted(overlay, 0.4, image_out, 0.6, 0, image_out)
        t_draw = time.time() - t_draw_start

        # On log le détail si on est en décalage
        total_internal = t_crop + t_inf + t_fuse + t_draw
        print(f"  [YOLO DETAILED] Crop={t_crop*1000:.1f}ms | Infer={t_inf*1000:.1f}ms | Fuse={t_fuse*1000:.1f}ms | Draw={t_draw*1000:.1f}ms | Total={total_internal*1000:.1f}ms")

        return int(total_count), image_out if draw_boxes else int(total_count)


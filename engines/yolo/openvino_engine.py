import openvino as ov
import numpy as np
import time
import cv2
import os
from .base import YoloEngine

class YoloOpenVINOEngine(YoloEngine):
    def __init__(self, model_path, confidence_threshold=0.25, device='GPU'):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.person_class_id = 0
        self.last_perf = {'preprocess': 0, 'infer': 0, 'postprocess': 0}
        
        core = ov.Core()
        if os.path.isdir(model_path):
            xml_candidates = [p for p in os.listdir(model_path) if p.lower().endswith('.xml')]
            if xml_candidates: model_path = os.path.join(model_path, xml_candidates[0])
        
        ov_model = core.read_model(model_path)
        self.compiled_model = core.compile_model(ov_model, device)
        self.input_layer = self.compiled_model.inputs[0]
        self.output_layers = self.compiled_model.outputs
        
        try:
            self.device_name = self.compiled_model.get_property("FULL_DEVICE_NAME")
        except: self.device_name = f"OpenVINO ({device})"

    def infer(self, tiles, metadata):
        target_h, target_w = 640, 640
        all_boxes, all_scores, all_masks, all_visible_boxes = [], [], [], []
        self.last_perf = {k: 0 for k in self.last_perf}
        
        batch_size = 4 # Conservative default for OpenVINO tiling
        
        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i+batch_size]
            actual_b = len(chunk)
            
            t0 = time.time()
            # Optimisation majeure : utilisation de cv2.dnn.blobFromImages (C++/Vecteur)
            input_data = cv2.dnn.blobFromImages(chunk, 1.0/255.0, (target_w, target_h), (0, 0, 0), swapRB=False, crop=False)
            self.last_perf['preprocess'] += time.time() - t0

            t1 = time.time()
            results = self.compiled_model(input_data)
            self.last_perf['infer'] += time.time() - t1

            t2 = time.time()
            preds = None
            protos = None
            for out in self.output_layers:
                shape = list(out.get_shape())
                if len(shape) == 3: preds = results[out]
                if len(shape) == 4: protos = results[out]

            if preds is None: continue

            for j in range(actual_b):
                p = preds[j]
                if p.shape[0] < p.shape[1]: p = p.T
                
                num_mask_coeffs = 32
                total_cols = p.shape[1]
                num_classes = total_cols - 4 - num_mask_coeffs if protos is not None else total_cols - 4
                
                scores = p[:, 4:4+num_classes]
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)
                mask_idx = np.where((class_ids == self.person_class_id) & (confidences >= self.confidence_threshold))[0]
                
                if len(mask_idx) == 0: continue
                
                x_offset, y_offset, _, _, ow, oh = metadata[i+j]
                sw, sh = ow / target_w, oh / target_h
                
                if protos is not None:
                    coeffs = p[mask_idx, 4+num_classes : 4+num_classes+num_mask_coeffs]
                    m_protos = protos[j]
                    m_raw = 1 / (1 + np.exp(-(coeffs @ m_protos.reshape(num_mask_coeffs, -1))))
                    m_raw = m_raw.reshape(-1, 160, 160)
                    
                    for k in range(len(mask_idx)):
                        bx, by, bw_t, bh_t = p[mask_idx[k], :4]
                        tx1, ty1 = bx - bw_t/2, by - bh_t/2
                        tx2, ty2 = bx + bw_t/2, by + bh_t/2
                        
                        ix1, iy1 = max(0, tx1), max(0, ty1)
                        ix2, iy2 = min(640, tx2), min(640, ty2)
                        
                        if ix2 <= ix1 or iy2 <= iy1: continue
                        
                        gx1_f, gy1_f = tx1 * sw + x_offset, ty1 * sh + y_offset
                        gx2_f, gy2_f = tx2 * sw + x_offset, ty2 * sh + y_offset
                        gx1_v, gy1_v = ix1 * sw + x_offset, iy1 * sh + y_offset
                        gx2_v, gy2_v = ix2 * sw + x_offset, iy2 * sh + y_offset
                        
                        mx1, my1 = int(ix1 / 4), int(iy1 / 4)
                        mx2, my2 = int(ix2 / 4), int(iy2 / 4)
                        
                        all_boxes.append([gx1_f, gy1_f, gx2_f, gy2_f])
                        all_scores.append(confidences[mask_idx[k]])
                        all_masks.append(m_raw[k, my1:my2, mx1:mx2].copy())
                        all_visible_boxes.append([gx1_v, gy1_v, gx2_v, gy2_v])
                else:
                    for k in range(len(mask_idx)):
                        bx, by, bw_h, bh_h = p[mask_idx[k], :4]
                        gx1 = (bx - bw_h/2) * sw + x_offset
                        gy1 = (by - bh_h/2) * sh + y_offset
                        gx2 = (bx + bw_h/2) * sw + x_offset
                        gy2 = (by + bh_h/2) * sh + y_offset
                        all_boxes.append([gx1, gy1, gx2, gy2])
                        all_scores.append(confidences[mask_idx[k]])
                        all_masks.append(None)
                        all_visible_boxes.append([gx1, gy1, gx2, gy2])
            self.last_perf['postprocess'] += time.time() - t2

        return np.array(all_boxes), np.array(all_scores), all_masks, all_visible_boxes

    def get_perf(self):
        return self.last_perf

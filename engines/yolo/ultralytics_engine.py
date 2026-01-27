from ultralytics import YOLO
import numpy as np
import time
from .base import YoloEngine

class YoloUltralyticsEngine(YoloEngine):
    def __init__(self, model_path, confidence_threshold=0.25, device='cpu'):
        self.model = YOLO(model_path, task='segment')
        self.model.to(device)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0
        self.last_perf = {'infer': 0}

    def infer(self, tiles, metadata):
        t0 = time.time()
        results = self.model(tiles, conf=self.confidence_threshold, verbose=False, task='segment')
        self.last_perf['infer'] = time.time() - t0
        
        all_boxes, all_scores, all_masks, all_visible_boxes = [], [], [], []
        for j, r in enumerate(results):
            x_offset, y_offset, _, _, ow, oh = metadata[j]
            sw, sh = ow / 640.0, oh / 640.0
            if r.boxes:
                m_pers = r.boxes.cls == self.person_class_id
                boxes = r.boxes.xyxy[m_pers].cpu().numpy()
                scores = r.boxes.conf[m_pers].cpu().numpy()
                if r.masks is not None:
                    masks_data = r.masks.data[m_pers].cpu().numpy()
                    for k in range(len(boxes)):
                        tx1, ty1, tx2, ty2 = boxes[k]
                        ix1, iy1, ix2, iy2 = max(0, tx1), max(0, ty1), min(640, tx2), min(640, ty2)
                        
                        if ix2 <= ix1 or iy2 <= iy1: continue
                        
                        mh, mw = masks_data[k].shape[:2]
                        scalew, scaleh = mw / 640.0, mh / 640.0
                        mx1, my1 = int(ix1 * scalew), int(iy1 * scaleh)
                        mx2, my2 = int(ix2 * scalew), int(iy2 * scaleh)
                        m_crop = masks_data[k, my1:my2, mx1:mx2].copy()
                        
                        all_boxes.append([tx1 * sw + x_offset, ty1 * sh + y_offset, tx2 * sw + x_offset, ty2 * sh + y_offset])
                        all_scores.append(scores[k])
                        all_masks.append(m_crop)
                        all_visible_boxes.append([ix1 * sw + x_offset, iy1 * sh + y_offset, ix2 * sw + x_offset, iy2 * sh + y_offset])
                else:
                    for k in range(len(boxes)):
                        gx1 = boxes[k][0] * sw + x_offset
                        gy1 = boxes[k][1] * sh + y_offset
                        gx2 = boxes[k][2] * sw + x_offset
                        gy2 = boxes[k][3] * sh + y_offset
                        all_boxes.append([gx1, gy1, gx2, gy2])
                        all_scores.append(scores[k])
                        all_masks.append(None)
                        all_visible_boxes.append([gx1, gy1, gx2, gy2])
        return np.array(all_boxes), np.array(all_scores), all_masks, all_visible_boxes

    def get_perf(self):
        return self.last_perf

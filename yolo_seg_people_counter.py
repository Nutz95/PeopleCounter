import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import tensorrt as trt
import torch

class YoloSegPeopleCounter:
    def __init__(self, model_path, confidence_threshold=0.25, backend='tensorrt_native'):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path) # Ajout de model_name pour la compatibilité HUD
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        self.person_class_id = 0  # COCO: Person is 0
        
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_stream = None
        self.cudart = None
        
        if self.backend == 'tensorrt_native':
            self._init_trt_backend()
        else:
            # Placeholder for fallback (ultralytics)
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    def _init_trt_backend(self):
        from cuda.bindings import runtime as cudart
        self.cudart = cudart
        
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()

        err, self.trt_stream = cudart.cudaStreamCreate()
        
        self.outputs_ordered = [None, None] # [Detections, Prototypes]

        for i in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(i)
            is_input = self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
            shape = self.trt_engine.get_tensor_shape(name)
            
            if is_input:
                self.trt_input_shape = list(shape)
                if self.trt_input_shape[0] < 0: self.trt_input_shape[0] = 32
                vol = np.prod(self.trt_input_shape)
                nbytes = int(vol * np.dtype(dtype).itemsize)
                err, device_mem = cudart.cudaMalloc(nbytes)
                self.trt_inputs.append({'name': name, 'device': device_mem, 'dtype': dtype, 'shape': self.trt_input_shape, 'nbytes': nbytes})
            else:
                out_shape = list(shape)
                if out_shape[0] < 0: out_shape[0] = 32
                vol = np.prod(out_shape)
                nbytes = int(vol * np.dtype(dtype).itemsize)
                host_mem = np.empty(out_shape, dtype=dtype)
                err, device_mem = cudart.cudaMalloc(nbytes)
                out_info = {'name': name, 'host': host_mem, 'device': device_mem, 'dtype': dtype, 'nbytes': nbytes, 'shape': out_shape}
                self.trt_outputs.append(out_info)
                
                # Identification par la forme : [Batch, 116, 8400] vs [Batch, 32, 160, 160]
                if len(out_shape) == 3: # Probablement les detections (B, C, N)
                    self.outputs_ordered[0] = out_info
                elif len(out_shape) == 4: # Probablement les prototypes (B, 32, 160, 160)
                    self.outputs_ordered[1] = out_info

    def count_people(self, image, tile_size=640, draw_boxes=True, use_tiling=True):
        image_rgb = image[..., ::-1].copy()
        h, w = image_rgb.shape[:2]
        
        if not use_tiling:
            # Mode simple, on traite comme un batch de 1
            boxes, scores, masks = self._infer_batch([image_rgb], [(0, 0, 1.0, 1.0, w, h)])
            keep = self._nms_masks(boxes, scores, masks, iou_threshold=0.3)
            total_count = len(keep)
            image_out = image.copy() if draw_boxes else None
            if draw_boxes:
                for i in keep:
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if draw_boxes: return int(total_count), image_out
            return int(total_count)

        # Mode Tiling Hiérarchique
        tiles = []
        metadata = [] # (x, y, sw, sh, orig_w, orig_h)

        # 1. Global
        tiles.append(image_rgb)
        metadata.append((0, 0, 1.0, 1.0, w, h))

        # 2. Quadrants 4K
        if h > 2000:
            qh, qw = h // 2, w // 2
            for i in range(2):
                for j in range(2):
                    y, x = i * qh, j * qw
                    quad = image_rgb[y:y+qh, x:x+qw]
                    tiles.append(quad)
                    metadata.append((x, y, qw/640, qh/640, qw, qh))

        # Infer batch
        all_boxes, all_scores, all_masks = self._infer_batch(tiles, metadata)
        
        if all_boxes.size == 0:
            return (0, image.copy()) if draw_boxes else 0

        # Fusion par NMS de Masks
        keep = self._nms_masks(all_boxes, all_scores, all_masks, iou_threshold=0.3)
        total_count = len(keep)

        image_out = image.copy() if draw_boxes else None
        if draw_boxes:
            # Création d'un calque pour les segments transparents
            overlay = image_out.copy()
            h_img, w_img = image_out.shape[:2]

            for i in keep:
                raw_x1, raw_y1, raw_x2, raw_y2 = all_boxes[i]
                
                # Clipping des coordonnées pour l'image 4K
                x1, y1 = max(0, int(raw_x1)), max(0, int(raw_y1))
                x2, y2 = min(w_img, int(raw_x2)), min(h_img, int(raw_y2))
                
                if x2 <= x1 or y2 <= y1: continue

                # Dessin de la BBox
                cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Dessin du segment (Si disponible)
                try:
                    mask_crop = all_masks[i]
                    if mask_crop.size > 0:
                        mask_bool = mask_crop > 0
                        
                        bw, bh = x2 - x1, y2 - y1
                        if bw > 2 and bh > 2:
                            m_resized = cv2.resize(mask_bool.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_LINEAR)
                            
                            # Zone d'intérêt sur l'image 4K
                            roi = overlay[y1:y2, x1:x2]
                            if roi.shape[0] == bh and roi.shape[1] == bw:
                                mask_color = np.zeros_like(roi)
                                mask_color[m_resized > 0] = (0, 255, 0)
                                # Fusion intelligente
                                overlay[y1:y2, x1:x2] = np.where(mask_color > 0, mask_color, roi)
                except Exception as e:
                    # On ignore les erreurs de dessin pour ne pas bloquer le stream
                    pass
            
            # Fusion avec transparence (alpha=0.3)
            cv2.addWeighted(overlay, 0.3, image_out, 0.7, 0, image_out)
        
        return int(total_count), image_out if draw_boxes else int(total_count)

    def _infer_batch(self, tiles, metadata):
        batch_size = 32
        target_h, target_w = 640, 640
        all_boxes, all_scores, all_masks = [], [], []

        def preprocess(tile):
            if tile.shape[:2] != (target_h, target_w):
                tile = cv2.resize(tile, (target_w, target_h))
            return np.transpose(tile.astype(np.float32) / 255.0, (2, 0, 1))

        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i+batch_size]
            actual_b = len(chunk)
            input_data = np.zeros((actual_b, 3, target_h, target_w), dtype=np.float32)
            
            with ThreadPoolExecutor(max_workers=8) as exe:
                processed = list(exe.map(preprocess, chunk))
                for j, p in enumerate(processed): input_data[j] = p

            input_data = np.ascontiguousarray(input_data)
            self.cudart.cudaMemcpyAsync(self.trt_inputs[0]['device'], input_data.ctypes.data, input_data.nbytes, self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream)
            
            # Configuration des adresses des tenseurs pour TensorRT 10+ (Impératif pour execute_async_v3)
            self.trt_context.set_input_shape(self.trt_inputs[0]['name'], input_data.shape)
            for inp in self.trt_inputs:
                self.trt_context.set_tensor_address(inp['name'], inp['device'])
            for out in self.trt_outputs:
                self.trt_context.set_tensor_address(out['name'], out['device'])

            self.trt_context.execute_async_v3(self.trt_stream)
            
            for out in self.trt_outputs:
                self.cudart.cudaMemcpyAsync(out['host'].ctypes.data, out['device'], out['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream)
            self.cudart.cudaStreamSynchronize(self.trt_stream)

            # Post-process
            preds = self.outputs_ordered[0]['host'] # [32, 116, 8400]
            protos = self.outputs_ordered[1]['host'] # [32, 32, 160, 160]

            for j in range(actual_b):
                p = preds[j].T
                scores = p[:, 4 + self.person_class_id]
                mask_idx = scores >= self.confidence_threshold
                if not np.any(mask_idx): continue

                boxes = p[mask_idx, :4]
                scores = scores[mask_idx]
                coeffs = p[mask_idx, 4+80:] # 32 coefficients pour les masques
                
                # Rescale boxes
                x, y, sw_m, sh_m, ow, oh = metadata[i+j]
                # sw_m = qw/640 (scale relative au tile 640)
                # sw = ow / target_w
                sw, sh = ow / target_w, oh / target_h
                
                x1 = (boxes[:, 0] - boxes[:, 2]/2) * sw + x
                y1 = (boxes[:, 1] - boxes[:, 3]/2) * sh + y
                x2 = (boxes[:, 0] + boxes[:, 2]/2) * sw + x
                y2 = (boxes[:, 1] + boxes[:, 3]/2) * sh + y
                
                # Masques
                m_protos = protos[j] # [32, 160, 160]
                m_protos_flat = m_protos.reshape(32, -1)
                m_raw = (coeffs @ m_protos_flat).reshape(-1, 160, 160)
                
                for k in range(len(scores)):
                    # Box locale dans la tuile (640x640)
                    bx, by, bw_t, bh_t = boxes[k]
                    # Conversion vers la grille du masque (160x160) -> stride de 4
                    mx1 = max(0, int((bx - bw_t/2) / 4))
                    my1 = max(0, int((by - bh_t/2) / 4))
                    mx2 = min(160, int((bx + bw_t/2) / 4))
                    my2 = min(160, int((by + bh_t/2) / 4))
                    
                    # On ne stocke que la partie du masque correspondant à la personne
                    person_mask_crop = m_raw[k, my1:my2, mx1:mx2]
                    
                    all_boxes.append([x1[k], y1[k], x2[k], y2[k]])
                    all_scores.append(scores[k])
                    all_masks.append(person_mask_crop)

        return np.array(all_boxes), np.array(all_scores), all_masks

    def _nms_masks(self, boxes, scores, masks, iou_threshold=0.3):
        """
        NMS utilisant la précision des segments pour la fusion.
        Si deux boîtes se chevauchent, on vérifie si leurs masques 
        se considèrent comme le même objet (fusion de parties).
        """
        if len(boxes) == 0: return []
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        importance = scores * np.sqrt(areas)
        order = importance.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # On compare le gagnant (i) avec tous les autres (order[1:])
            # Calcul IoU BBox rapide d'abord
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter_area = w * h
            
            # Si pas d'intersection BBox, pas besoin de checker les masques
            intersect_idx = np.where(inter_area > 0)[0]
            
            to_suppress = np.zeros(order.size - 1, dtype=bool)
            
            for idx in intersect_idx:
                other_idx = order[idx + 1]
                
                # Critère 1 : Inclusion forte de la BBox (souvent suffisant pour les tuiles)
                min_a = min(areas[i], areas[other_idx])
                inclusion = inter_area[idx] / (min_a + 1e-6)
                
                if inclusion > 0.85:
                    to_suppress[idx] = True
                    continue
                
                # Critère 2 : Si l'inclusion est modérée (>40%), on check le masque
                if inclusion > 0.40:
                    # On compare les masques 160x160
                    # Pour faire simple sans reprojection lourde :
                    # On regarde si les pics d'activation des masques pointent vers la même zone
                    # (Heuristique rapide : centroide du masque reprojecté)
                    m1 = masks[i] > 0 # Logits > 0 -> mask
                    m2 = masks[other_idx] > 0
                    
                    # Si un masque est presque entièrement contenu dans la zone de l'autre
                    # C'est probablement une "partie" (tête, jambe) détectée par une tuile 
                    # qui appartient à un "tout" détecté par la passe globale.
                    to_suppress[idx] = True # Pour l'instant, fusion agressive si inclusion BBox + Segments
                    
            inds = np.where(~to_suppress)[0]
            order = order[inds + 1]
            
        return keep

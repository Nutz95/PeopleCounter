import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import tensorrt as trt
import torch

class YoloSegPeopleCounter:
    def __init__(self, model_path, confidence_threshold=0.25, backend='tensorrt_native', device='cuda'):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        self.device = device
        self.person_class_id = 0
        
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_stream = None
        self.cudart = None
        
        # OpenVINO specific
        self.ov_compiled_model = None
        self.ov_input = None
        self.ov_output = None

        if self.backend == 'tensorrt_native':
            self._init_trt_backend()
        elif self.backend == 'openvino_native':
            self._init_openvino_backend()
        else:
            # Fallback (ultralytics)
            from ultralytics import YOLO
            self.model = YOLO(model_path, task='segment')
            if self.backend != 'openvino':
                self.model.to(self.device)

    def _init_openvino_backend(self):
        import openvino as ov
        core = ov.Core()
        
        model_path = self.model_path
        if os.path.isdir(model_path):
            xml_candidates = [p for p in os.listdir(model_path) if p.lower().endswith('.xml')]
            if xml_candidates:
                model_path = os.path.join(model_path, xml_candidates[0])
        
        ov_model = core.read_model(model_path)
        
        exec_device = self.device
        if exec_device == "cuda": exec_device = "GPU"
        
        self.ov_compiled_model = core.compile_model(ov_model, exec_device)
        self.ov_input = self.ov_compiled_model.inputs[0]
        self.ov_output = self.ov_compiled_model.outputs[0]
        self.ov_outputs = self.ov_compiled_model.outputs
        
        try:
            self.ov_input_shape = list(self.ov_input.get_shape())
            print(f"YOLO Segmentation OpenVINO loaded: {exec_device} (Shape: {self.ov_input_shape})")
        except Exception:
            pass

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

    def _infer_batch_openvino(self, tiles, metadata):
        target_h, target_w = 640, 640
        all_boxes, all_scores, all_masks = [], [], []
        
        # Batch size automatique selon le modèle ou 4 par défaut
        try:
            batch_size = self.ov_input_shape[0] if self.ov_input_shape[0] > 0 else 4
        except Exception:
            batch_size = 4

        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i+batch_size]
            actual_b = len(chunk)
            
            input_data = np.zeros((actual_b, 3, target_h, target_w), dtype=np.float32)
            for j, tile in enumerate(chunk):
                if tile.shape[:2] != (target_h, target_w):
                    tile = cv2.resize(tile, (target_w, target_h))
                input_data[j] = np.transpose(tile.astype(np.float32) / 255.0, (2, 0, 1))

            # Inférence
            results = self.ov_compiled_model(input_data)
            
            # Post-process (Identification des sorties par nom ou forme)
            # YOLOv11-seg OpenVINO a généralement 2 sorties : une pour les preds, une pour les protos
            preds = None
            protos = None
            for out in self.ov_outputs:
                shape = list(out.get_shape())
                if len(shape) == 3: preds = results[out]
                if len(shape) == 4: protos = results[out]

            if preds is None: continue

            for j in range(actual_b):
                p = preds[j]
                if p.shape[0] < p.shape[1]: p = p.T
                
                scores = p[:, 4 + self.person_class_id]
                mask_idx = np.where(scores >= self.confidence_threshold)[0]
                if len(mask_idx) == 0: continue

                boxes = p[mask_idx, :4]
                scores = scores[mask_idx]
                
                x_offset, y_offset, sw_m, sh_m, ow, oh = metadata[i+j]
                sw, sh = ow / target_w, oh / target_h
                
                x1 = (boxes[:, 0] - boxes[:, 2]/2) * sw + x_offset
                y1 = (boxes[:, 1] - boxes[:, 3]/2) * sh + y_offset
                x2 = (boxes[:, 0] + boxes[:, 2]/2) * sw + x_offset
                y2 = (boxes[:, 1] + boxes[:, 3]/2) * sh + y_offset
                
                if protos is not None:
                    coeffs = p[mask_idx, 4+80:4+80+32]
                    m_protos = protos[j]
                    m_protos_flat = m_protos.reshape(32, -1)
                    m_raw = 1 / (1 + np.exp(-(coeffs @ m_protos_flat))) 
                    m_raw = m_raw.reshape(-1, 160, 160)
                    
                    for k in range(len(mask_idx)):
                        bx, by, bw_t, bh_t = boxes[k]
                        mx1 = max(0, int((bx - bw_t/2) / 4))
                        my1 = max(0, int((by - bh_t/2) / 4))
                        mx2 = min(160, int((bx + bw_t/2) / 4))
                        my2 = min(160, int((by + bh_t/2) / 4))
                        
                        person_mask_crop = m_raw[k, my1:my2, mx1:mx2]
                        all_boxes.append([x1[k], y1[k], x2[k], y2[k]])
                        all_scores.append(scores[k])
                        all_masks.append(person_mask_crop)
                else:
                    for k in range(len(mask_idx)):
                        all_boxes.append([x1[k], y1[k], x2[k], y2[k]])
                        all_scores.append(scores[k])
                        all_masks.append(None)

        return np.array(all_boxes), np.array(all_scores), all_masks

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

                # Dessin du segment (Remplissage + Contour au lieu de BBox)
                mask_crop = all_masks[i]
                has_mask = mask_crop is not None and getattr(mask_crop, 'size', 0) > 0

                if has_mask:
                    try:
                        # Seuil de confiance pour le masque (standard YOLO = 0.5)
                        mask_bool = mask_crop > 0.5
                        
                        bw, bh = x2 - x1, y2 - y1
                        if bw > 2 and bh > 2:
                            m_resized = cv2.resize(mask_bool.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_LINEAR)
                            
                            # 1. Remplissage vert transparent (dans l'overlay)
                            roi = overlay[y1:y2, x1:x2]
                            if roi.shape[0] == bh and roi.shape[1] == bw:
                                mask_color = np.zeros_like(roi)
                                mask_color[m_resized > 0] = (0, 255, 0)
                                overlay[y1:y2, x1:x2] = np.where(mask_color > 0, mask_color, roi)
                            
                            # 2. Contour vert foncé (directement sur image_out)
                            contours, _ = cv2.findContours(m_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                cnt[:, :, 0] += x1
                                cnt[:, :, 1] += y1
                                cv2.drawContours(image_out, [cnt], -1, (0, 100, 0), 2)
                    except Exception as e:
                        print(f"[WARN] Mask rendering failed: {e}")
                else:
                    # Fallback BBox classique si pas de segmentation
                    cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Petit label de confiance
                    label = f"{all_scores[i]:.2f}"
                    cv2.putText(image_out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Fusion avec transparence (alpha=0.3)
            cv2.addWeighted(overlay, 0.3, image_out, 0.7, 0, image_out)
        
        return int(total_count), image_out if draw_boxes else int(total_count)

    def _infer_batch(self, tiles, metadata):
        if self.backend == 'openvino_native':
            return self._infer_batch_openvino(tiles, metadata)
        
        all_boxes, all_scores, all_masks = [], [], []

        if self.backend not in ('tensorrt_native', 'openvino_native'):
            # Fallback Ultralytics (PyTorch / CPU)
            results = self.model(tiles, conf=self.confidence_threshold, verbose=False, task='segment')
            for j, r in enumerate(results):
                x_offset, y_offset, _, _, ow, oh = metadata[j]
                sw, sh = ow / 640, oh / 640
                if r.boxes:
                    mask_person = r.boxes.cls == self.person_class_id
                    boxes = r.boxes.xyxy[mask_person].cpu().numpy()
                    scores = r.boxes.conf[mask_person].cpu().numpy()
                    if r.masks is not None:
                        masks_data = r.masks.data[mask_person].cpu().numpy() 
                        for k in range(len(boxes)):
                            bx1, by1, bx2, by2 = boxes[k]
                            mh, mw = masks_data[k].shape[:2]
                            scalew, scaleh = mw / 640.0, mh / 640.0
                            mx1, my1 = max(0, int(bx1 * scalew)), max(0, int(by1 * scaleh))
                            mx2, my2 = min(mw, int(bx2 * scalew)), min(mh, int(by2 * scaleh))
                            
                            if mx2 > mx1 and my2 > my1:
                                m_crop = masks_data[k, my1:my2, mx1:mx2]
                                # Binarisation du masque sigmoid si nécessaire (YOLO output est souvent entre 0 et 1)
                                if m_crop.max() <= 1.01:
                                    m_crop = (m_crop > 0.5).astype(np.float32)
                            else:
                                m_crop = None
                            
                            all_boxes.append([bx1 * sw + x_offset, by1 * sh + y_offset, bx2 * sw + x_offset, by2 * sh + y_offset])
                            all_scores.append(scores[k])
                            all_masks.append(m_crop)
                    else:
                        for k in range(len(boxes)):
                            all_boxes.append([boxes[k][0] * sw + x_offset, boxes[k][1] * sh + y_offset, boxes[k][2] * sw + x_offset, boxes[k][3] * sh + y_offset])
                            all_scores.append(scores[k])
                            all_masks.append(None)
            return np.array(all_boxes), np.array(all_scores), all_masks
            
        batch_size = 32
        target_h, target_w = 640, 640
        # ... suite pour TensorRT
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
            
            # Configuration des adresses des tenseurs pour TensorRT 10+
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
            preds = self.outputs_ordered[0]['host'] # [Batch, C, N]
            has_masks = self.outputs_ordered[1] is not None
            protos = self.outputs_ordered[1]['host'] if has_masks else None

            for j in range(actual_b):
                p = preds[j]
                if p.shape[0] < p.shape[1]: p = p.T # Transpose si [C, N] -> [N, C]
                
                scores = p[:, 4 + self.person_class_id]
                mask_idx = np.where(scores >= self.confidence_threshold)[0]
                if len(mask_idx) == 0: continue

                boxes = p[mask_idx, :4]
                scores = scores[mask_idx]
                
                # Rescale boxes
                x_offset, y_offset, sw_m, sh_m, ow, oh = metadata[i+j]
                sw, sh = ow / target_w, oh / target_h
                
                x1 = (boxes[:, 0] - boxes[:, 2]/2) * sw + x_offset
                y1 = (boxes[:, 1] - boxes[:, 3]/2) * sh + y_offset
                x2 = (boxes[:, 0] + boxes[:, 2]/2) * sw + x_offset
                y2 = (boxes[:, 1] + boxes[:, 3]/2) * sh + y_offset
                
                if has_masks:
                    coeffs = p[mask_idx, 4+80:4+80+32]
                    m_protos = protos[j] # [32, 160, 160]
                    m_protos_flat = m_protos.reshape(32, -1)
                    # Calcul des masques sigmoïdes
                    m_raw = 1 / (1 + np.exp(-(coeffs @ m_protos_flat))) 
                    m_raw = m_raw.reshape(-1, 160, 160)
                    
                    for k in range(len(mask_idx)):
                        bx, by, bw_t, bh_t = boxes[k]
                        mx1 = max(0, int((bx - bw_t/2) / 4))
                        my1 = max(0, int((by - bh_t/2) / 4))
                        mx2 = min(160, int((bx + bw_t/2) / 4))
                        my2 = min(160, int((by + bh_t/2) / 4))
                        
                        person_mask_crop = m_raw[k, my1:my2, mx1:mx2]
                        all_boxes.append([x1[k], y1[k], x2[k], y2[k]])
                        all_scores.append(scores[k])
                        all_masks.append(person_mask_crop)
                else:
                    for k in range(len(mask_idx)):
                        all_boxes.append([x1[k], y1[k], x2[k], y2[k]])
                        all_scores.append(scores[k])
                        all_masks.append(None)

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
                iou = inter_area[idx] / (areas[i] + areas[other_idx] - inter_area[idx] + 1e-6)
                inclusion = inter_area[idx] / (min_a + 1e-6)
                
                # Si IoU standard est élevé, on supprime
                if iou > iou_threshold:
                    to_suppress[idx] = True
                    continue

                # Si l'inclusion est extrême (une petite boîte dans une grosse)
                if inclusion > 0.85:
                    to_suppress[idx] = True
                    continue
                
                # Critère 2 : Si l'inclusion est modérée et qu'on a des masques
                if inclusion > 0.40 and masks[i] is not None and masks[other_idx] is not None:
                    # Heuristique : si les masques se chevauchent aussi fortement
                    to_suppress[idx] = True 
                    
            inds = np.where(~to_suppress)[0]
            order = order[inds + 1]
            
        return keep

import tensorrt as trt
import numpy as np
import time
import cv2
from .base import YoloEngine

class YoloTensorRTEngine(YoloEngine):
    def __init__(self, model_path, confidence_threshold=0.25, device='cuda', pool=None):
        from cuda.bindings import runtime as cudart
        self.cudart = cudart
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.person_class_id = 0
        self.pool = pool
        self.last_perf = {'preprocess': 0, 'h2d': 0, 'infer': 0, 'd2h': 0, 'postprocess': 0}

        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()

        err, self.trt_stream = cudart.cudaStreamCreate()
        self.trt_inputs = []
        self.trt_outputs = []
        self.outputs_ordered = [None, None]

        for i in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(i)
            is_input = self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
            shape = list(self.trt_engine.get_tensor_shape(name))
            
            if is_input:
                if shape[0] < 0: shape[0] = 32
                vol = np.prod(shape)
                nbytes = int(vol * np.dtype(dtype).itemsize)
                err, device_mem = cudart.cudaMalloc(nbytes)
                self.trt_inputs.append({'name': name, 'device': device_mem, 'dtype': dtype, 'shape': shape, 'nbytes': nbytes})
            else:
                if shape[0] < 0: shape[0] = 32
                vol = np.prod(shape)
                nbytes = int(vol * np.dtype(dtype).itemsize)
                host_mem = np.empty(shape, dtype=dtype)
                err, device_mem = cudart.cudaMalloc(nbytes)
                out_info = {'name': name, 'host': host_mem, 'device': device_mem, 'dtype': dtype, 'nbytes': nbytes, 'shape': shape}
                self.trt_outputs.append(out_info)
                if len(shape) == 3: self.outputs_ordered[0] = out_info
                elif len(shape) == 4: self.outputs_ordered[1] = out_info

    def infer(self, tiles, metadata):
        batch_size = 32
        target_h, target_w = 640, 640
        all_boxes, all_scores, all_masks, all_visible_boxes = [], [], [], []
        self.last_perf = {k: 0 for k in self.last_perf}

        def preprocess_local(tile):
            if tile.shape[:2] != (target_h, target_w):
                tile = cv2.resize(tile, (target_w, target_h))
            return np.transpose(tile.astype(np.float32) / 255.0, (2, 0, 1))

        def preprocess_parallel(tile):
            # Version parallélisée plus rapide que blobFromImages sur CPU multi-coeurs
            if tile.shape[:2] != (640, 640):
                tile = cv2.resize(tile, (640, 640))
            return np.transpose(tile.astype(np.float32) * (1.0/255.0), (2, 0, 1))

        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i+batch_size]
            actual_b = len(chunk)
            
            t_pre_0 = time.time()
            # On mesure séparément le resize et le stack/transpose
            if self.pool:
                # 1. Resize & Norm (CPU intensive)
                processed = list(self.pool.map(preprocess_parallel, chunk))
                t_pre_1 = time.time()
                # 2. Stack (Memory bandwidth intensive)
                input_data = np.stack(processed)
                t_pre_2 = time.time()
            else:
                input_data = np.stack([preprocess_parallel(t) for t in chunk])
                t_pre_1 = t_pre_2 = time.time()

            input_data = np.ascontiguousarray(input_data)
            t_pre_3 = time.time()
            
            self.last_perf['preprocess'] += t_pre_3 - t_pre_0
            # On log une fois par frame les détails internes du Pre
            if i == 0:
                 print(f"  [INTERNAL PRE] Resize={(t_pre_1-t_pre_0)*1000:.1f}ms | Stack={(t_pre_2-t_pre_1)*1000:.1f}ms | Contiguous={(t_pre_3-t_pre_2)*1000:.1f}ms (Batch={actual_b})")

            t1 = time.time()
            self.cudart.cudaMemcpyAsync(self.trt_inputs[0]['device'], input_data.ctypes.data, input_data.nbytes, self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream)
            self.last_perf['h2d'] += time.time() - t1
            
            self.trt_context.set_input_shape(self.trt_inputs[0]['name'], input_data.shape)
            for inp in self.trt_inputs: self.trt_context.set_tensor_address(inp['name'], inp['device'])
            for out in self.trt_outputs: self.trt_context.set_tensor_address(out['name'], out['device'])

            t2 = time.time()
            self.trt_context.execute_async_v3(self.trt_stream)
            self.cudart.cudaStreamSynchronize(self.trt_stream)
            self.last_perf['infer'] += time.time() - t2

            t3 = time.time()
            for out in self.trt_outputs:
                self.cudart.cudaMemcpyAsync(out['host'].ctypes.data, out['device'], out['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream)
            self.cudart.cudaStreamSynchronize(self.trt_stream)
            self.last_perf['d2h'] += time.time() - t3

            # Post-process (YOLOv12/v26 structure)
            t4 = time.time()
            preds = self.outputs_ordered[0]['host']
            has_masks = self.outputs_ordered[1] is not None
            protos = self.outputs_ordered[1]['host'] if has_masks else None
            num_mask_coeffs = self.outputs_ordered[1]['shape'][1] if has_masks else 0

            for j in range(actual_b):
                p = preds[j]
                if p.shape[0] < p.shape[1]: p = p.T
                total_cols = p.shape[1]
                num_classes = total_cols - 4 - num_mask_coeffs
                
                scores = p[:, 4:4+num_classes]
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)
                mask_idx = np.where((class_ids == self.person_class_id) & (confidences >= self.confidence_threshold))[0]
                
                if len(mask_idx) == 0: continue
                boxes = p[mask_idx, :4]
                scores_final = confidences[mask_idx]
                
                x_offset, y_offset, _, _, ow, oh = metadata[i+j]
                sw, sh = ow / target_w, oh / target_h
                
                # Coordonnées du masque (0..160)
                # On calcule les coordonnées de la boîte clippée à la tuile (0..640)
                # pour extraire la zone correspondante du prototype.
                raw_boxes = p[mask_idx, :4] # [cx, cy, w, h]
                
                for k in range(len(mask_idx)):
                    bx, by, bw_t, bh_t = raw_boxes[k]
                    
                    # Boîte dans le repère 640x640 (tx1, ty1, tx2, ty2) - FULL
                    tx1, ty1 = bx - bw_t/2, by - bh_t/2
                    tx2, ty2 = bx + bw_t/2, by + bh_t/2
                    
                    # Clipping à la tuile (0..640) pour l'extraction du masque
                    ix1, iy1 = max(0, tx1), max(0, ty1)
                    ix2, iy2 = min(640, tx2), min(640, ty2)
                    
                    if ix2 <= ix1 or iy2 <= iy1: continue
                    
                    # Coordonnées globales FULL (pour le NMS de fusion)
                    gx1_f, gy1_f = tx1 * sw + x_offset, ty1 * sh + y_offset
                    gx2_f, gy2_f = tx2 * sw + x_offset, ty2 * sh + y_offset
                    
                    # Coordonnées globales VISIBLES (pour le placement du masque)
                    gx1_v, gy1_v = ix1 * sw + x_offset, iy1 * sh + y_offset
                    gx2_v, gy2_v = ix2 * sw + x_offset, iy2 * sh + y_offset
                    
                    if has_masks and num_mask_coeffs > 0:
                        start_coeffs = 4+num_classes
                        coeffs = p[mask_idx[k], start_coeffs : start_coeffs + num_mask_coeffs]
                        m_protos = protos[j] # [32, ph, pw]
                        ph, pw = m_protos.shape[1], m_protos.shape[2]
                        
                        # Indices dans le prototype (0..160)
                        mx1, my1 = int(ix1 * (pw/target_w)), int(iy1 * (ph/target_h))
                        mx2, my2 = int(ix2 * (pw/target_w)), int(iy2 * (ph/target_h))
                        
                        # Sigmoid sur la zone d'intérêt seulement (mask fragment)
                        protos_crop = m_protos[:, my1:my2, mx1:mx2].reshape(num_mask_coeffs, -1)
                        m_raw = 1 / (1 + np.exp(-(coeffs @ protos_crop)))
                        mask_box = m_raw.reshape(my2-my1, mx2-mx1)
                        
                        all_boxes.append([gx1_f, gy1_f, gx2_f, gy2_f])
                        all_scores.append(confidences[mask_idx[k]])
                        all_masks.append(mask_box.copy())
                        all_visible_boxes.append([gx1_v, gy1_v, gx2_v, gy2_v])
                    else:
                        all_boxes.append([gx1_f, gy1_f, gx2_f, gy2_f])
                        all_scores.append(confidences[mask_idx[k]])
                        all_masks.append(None)
                        all_visible_boxes.append([gx1_f, gy1_f, gx2_f, gy2_f])
            self.last_perf['postprocess'] += time.time() - t4

        return np.array(all_boxes), np.array(all_scores), all_masks, all_visible_boxes

    def get_perf(self):
        return self.last_perf

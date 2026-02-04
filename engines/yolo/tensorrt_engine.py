import tensorrt as trt
import numpy as np
import time
from .base import YoloEngine
from .preprocessors import CpuPreprocessor, GpuPreprocessor

# Optional GPU helpers
try:
    import cupy as cp
except Exception:
    cp = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

class YoloTensorRTEngine(YoloEngine):
    def __init__(self, model_path, confidence_threshold=0.25, device='cuda', pool=None, use_gpu_preproc=False, use_gpu_post=False):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.person_class_id = 0
        self.pool = pool
        self.last_perf = {'preprocess': 0, 'h2d': 0, 'infer': 0, 'd2h': 0, 'postprocess': 0}
        self.use_gpu_preproc = bool(use_gpu_preproc and (torch is not None) and torch.cuda.is_available())
        self.use_gpu_post = use_gpu_post and (torch is not None)
        self.target_size = (640, 640)
        preprocessor_cls = GpuPreprocessor if self.use_gpu_preproc else CpuPreprocessor
        self.preprocessor = preprocessor_cls(self.target_size)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()

        # Stream and memory management via Torch (highly stable)
        if torch is not None and torch.cuda.is_available():
            self.trt_stream = torch.cuda.current_stream().cuda_stream
        else:
            self.trt_stream = 0

        self.trt_inputs = []
        self.trt_outputs = []
        self._torch_tensors = {} # Keep references to prevent GC
        self.outputs_ordered = [None, None]

        for i in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(i)
            is_input = self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
            shape = list(self.trt_engine.get_tensor_shape(name))
            
            # Fix dynamic batch
            if shape[0] < 0: shape[0] = 32 # Default max batch
            
            # Map numpy dtype to torch dtype
            t_dtype = torch.from_numpy(np.zeros(1, dtype=dtype)).dtype
            t_tensor = torch.empty(tuple(shape), dtype=t_dtype, device='cuda')
            self._torch_tensors[name] = t_tensor

            if is_input:
                self.trt_inputs.append({'name': name, 'device': t_tensor.data_ptr(), 'dtype': dtype, 'shape': shape})
            else:
                host_mem = np.empty(shape, dtype=dtype)
                out_info = {'name': name, 'host': host_mem, 'device': t_tensor.data_ptr(), 'dtype': dtype, 'shape': shape}
                self.trt_outputs.append(out_info)
                
                # Robustly identify Box vs Proto outputs
                # Box output is usually 3D (B, 4+C+K, 8400) or similar
                # Proto output for Segmentation is 4D (B, 32, 160, 160)
                if len(shape) == 3:
                     self.outputs_ordered[0] = out_info
                elif len(shape) == 4:
                     self.outputs_ordered[1] = out_info
                else:
                     # Default fallback if only one output and it's not 3D/4D (rare)
                     if self.outputs_ordered[0] is None:
                         self.outputs_ordered[0] = out_info

        # Safety: if only one output, ensure it's at index 0
        if self.outputs_ordered[0] is None and len(self.trt_outputs) > 0:
            self.outputs_ordered[0] = self.trt_outputs[0]


    def infer(self, tiles, metadata):
        batch_size = 32
        target_h, target_w = self.target_size
        all_boxes, all_scores, all_masks, all_visible_boxes = [], [], [], []
        self.last_perf = {k: 0 for k in self.last_perf}

        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i+batch_size]
            actual_b = len(chunk)

            t_pre_0 = time.time()
            input_tensor = self.preprocessor.preprocess(chunk)
            self.last_perf['preprocess'] += time.time() - t_pre_0

            if actual_b == 0:
                continue

            t1 = time.time()
            input_name = self.trt_inputs[0]['name']
            input_chunk = input_tensor[:actual_b]
            if input_chunk.device.type != 'cuda':
                input_chunk = input_chunk.to('cuda', non_blocking=True)
            self._torch_tensors[input_name][:actual_b].copy_(input_chunk)
            self.last_perf['h2d'] += time.time() - t1
            
            # Execution
            self.trt_context.set_input_shape(input_name, (actual_b, 3, target_h, target_w))
            for name, tensor in self._torch_tensors.items():
                self.trt_context.set_tensor_address(name, tensor.data_ptr())

            t2 = time.time()
            self.trt_context.execute_async_v3(self.trt_stream)
            torch.cuda.synchronize()
            self.last_perf['infer'] += time.time() - t2

            t3 = time.time()
            # D2H via Torch
            for out in self.trt_outputs:
                out['host'][:actual_b] = self._torch_tensors[out['name']][:actual_b].cpu().numpy()
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
                        mx1 = max(0, int(ix1 * (pw/target_w)))
                        my1 = max(0, int(iy1 * (ph/target_h)))
                        mx2 = min(pw, int(ix2 * (pw/target_w)))
                        my2 = min(ph, int(iy2 * (ph/target_h)))
                        
                        if mx2 > mx1 and my2 > my1:
                            # Sigmoid sur la zone d'intérêt seulement (mask fragment)
                            protos_crop = m_protos[:, my1:my2, mx1:mx2].reshape(num_mask_coeffs, -1)
                            # GPU post-processing path
                            if self.use_gpu_post and cp is not None:
                                try:
                                    coeffs_gpu = cp.asarray(coeffs)
                                    protos_gpu = cp.asarray(protos_crop)
                                    m_raw_gpu = 1.0 / (1.0 + cp.exp(- (coeffs_gpu @ protos_gpu)))
                                    mask_box = cp.asnumpy(m_raw_gpu).reshape(my2-my1, mx2-mx1)
                                except Exception:
                                    # fallback to numpy
                                    m_raw = 1 / (1 + np.exp(-(coeffs @ protos_crop)))
                                    mask_box = m_raw.reshape(my2-my1, mx2-mx1)
                            elif self.use_gpu_post and torch is not None and torch.cuda.is_available():
                                try:
                                    t_coeff = torch.from_numpy(coeffs).cuda()
                                    t_protos = torch.from_numpy(protos_crop).cuda()
                                    t_res = torch.matmul(t_coeff, t_protos)
                                    t_sig = torch.sigmoid(t_res)
                                    mask_box = t_sig.cpu().numpy().reshape(my2-my1, mx2-mx1)
                                except Exception:
                                    m_raw = 1 / (1 + np.exp(-(coeffs @ protos_crop)))
                                    mask_box = m_raw.reshape(my2-my1, mx2-mx1)
                            else:
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

    def set_preprocessor_mode(self, mode):
        target = (mode or 'auto').lower()
        if target not in ('auto', 'cpu', 'gpu'):
            target = 'auto'
        if target == 'gpu' and torch is not None and torch.cuda.is_available():
            self.use_gpu_preproc = True
            self.preprocessor = GpuPreprocessor(self.target_size)
        elif target == 'auto':
            use_gpu = torch is not None and torch.cuda.is_available()
            self.use_gpu_preproc = use_gpu
            preprocessor_cls = GpuPreprocessor if use_gpu else CpuPreprocessor
            self.preprocessor = preprocessor_cls(self.target_size)
        else:
            self.use_gpu_preproc = False
            self.preprocessor = CpuPreprocessor(self.target_size)

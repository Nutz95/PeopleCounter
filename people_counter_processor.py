import sys
import os
# Use local vendors folder for lwcc if present; do not hardcode Windows paths
from pathlib import Path
# Remove hardcoded Windows path and import lwcc robustly
try:
    # Normal import (preferred)
    from lwcc import get_count_from_frame
    LWCC_AVAILABLE = True
except Exception:
    # Fallbacks: try to import module and locate function in submodules
    try:
        import lwcc as _lwcc_mod
        if hasattr(_lwcc_mod, 'get_count_from_frame'):
            get_count_from_frame = _lwcc_mod.get_count_from_frame
            LWCC_AVAILABLE = True
        elif hasattr(_lwcc_mod, 'api') and hasattr(_lwcc_mod.api, 'get_count_from_frame'):
            get_count_from_frame = _lwcc_mod.api.get_count_from_frame
            LWCC_AVAILABLE = True
        else:
            LWCC_AVAILABLE = False
            # last-resort stub that raises a clear error at call-time
            def get_count_from_frame(*args, **kwargs):
                raise ImportError('lwcc.get_count_from_frame not found in installed lwcc package')
    except Exception:
        LWCC_AVAILABLE = False
        def get_count_from_frame(*args, **kwargs):
            raise ImportError('lwcc package not available')

try:
    # preprocess_frame may live in different submodules
    from lwcc.util.functions import preprocess_frame
except Exception:
    try:
        if 'lwcc' in globals():
            if hasattr(_lwcc_mod, 'util') and hasattr(_lwcc_mod.util, 'functions') and hasattr(_lwcc_mod.util.functions, 'preprocess_frame'):
                preprocess_frame = _lwcc_mod.util.functions.preprocess_frame
            else:
                def preprocess_frame(x, *a, **k):
                    return x
        else:
            def preprocess_frame(x, *a, **k):
                return x
    except Exception:
        def preprocess_frame(x, *a, **k):
            return x
import cv2
import numpy as np
from PIL import Image
import torch
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class PeopleCounterProcessor:
    def __init__(self, model_name="CSRNet", model_weights="SHA", backend="torch", openvino_device="GPU", density_threshold=15):
        """_summary_

        Args:
            model_name (str, optional): _description_. Defaults to "CSRNet".
            model_weights (str, optional): _description_. Defaults to "SHA".
            possible models: 'CSRNet' (SHA, SHB), 'SFANet' (SHA, SHB),'Bay' (QNRF, SHA, SHB), 'DM-Count' (QNRF, SHA, SHB)
        """
        self.model_name = model_name
        self.model_weights = model_weights
        self.model = None
        self.backend = (backend or "torch").lower()
        self.openvino_device = openvino_device
        self.ov_compiled_model = None
        self.ov_input = None
        self.ov_output = None
        self.ov_input_shape = None
        self.density_threshold = density_threshold
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_bindings = []
        self.cudart = None
        self.lock = threading.Lock() # Lock pour éviter les accès concurrents au contexte TRT/OpenVINO
        self.last_device = "Unknown"
        self.last_perf = {'preprocess': 0, 'h2d': 0, 'infer': 0, 'd2h': 0, 'postprocess': 0}
        max_workers = min(32, (os.cpu_count() or 4) * 2)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

        if self.backend == "tensorrt":
            engine_path = os.environ.get("LWCC_TRT_ENGINE", "dm_count.engine")
            if os.path.isfile(engine_path):
                try:
                    import tensorrt as trt
                    from cuda.bindings import runtime as cudart
                    self.cudart = cudart
                    self.last_device = "GPU (TensorRT)"
                    logger = trt.Logger(trt.Logger.INFO)
                    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
                        self.trt_engine = runtime.deserialize_cuda_engine(f.read())
                    print(f"LWCC backend loaded into TensorRT: {engine_path}")
                    
                    err, self.trt_stream = cudart.cudaStreamCreate()
                    if int(err) != 0:
                        raise RuntimeError(f"CUDA Stream Creation failed: {err}")
                        
                    self.trt_context = self.trt_engine.create_execution_context()
                    
                    for i in range(self.trt_engine.num_io_tensors):
                        name = self.trt_engine.get_tensor_name(i)
                        dtype = self.trt_engine.get_tensor_dtype(name)
                        shape = list(self.trt_engine.get_tensor_shape(name))
                        
                        is_density = any(word in self.model_name.lower() or word in engine_path.lower() for word in ["dm_count", "csrnet", "sfanet", "bay", "density"])

                        # Fix for dynamic/corrupted shapes
                        if any(s < 0 or s > 1e6 for s in shape):
                            try:
                                # On cherche le max dans tous les profils disponibles
                                for p_idx in range(self.trt_engine.num_optimization_profiles):
                                    p_shape = self.trt_engine.get_tensor_profile_shape(name, p_idx)
                                    if p_shape and len(p_shape) >= 3:
                                        # On prend la forme OPTIMALE (index 1 dans p_shape)
                                        opt_shape = p_shape[1] 
                                        for j in range(len(shape)):
                                            if shape[j] < 0 or shape[j] > 1e6:
                                                shape[j] = opt_shape[j]
                                print(f"[DEBUG] Density: Resolved shape for {name} from profile: {shape}")
                            except Exception as e:
                                print(f"[WARN] Profile shape lookup failed for {name}: {e}")
                                # Fallback structurel pour DM-Count/CSRNet (Batch accru et résolution optimisée)
                                b_max = 8
                                if "input" in name.lower():
                                    shape = [b_max, 3, 544, 960] if is_density else [b_max, 3, 640, 640]
                                else:
                                    # Pour DM-Count, la sortie est divisée par 8
                                    shape = [b_max, 1, 544 // 8, 960 // 8] if is_density else [b_max, 1, 640, 640]
                                print(f"[DEBUG] Density: Using fallback shape for {name}: {shape}")
                        
                        # Si c'est un modèle de densité, on garde la résolution d'origine (1080p recommandé pour précision)
                        # La réduction à 544p entraînait des sur-comptages sur les gros plans
                        if is_density and "input" in name.lower() and shape[2] == 1080:
                            print(f"[INFO] Performance: Using native resolution for density model {name} (1080p).")
                            shape[2] = 1080
                            shape[3] = 1920

                        shape = tuple(shape)
                        print(f"[DEBUG] Density: Final allocation shape for {name}: {shape}")
                            
                        # Final check to avoid huge allocations
                        size = 1
                        for s in shape: size *= s
                        if size > 1e9: # > 1B elements is suspicious
                             print(f"[WARN] Suspiciously large shape for {name}: {shape}. Clamping.")
                             shape = [4, 3, 1080, 1920] if i==0 else [4, 1, 1080, 1920]

                        shape = tuple(shape)
                        dtype_np = trt.nptype(dtype)
                        nbytes = trt.volume(shape) * np.dtype(dtype_np).itemsize
                        
                        err, device_ptr = cudart.cudaMalloc(nbytes)
                        if int(err) != 0:
                            raise RuntimeError(f"CUDA Malloc failed for {name}: {err}")
                            
                        host_mem = np.empty(shape, dtype=dtype_np)
                        binding = {'name': name, 'dtype': dtype_np, 'shape': shape, 'size': trt.volume(shape), 'nbytes': nbytes, 'host': host_mem, 'device': device_ptr}
                        
                        if self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                            self.trt_inputs.append(binding)
                            self.trt_input_shape = shape
                        else:
                            self.trt_outputs.append(binding)
                        
                        self.trt_context.set_tensor_address(name, int(device_ptr))
                    print(f"PeopleCounterProcessor using TensorRT on RTX: {engine_path}")
                except Exception as exc:
                    print(f"[WARN] TensorRT init failed ({exc}), falling back to openvino.")
                    self.backend = "openvino"
            else:
                self.backend = "openvino"

        if self.backend == "openvino":
            # On cherche d'abord dans models/openvino, puis dans le home
            xml_name = f"{model_name.lower().replace('-', '_')}_{model_weights.lower()}.xml"
            candidates = [
                os.path.join("models", "openvino", xml_name),
                os.path.join(str(Path.home()), ".lwcc", "openvino", xml_name)
            ]
            
            xml_path = None
            for c in candidates:
                if os.path.isfile(c):
                    xml_path = c
                    break

            if xml_path:
                try:
                    import openvino as ov
                    core = ov.Core()
                    ov_model = core.read_model(xml_path)
                    self.ov_compiled_model = core.compile_model(ov_model, self.openvino_device)
                    try:
                        self.last_device = self.ov_compiled_model.get_property("FULL_DEVICE_NAME")
                    except:
                        self.last_device = f"OpenVINO ({self.openvino_device})"
                    self.ov_input = self.ov_compiled_model.inputs[0]
                    self.ov_output = self.ov_compiled_model.outputs[0]
                    try:
                        self.ov_input_shape = list(self.ov_input.get_shape())
                    except Exception:
                        self.ov_input_shape = None
                    print(f"PeopleCounterProcessor using OpenVINO on {self.openvino_device}: {xml_path}")
                except Exception as exc:
                    print(f"[WARN] OpenVINO init failed ({exc}), falling back to torch.")
                    self.backend = "torch"
            else:
                print(f"[WARN] OpenVINO IR not found at {xml_path}, falling back to torch.")
                self.backend = "torch"

        if self.backend == "torch":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"PeopleCounterProcessor running on device: {device_str}")

    def process(self, frame):
        """
        Traite une seule image et retourne (frame, color_map, count, mask).
        """
        # Logic for single frame processing based on backend
        if self.backend == "tensorrt" and self.trt_context is not None:
             # Pour la densité, on tente d'optimiser en réduisant la résolution des tiles si nécessaire
             # Les modèles de densité supportent bien le passage à une résolution inférieure (ex: 540p)
             _, colors, counts, masks = self.process_batch([frame])
             return frame, colors[0], counts[0], masks[0]
        
        elif self.backend == "openvino" and self.ov_compiled_model is not None:
            # Code OpenVINO optimisé pour une seule frame
            with self.lock:
                # Preprocess
                h, c, target_h, target_w = self.ov_input_shape or (1, 3, 1080, 1920)
                blob = cv2.resize(frame, (target_w, target_h))
                blob = blob.transpose((2, 0, 1)) # HWC to CHW
                blob = blob.reshape((1, 3, target_h, target_w))
                
                # Inference
                results = self.ov_compiled_model(blob)[self.ov_output]
                density_map = results[0, 0]
                
                # Post-process (identique à process_batch)
                d_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                d_norm = cv2.resize(d_norm, (frame.shape[1], frame.shape[0]))
                d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
                _, mask = cv2.threshold(d_norm, self.density_threshold, 255, cv2.THRESH_BINARY)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                d_color = cv2.bitwise_and(d_color, mask_3ch)
                
                count = int(np.sum(density_map))
                if count < 0.5: count = 0
                return frame, d_color, count, mask

        else:
            # Fallback PyTorch / LWCC direct
            # On utilise lwcc.get_count_from_frame si dispo; sinon fallback à 0
            global LWCC_AVAILABLE
            if 'LWCC_AVAILABLE' in globals() and LWCC_AVAILABLE:
                try:
                    # Force CPU pour éviter des conflits CUDA si on est en mode FALLBACK
                    count = get_count_from_frame(frame, model_name=self.model_name, model_weights=self.model_weights, return_density=False, use_gpu=False)
                except Exception as e:
                    print(f"[ERROR] LWCC CPU Inference failed: {e}")
                    # disable further attempts to avoid log spam
                    LWCC_AVAILABLE = False
                    count = 0
            else:
                count = 0
            
            # Pour torch, on va juste retourner un masque vide ou simulé
            return frame, np.zeros_like(frame), int(count), np.zeros(frame.shape[:2], dtype=np.uint8)

    def process_batch(self, frames):
        """
        Traite un lot (batch) d'images en une seule passe GPU si possible.
        """
        num_frames = len(frames)
        if num_frames == 0:
            return [], [], [], []

        if self.backend == "tensorrt" and self.trt_context is not None:
            # Récupération des dimensions attendues (Batch, C, H, W)
            b, c, target_h, target_w = self.trt_input_shape
            max_batch = b if b > 0 else num_frames 
            
            all_counts = []
            all_colors = []
            all_masks = []
            self.last_perf = {'preprocess': 0, 'h2d': 0, 'infer': 0, 'd2h': 0, 'postprocess': 0}

            with self.lock:
                def process_quad_local(f, target_h, target_w):
                    if f.shape[0] != target_h or f.shape[1] != target_w:
                        f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    img_floated = img_rgb.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    img_floated = (img_floated - mean) / std
                    return np.transpose(img_floated, (2, 0, 1))

                for i in range(0, num_frames, max_batch):
                    batch_chunk = frames[i:i+max_batch]
                    actual_b = len(batch_chunk)
                    
                    t0 = time.time()
                    batch_imgs = np.zeros((actual_b, c, target_h, target_w), dtype=np.float32)
                    processed = [process_quad_local(f, target_h, target_w) for f in batch_chunk]
                    for j, img in enumerate(processed): batch_imgs[j] = img
                    input_data = np.ascontiguousarray(batch_imgs)
                    self.last_perf['preprocess'] += time.time() - t0
                    
                    # Copy to device
                    required_size = input_data.nbytes
                    t1 = time.time()
                    self.cudart.cudaMemcpyAsync(self.trt_inputs[0]['device'], input_data.ctypes.data, required_size, self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream)
                    self.last_perf['h2d'] += time.time() - t1

                    t2 = time.time()
                    self.trt_context.set_input_shape(self.trt_inputs[0]['name'], input_data.shape)
                    self.trt_context.execute_async_v3(self.trt_stream)
                    self.cudart.cudaStreamSynchronize(self.trt_stream)
                    self.last_perf['infer'] += time.time() - t2
                    
                    t3 = time.time()
                    for output in self.trt_outputs:
                        self.cudart.cudaMemcpyAsync(output['host'].ctypes.data, output['device'], output['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream)
                    self.cudart.cudaStreamSynchronize(self.trt_stream)
                    self.last_perf['d2h'] += time.time() - t3
                    
                    t4 = time.time()
                    res = self.trt_outputs[0]['host'] # Buffer d'allocation
                    out_name = self.trt_outputs[0]['name']
                    out_shape = self.trt_context.get_tensor_shape(out_name)
                    real_h, real_w = out_shape[2], out_shape[3]
                    
                    for j in range(actual_b):
                        # On ne prend que la zone réellement générée par le modèle
                        density_map = res[j, 0, :real_h, :real_w]
                        all_counts.append(float(np.sum(density_map)))
                        all_colors.append(density_map.copy())
                    self.last_perf['postprocess'] += time.time() - t4

            # Post-processing commun
            t_final_start = time.time()
            final_colors, final_counts, final_masks = [], [], []
            for idx, density in enumerate(all_colors):
                orig_frame = frames[idx]
                
                # --- FILTRAGE DU BRUIT ---
                density[density < 0.001] = 0
                
                d_max = density.max()
                if d_max > 0.005:
                    d_norm = np.clip((density / max(d_max, 0.05)) * 255, 0, 255).astype(np.uint8)
                else:
                    d_norm = np.zeros(density.shape, dtype=np.uint8)

                # --- OPTIMISATION : COLORMAP SUR PETITE ÉCHELLE ---
                # On applique le colormap sur la petite image de densité d'abord
                d_color_small = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
                _, mask_small = cv2.threshold(d_norm, self.density_threshold, 255, cv2.THRESH_BINARY)
                
                # Masquage sur la petite version
                mask_3ch_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                d_color_small = cv2.bitwise_and(d_color_small, mask_3ch_small)

                # Resize final vers 4K (plus rapide sur l'image colorée qu'appliquer colormap sur 4K)
                d_color = cv2.resize(d_color_small, (orig_frame.shape[1], orig_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask_small, (orig_frame.shape[1], orig_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                c = float(np.sum(density))
                if c < 0.2: c = 0.0
                
                final_colors.append(d_color)
                final_counts.append(c)
                final_masks.append(mask)

            self.last_perf['postprocess'] += time.time() - t_final_start
            print(f"  [DENS DETAILED] Pre={self.last_perf['preprocess']*1000:.1f}ms | Infer={self.last_perf['infer']*1000:.1f}ms | Post={self.last_perf['postprocess']*1000:.1f}ms")
            
            return frames, final_colors, final_counts, final_masks

        elif self.backend == "openvino" and self.ov_compiled_model is not None:
            # Traitement par batch optimisé pour OpenVINO
            num_frames = len(frames)
            all_counts = [0.0] * num_frames
            all_colors = [None] * num_frames
            all_masks = [None] * num_frames
            
            # Dimensions attendues
            is_density = any(word in self.model_name.lower() for word in ["dm_count", "csrnet", "sfanet", "bay", "density"])
            try:
                ov_shape = list(self.ov_input.get_shape())
                expected_b = ov_shape[0] if ov_shape[0] > 0 else num_frames
                c, target_h, target_w = ov_shape[1], ov_shape[2], ov_shape[3]
                
                # Optimisation de résolution pour la densité
                if is_density and target_h == 1080:
                    target_h, target_w = 544, 960
            except Exception:
                if is_density:
                    expected_b, c, target_h, target_w = num_frames, 3, 544, 960
                else:
                    expected_b, c, target_h, target_w = num_frames, 3, 1080, 1920

            # Optimisation: limiter le batch max pour éviter les pics de RAM
            max_ov_batch = min(expected_b, 4) 
            
            with self.lock:
                for i in range(0, num_frames, max_ov_batch):
                    chunk = frames[i:i+max_ov_batch]
                    actual_b = len(chunk)
                    
                    # Preprocess batch
                    input_blob = np.zeros((actual_b, c, target_h, target_w), dtype=np.float32)
                    for j, f in enumerate(chunk):
                        blob = cv2.resize(f, (target_w, target_h))
                        blob = blob.transpose((2, 0, 1)) # HWC to CHW
                        input_blob[j] = blob.astype(np.float32) / 255.0

                    # Inférence batch (Optimisée OpenVINO)
                    try:
                        res = self.ov_compiled_model(input_blob)[self.ov_output]
                    except Exception as e:
                        # Fallback séquentiel
                        res_shape = list(self.ov_output.get_shape())
                        res = np.zeros((actual_b, res_shape[1], res_shape[2], res_shape[3]), dtype=np.float32)
                        for j in range(actual_b):
                            single_blob = input_blob[j:j+1]
                            res[j:j+1] = self.ov_compiled_model(single_blob)[self.ov_output]
                    
                    for j in range(actual_b):
                        density_map = res[j, 0]
                        
                        # Post-process (identique au mode single)
                        d_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        d_norm = cv2.resize(d_norm, (chunk[j].shape[1], chunk[j].shape[0]))
                        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
                        _, mask = cv2.threshold(d_norm, self.density_threshold, 255, cv2.THRESH_BINARY)
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        d_color = cv2.bitwise_and(d_color, mask_3ch)
                        
                        count = float(np.sum(density_map))
                        if count < 0.5: count = 0
                        
                        abs_idx = i + j
                        all_counts[abs_idx] = count
                        all_colors[abs_idx] = d_color
                        all_masks[abs_idx] = mask
            
            return frames, all_colors, all_counts, all_masks
            
        else:
            # Fallback PyTorch
            all_counts, all_colors, final_masks = [], [], []
            for f in frames:
                _, c_map, count, mask = self.process(f)
                all_colors.append(c_map)
                all_counts.append(count)
                final_masks.append(mask)
            return frames, all_colors, all_counts, final_masks

import os
# Use local vendors folder for lwcc if present; do not hardcode Windows paths
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
from logger.filtered_logger import LogChannel, warning as log_warning

from cuda_stream_manager import CudaStreamManager
from trt_debug import dump_trt_bindings
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
        self.density_threshold = density_threshold
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_bindings = []
        self.cudart = None
        self.stream_mgr = None
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
                    _maybe_dump_binding_shapes(self.trt_engine, "Density")
                    print(f"LWCC backend loaded into TensorRT: {engine_path}")
                        print(f"LWCC backend loaded into TensorRT: {engine_path}")
                        from trt_debug import dump_trt_allocation_info
                        dump_trt_allocation_info(self.trt_engine, "Density")
                    self.stream_mgr = CudaStreamManager(name="density_trt")
                    self.stream_mgr.ensure_valid()
                    if not self.stream_mgr.handle():
                        raise RuntimeError("CUDA Stream unavailable via CudaStreamManager")

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
                                # Fallback structurel (Batch 8 par défaut)
                                b_max = 8
                                if "input" in name.lower():
                                    # Calibré sur 544p (multiple de 16) pour le tiling 1080p natif
                                    shape = [b_max, 3, 544, 960] if is_density else [b_max, 3, 640, 640]
                                else:
                                    shape = [b_max, 1, 544 // 8, 960 // 8] if is_density else [b_max, 1, 640, 640]
                                print(f"[DEBUG] Density: Using fallback shape for {name}: {shape}")
                        
                        # Si c'est un modèle de densité, on respecte la calibration 544p demandée pour le tiling
                        if is_density and "input" in name.lower() and (shape[2] == 1080 or shape[2] == 544):
                            print(f"[INFO] Performance: Using calibrated 544p resolution for density model (Tiling Optimization).")
                            shape[2] = 544
                            shape[3] = 960
                        elif is_density and "input" in name.lower() and shape[2] > 1080:
                            # Sécurité 4K
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
                    print(f"[WARN] TensorRT init failed ({exc}), falling back to torch.")
                    self.backend = "torch"
        else:
            self.backend = "torch"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PeopleCounterProcessor running on device: {device_str}")

    def _recreate_trt_stream(self):
        if not self.stream_mgr:
            return
        try:
            self.stream_mgr.reset()
        except Exception as exc:
            log_warning(LogChannel.DENSITY, f"Failed to recreate CUDA stream: {exc}")
            raise

    def process(self, frame):
        """
        Traite une seule image et retourne (frame, color_map, count, mask).
        """
        # Logic for single frame processing based on backend
        if self.backend == "tensorrt" and self.trt_context is not None:
             # Pour la densité, on tente d'optimiser en réduisant la résolution des tiles si nécessaire
             # Les modèles de densité supportent bien le passage à une résolution inférieure (ex: 540p)
             result = self.process_batch([frame])
             tiles = result.get('tiles', [])
             if not tiles:
                 empty_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                 return frame, np.zeros_like(frame), 0, empty_mask
             tile = tiles[0]
             mask = tile.get('mask')
             if mask is None:
                 mask = np.zeros(frame.shape[:2], dtype=np.uint8)
             mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
             color_small = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
             color_resized = cv2.resize(color_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
             return frame, color_resized, tile.get('count', 0.0), mask_resized
        
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
        if not frames:
            return {'total_count': 0.0, 'tiles': [], 'frame_width': 0, 'frame_height': 0}

        frame_h, frame_w = frames[0].shape[:2]

        if self.backend == "tensorrt" and self.trt_context is not None:
            b, c, target_h, target_w = self.trt_input_shape
            max_batch = b if b > 0 else len(frames)
            tiles = []
            total_count = 0.0
            self.last_perf = {'preprocess': 0, 'h2d': 0, 'infer': 0, 'd2h': 0, 'postprocess': 0}

            def _run_trt_batches():
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

                    for i in range(0, len(frames), max_batch):
                        batch_chunk = frames[i:i+max_batch]
                        actual_b = len(batch_chunk)

                        t0 = time.time()
                        batch_imgs = np.zeros((actual_b, c, target_h, target_w), dtype=np.float32)
                        processed = [process_quad_local(f, target_h, target_w) for f in batch_chunk]
                        for j, img in enumerate(processed):
                            batch_imgs[j] = img
                        input_data = np.ascontiguousarray(batch_imgs)
                        self.last_perf['preprocess'] += time.time() - t0

                        required_size = input_data.nbytes
                        t1 = time.time()
                        self.stream_mgr.ensure_valid()
                        stream_handle = self.stream_mgr.handle()
                        if not stream_handle:
                            raise RuntimeError("TensorRT CUDA stream unavailable")
                        self.cudart.cudaMemcpyAsync(
                            self.trt_inputs[0]['device'], input_data.ctypes.data, required_size,
                            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream_handle
                        )
                        self.last_perf['h2d'] += time.time() - t1

                        t2 = time.time()
                        self.trt_context.set_input_shape(self.trt_inputs[0]['name'], input_data.shape)
                        try:
                            self.trt_context.execute_async_v3(stream_handle)
                        except Exception as exc:
                            log_warning(LogChannel.DENSITY, f"TensorRT enqueueV3 failed ({exc}); recreating stream.")
                            self._recreate_trt_stream()
                            self.stream_mgr.ensure_valid()
                            stream_handle = self.stream_mgr.handle()
                            if not stream_handle:
                                raise RuntimeError("TensorRT CUDA stream unavailable after recreation")
                            self.trt_context.execute_async_v3(stream_handle)
                        self.stream_mgr.synchronize()
                        self.last_perf['infer'] += time.time() - t2

                        t3 = time.time()
                        for output in self.trt_outputs:
                            self.cudart.cudaMemcpyAsync(
                                output['host'].ctypes.data, output['device'], output['nbytes'],
                                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream_handle
                            )
                        self.stream_mgr.synchronize()
                        self.last_perf['d2h'] += time.time() - t3

                        t4 = time.time()
                        res = self.trt_outputs[0]['host']
                        out_name = self.trt_outputs[0]['name']
                        out_shape = self.trt_context.get_tensor_shape(out_name)
                        real_h, real_w = out_shape[2], out_shape[3]

                        for j in range(actual_b):
                            density_map = res[j, 0, :real_h, :real_w]
                            frame_idx = i + j
                            tile_entry = self._create_tile_entry(density_map, frames[frame_idx].shape[:2])
                            tiles.append(tile_entry)
                            total_count += tile_entry['count']
                        self.last_perf['postprocess'] += time.time() - t4

            _run_trt_batches()

            print(
                f"  [DENS DETAILED] Pre={self.last_perf['preprocess']*1000:.1f}ms | "
                f"Infer={self.last_perf['infer']*1000:.1f}ms | Post={self.last_perf['postprocess']*1000:.1f}ms"
            )
            return {'total_count': total_count, 'tiles': tiles, 'frame_width': frame_w, 'frame_height': frame_h}

        if self.backend == "torch":
            tiles = []
            total_count = 0.0
            for frame in frames:
                _, _, count, mask = self.process(frame)
                density_map = mask.astype(np.float32) if mask is not None else np.zeros((1, 1), dtype=np.float32)
                tile_entry = self._create_tile_entry(density_map, frame.shape[:2])
                if mask is not None:
                    tile_entry['mask'] = mask
                tile_entry['count'] = float(count)
                tiles.append(tile_entry)
                total_count += float(count)
            return {'total_count': total_count, 'tiles': tiles, 'frame_width': frame_w, 'frame_height': frame_h}
        tiles = [self._build_empty_tile_entry(f.shape[:2]) for f in frames]
        return {'total_count': 0.0, 'tiles': tiles, 'frame_width': frame_w, 'frame_height': frame_h}

    def _normalize_density_mask(self, density_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(density_map, dtype=np.float32, order='C').copy()
        array[array < 0.001] = 0
        max_val = array.max()
        if max_val > 0.005:
            scale = 255.0 / max(max_val, 0.05)
            normalized = np.clip(array * scale, 0, 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(array, dtype=np.uint8)
        mask = np.zeros_like(normalized, dtype=np.uint8)
        if self.density_threshold is not None and normalized.size:
            mask = (normalized >= self.density_threshold).astype(np.uint8) * 255
        return normalized, mask

    def _create_tile_entry(self, density_map: np.ndarray, frame_shape: tuple[int, int]) -> dict:
        heatmap, mask = self._normalize_density_mask(density_map)
        heatmap_h, heatmap_w = heatmap.shape
        mask_h, mask_w = mask.shape
        tile_h, tile_w = frame_shape
        scale_x = (tile_w / mask_w) if mask_w else 1.0
        scale_y = (tile_h / mask_h) if mask_h else 1.0
        total = float(np.sum(density_map))
        if total < 0.2:
            total = 0.0
        return {
            'count': total,
            'mask': mask,
            'heatmap': heatmap,
            'mask_width': mask_w,
            'mask_height': mask_h,
            'tile_width': tile_w,
            'tile_height': tile_h,
            'scale_x': scale_x,
            'scale_y': scale_y,
        }

    def _build_empty_tile_entry(self, frame_shape: tuple[int, int]) -> dict:
        tile_h, tile_w = frame_shape
        mask = np.zeros((1, 1), dtype=np.uint8)
        heatmap = mask.copy()
        return {
            'count': 0.0,
            'mask': mask,
            'heatmap': heatmap,
            'mask_width': 1,
            'mask_height': 1,
            'tile_width': tile_w,
            'tile_height': tile_h,
            'scale_x': float(tile_w),
            'scale_y': float(tile_h),
        }

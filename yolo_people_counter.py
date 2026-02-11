from ultralytics import YOLO
import numpy as np
import cv2
import os
import sys
import time
from tiling_manager import TilingManager
from logger.filtered_logger import LogChannel, warning as log_warning
from cuda_stream_manager import CudaStreamManager
from trt_debug import dump_trt_bindings
from trt_output_reconstructor import dump_flattened, flatten_trt_output
from trt_output_reconstructor import flatten_trt_output

class YoloPeopleCounter:
    def __init__(self, model_name='yolov26n', device='cpu', confidence_threshold=0.3, backend='torch', tile_size=640, overlap=0.15):
        """
        model_name: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'yolo11n', etc.
        device: 'cpu' or 'cuda'
        confidence_threshold: float, minimum confidence for detection
        """
        self.backend = (backend or 'torch').lower()
        log_warning(LogChannel.YOLO, f"YOLO backend: {self.backend}")
        self.device = device
        self.person_class_id = 0  # COCO: 0 = person
        self.confidence_threshold = confidence_threshold
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiler = TilingManager(tile_size=tile_size) # On ne passe plus overlap au constructeur si TilingManager calcule tout seul
        
        self.model = None
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_bindings = []
        self.trt_stream = None
        self.stream_mgr = None
        self.last_perf = {}
        self._dumped_trt_outputs = False

        if self.backend == 'tensorrt_native':
            self.model_name = model_name
            import tensorrt as trt
            from cuda.bindings import runtime as cudart
            self.cudart = cudart
            
            self.trt_logger = trt.Logger(trt.Logger.INFO)
            with open(model_name, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            dump_trt_bindings(self.trt_engine, "YOLO")
            
            self.stream_mgr = CudaStreamManager(name="yolo_trt")
            if not self._stream_handle():
                raise RuntimeError("CUDA stream initialization failed via CudaStreamManager")
            
            print(f"YOLO backend loaded into TensorRT: {model_name}")
            self.trt_context = self.trt_engine.create_execution_context()
            
            for i in range(self.trt_engine.num_io_tensors):
                name = self.trt_engine.get_tensor_name(i)
                dtype = self.trt_engine.get_tensor_dtype(name)
                shape = list(self.trt_engine.get_tensor_shape(name))
                
                # Gestion des dimensions dynamiques via les profils
                if any(d < 0 or d > 1e6 for d in shape):
                    try:
                        # On cherche le max dans tous les profils disponibles
                        for p_idx in range(self.trt_engine.num_optimization_profiles):
                            p_shape = self.trt_engine.get_tensor_profile_shape(name, p_idx)
                            if p_shape and len(p_shape) >= 3:
                                max_shape = p_shape[2]
                                for j in range(len(shape)):
                                    if shape[j] < 0 or shape[j] > 1e6:
                                        shape[j] = max(shape[j], max_shape[j])
                        print(f"[DEBUG] YOLO: Resolved shape for {name}: {shape}")
                    except Exception as e:
                        print(f"[WARN] Failed to get profile shape for {name}: {e}")
                        # Fallback structurel structurel pour YOLO
                        # On essaye de deviner la taille max (32 pour le batch)
                        b_max = 32
                        if "output" in name.lower():
                            shape = [b_max, 84, 8400]
                        else:
                            shape = [b_max, 3, 640, 640]
                        print(f"[DEBUG] YOLO: Using fallback shape for {name}: {shape}")
                
                # Nettoyage final pour s'assurer qu'on n'a plus de -1
                for j in range(len(shape)):
                    if shape[j] < 1 or shape[j] > 1e6:
                        shape[j] = 1 
                
                shape = tuple(shape)
                print(f"[DEBUG] YOLO: Final allocation shape for {name}: {shape}")
                dtype_np = trt.nptype(dtype)
                size = trt.volume(shape)
                nbytes = size * np.dtype(dtype_np).itemsize
                
                # Allocate device memory
                err, device_ptr = cudart.cudaMalloc(nbytes)
                if int(err) != 0:
                    raise RuntimeError(f"CUDA Malloc failed: {err}")
                
                # Allocation host memory
                host_mem = np.empty(shape, dtype=dtype_np)
                
                binding = {
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'size': size,
                    'nbytes': nbytes,
                    'host': host_mem,
                    'device': device_ptr
                }
                
                if self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.trt_inputs.append(binding)
                    self.trt_input_shape = shape
                else:
                    self.trt_outputs.append(binding)
                
                self.trt_context.set_tensor_address(name, int(device_ptr))

        else:
            self.model_name = model_name
            self.model = YOLO(model_name, task='detect')
            self.model.to(device)

    def _recreate_trt_stream(self):
        if not self.stream_mgr or not hasattr(self, 'cudart'):
            return
        try:
            self.stream_mgr.reset()
        except Exception as exc:
            log_warning(LogChannel.YOLO, f"Failed to recreate CUDA stream: {exc}")
            raise
        self._stream_handle()

    def _stream_handle(self):
        if not self.stream_mgr:
            return None
        ensure_valid = getattr(self.stream_mgr, 'ensure_valid', None)
        if callable(ensure_valid):
            ensure_valid()
        stream = self.stream_mgr.handle()
        self.trt_stream = stream
        return stream

    def _synchronize_current_stream(self):
        sync_manager = getattr(self.stream_mgr, 'synchronize', None)
        if callable(sync_manager):
            sync_manager()
            return
        if not (getattr(self, 'cudart', None) and self.trt_stream):
            return
        self.cudart.cudaStreamSynchronize(self.trt_stream)

    def _maybe_dump_trt_outputs(self):
        if self._dumped_trt_outputs:
            return
        flag = os.environ.get('PEOPLE_COUNTER_DUMP_TRT_BINDINGS', '0').lower()
        if flag not in ('1', 'true', 'yes'):
            return
        for idx, binding in enumerate(self.trt_outputs):
            dump_flattened(binding, f'YOLO output {idx}')
        self._dumped_trt_outputs = True
        print('[DBG] Dumped TensorRT outputs, exiting per debug flag')
        sys.exit(0)

    def count_people(self, image, tile_size=640, draw_boxes=True, use_tiling=True, draw_tiles=False, global_tile_only=False):
        """
        Détection de personnes avec pipeline grille dense optimisée.
        """
        # Conversion RGB pour la cohérence des modèles YOLO
        image_rgb = image[..., ::-1].copy()
        if global_tile_only:
            use_tiling = False
            draw_tiles = False
        
        if not use_tiling:
            # Inference sur image entière redimensionnée (beaucoup plus rapide)
            if self.backend == 'tensorrt_native':
                boxes, scores = self._trt_native_infer(image_rgb)
                keep = self.tiler.fuse_results(boxes, scores, iou_threshold=0.25)
                total_count = len(keep)
                image_out = image.copy() if draw_boxes else None
                if draw_boxes:
                    for i in keep:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                results = self.model.predict(image_rgb, verbose=False, imgsz=tile_size)
                r = results[0]
                image_out = image.copy() if draw_boxes else None
                clusters = []
                boxes = np.zeros((0, 4), dtype=np.float32)
                if hasattr(r, 'boxes'):
                    person_mask = (r.boxes.cls.cpu().numpy() == self.person_class_id) & (r.boxes.conf.cpu().numpy() >= self.confidence_threshold)
                    selected = np.where(person_mask)[0]
                    if selected.size:
                        boxes = r.boxes.xyxy[selected].cpu().numpy()
                        scores = r.boxes.conf[selected].cpu().numpy()
                        clusters = self.tiler.fuse_results(boxes, scores, iou_threshold=0.25)
                total_count = len(clusters)
                if draw_boxes and image_out is not None:
                    for cluster in clusters:
                        idx = cluster[0] if isinstance(cluster, (list, tuple, np.ndarray)) else cluster
                        x1, y1, x2, y2 = boxes[idx].astype(int)
                        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if draw_boxes:
                    return int(total_count), image_out
                return int(total_count)

        # Utilisation du TilingManager pour une grille dense (ex: 7x4 pour 4K)
        tiles_coords = self.tiler.get_tiles(image_rgb.shape)
        
        tiles_to_infer = []
        tile_metadata = [] # (x, y, sw, sh)
        for x1, y1, x2, y2 in tiles_coords:
            tw, th = x2 - x1, y2 - y1
            tiles_to_infer.append(image_rgb[y1:y2, x1:x2])
            tile_metadata.append((x1, y1, tw/self.tile_size, th/self.tile_size))

        all_boxes_list = []
        all_scores_list = []

        # --- ÉTAPE 2 : INFÉRENCE BATCHÉE ---
        if self.backend == 'tensorrt_native':
            batch_boxes, batch_scores = self._trt_native_infer_batch(tiles_to_infer)
            for i, (boxes, scores) in enumerate(zip(batch_boxes, batch_scores)):
                if boxes.size > 0:
                    x0, y0, sw, sh = tile_metadata[i]
                    if sw != 1.0 or sh != 1.0:
                        boxes[:, [0, 2]] *= sw
                        boxes[:, [1, 3]] *= sh
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes_list.append(boxes)
                    all_scores_list.append(scores)
        else:
            for i, tile in enumerate(tiles_to_infer):
                results = self.model.predict(tile, verbose=False, imgsz=self.tile_size)
                mask = (results[0].boxes.cls.cpu().numpy() == self.person_class_id) & (results[0].boxes.conf.cpu().numpy() >= self.confidence_threshold)
                boxes = results[0].boxes.xyxy[mask].cpu().numpy()
                scores = results[0].boxes.conf[mask].cpu().numpy()
                
                if boxes.size > 0:
                    x0, y0, sw, sh = tile_metadata[i]
                    if sw != 1.0 or sh != 1.0:
                        boxes[:, [0, 2]] *= sw
                        boxes[:, [1, 3]] *= sh
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes_list.append(boxes)
                    all_scores_list.append(scores)

        if not all_boxes_list:
            final_img = image.copy()
            if draw_tiles:
                for t1, t1y, t2, t2y in tiles_coords:
                    cv2.rectangle(final_img, (t1, t1y), (t2, t2y), (255, 0, 0), 2)
            return (0, final_img) if draw_boxes else 0

        # Concaténation et NMS via TilingManager
        all_boxes = np.concatenate(all_boxes_list, axis=0)
        all_scores = np.concatenate(all_scores_list, axis=0)
        
        clusters = self.tiler.fuse_results(all_boxes, all_scores, iou_threshold=0.3)

        total_count = len(clusters)
        image_out = image.copy() if draw_boxes else None
        if draw_boxes:
            # Debug Tiling YOLO (Bleu)
            if draw_tiles:
                for x1, y1, x2, y2 in tiles_coords:
                    cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 2)

            for cluster in clusters:
                idx = cluster[0] if isinstance(cluster, (list, tuple, np.ndarray)) else cluster
                x1, y1, x2, y2 = all_boxes[int(idx)].astype(int)
                cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return (int(total_count), image_out) if draw_boxes else int(total_count)

    def _trt_native_infer(self, tile_rgb):
        import numpy as np
        h_orig, w_orig = tile_rgb.shape[:2]

        pre_start = time.monotonic()
        img = tile_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        target_w, target_h = w_orig, h_orig
        if self.trt_input_shape and len(self.trt_input_shape) == 4:
            _, _, target_h, target_w = self.trt_input_shape
            if img.shape[2] != target_h or img.shape[3] != target_w:
                resized = cv2.resize(tile_rgb, (target_w, target_h))
                img = resized.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[None, ...]

        input_data = np.ascontiguousarray(img)
        pre_end = time.monotonic()
        self.last_perf['preprocess'] = pre_end - pre_start

        infer_start = time.monotonic()
        cuda_input = self.trt_inputs[0]
        stream_handle = self._stream_handle()
        if not stream_handle:
            raise RuntimeError("TensorRT CUDA stream unavailable")

        self.cudart.cudaMemcpyAsync(
            cuda_input['device'],
            input_data.ctypes.data,
            input_data.nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream_handle,
        )
        self.trt_context.set_input_shape(cuda_input['name'], input_data.shape)

        def stream_is_capturing():
            return bool(self.stream_mgr and self.stream_mgr.is_capturing())

        try:
            success = self.trt_context.execute_async_v3(stream_handle)
            if not success:
                raise RuntimeError("TensorRT execute_async_v3 returned False")
        except Exception as exc:
            reason = "capture" if stream_is_capturing() else "execute_async_v3 failure"
            log_warning(LogChannel.YOLO, f"TensorRT stream {reason}; recreating stream ({exc})")
            self._recreate_trt_stream()
            stream_handle = self._stream_handle()
            success = self.trt_context.execute_async_v3(stream_handle)
            if not success:
                raise RuntimeError("TensorRT execute_async_v3 failed even after recreating stream")

        for output in self.trt_outputs:
            self.cudart.cudaMemcpyAsync(
                output['host'].ctypes.data,
                output['device'],
                output['nbytes'],
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream_handle,
            )
        self._synchronize_current_stream()
        self._maybe_dump_trt_outputs()
        infer_end = time.monotonic()
        self.last_perf['infer'] = infer_end - infer_start

        post_start = time.monotonic()
        binding = self.trt_outputs[0]
        pred, _ = flatten_trt_output(binding)
        boxes = pred[:, :4]
        scores = pred[:, 4 + self.person_class_id]
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        if boxes.size == 0:
            post_end = time.monotonic()
            self.last_perf['postprocess'] = post_end - post_start
            return boxes, scores

        sw = w_orig / target_w
        sh = h_orig / target_h
        x = boxes[:, 0] * sw
        y = boxes[:, 1] * sh
        w = boxes[:, 2] * sw
        h = boxes[:, 3] * sh
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        post_end = time.monotonic()
        self.last_perf['postprocess'] = post_end - post_start
        return boxes, scores

    def _trt_native_infer_batch(self, batch_imgs):
        """
        Inférence YOLO en Batch pour TensorRT.
        batch_imgs: Liste d'images (tiles)
        """
        num_imgs = len(batch_imgs)
        if num_imgs == 0:
            return [], []

        target_h, target_w = self.trt_input_shape[2], self.trt_input_shape[3]
        max_batch = self.trt_input_shape[0] if self.trt_input_shape[0] > 0 else num_imgs

        all_boxes, all_scores = [], []
        perf_pre, perf_infer, perf_post = 0.0, 0.0, 0.0

        from concurrent.futures import ThreadPoolExecutor

        def process_tile(tile, target_h, target_w):
            if tile.shape[0] != target_h or tile.shape[1] != target_w:
                tile = cv2.resize(tile, (target_w, target_h))
            tile = tile.astype(np.float32)
            tile *= (1.0 / 255.0)
            return np.transpose(tile, (2, 0, 1))

        for i in range(0, num_imgs, max_batch):
            chunk = batch_imgs[i:i+max_batch]
            actual_b = len(chunk)

            pre_start = time.monotonic()
            processed = np.zeros((actual_b, 3, target_h, target_w), dtype=np.float32)

            with ThreadPoolExecutor(max_workers=min(8, actual_b)) as executor:
                futures = [executor.submit(process_tile, chunk[j], target_h, target_w) for j in range(actual_b)]
                for j, future in enumerate(futures):
                    processed[j] = future.result()

            input_data = np.ascontiguousarray(processed)
            pre_end = time.monotonic()
            perf_pre += pre_end - pre_start

            if self.trt_input_shape[0] > 0 and actual_b > self.trt_input_shape[0]:
                print(f"[WARN] Batch size {actual_b} exceeds engine max {self.trt_input_shape[0]}. Falling back to sequential.")
                for j in range(actual_b):
                    single_boxes, single_scores = self._trt_native_infer(chunk[j])
                    all_boxes.append(single_boxes)
                    all_scores.append(single_scores)
            else:
                infer_start = time.monotonic()
                stream_handle = self._stream_handle()
                if not stream_handle:
                    raise RuntimeError("TensorRT CUDA stream unavailable for batch inference")

                self.cudart.cudaMemcpyAsync(
                    self.trt_inputs[0]['device'],
                    input_data.ctypes.data,
                    input_data.nbytes,
                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream_handle,
                )
                self.trt_context.set_input_shape(self.trt_inputs[0]['name'], input_data.shape)

                def stream_is_capturing():
                    return bool(self.stream_mgr and self.stream_mgr.is_capturing())

                try:
                    success = self.trt_context.execute_async_v3(stream_handle)
                    if not success:
                        raise RuntimeError("TensorRT execute_async_v3 returned False")
                except Exception as exc:
                    reason = "capture" if stream_is_capturing() else "execute_async_v3 failure"
                    log_warning(LogChannel.YOLO, f"TensorRT stream {reason}; recreating stream ({exc})")
                    self._recreate_trt_stream()
                    stream_handle = self._stream_handle()
                    success = self.trt_context.execute_async_v3(stream_handle)
                    if not success:
                        raise RuntimeError("TensorRT execute_async_v3 failed even after recreating stream")

                for output in self.trt_outputs:
                    self.cudart.cudaMemcpyAsync(
                        output['host'].ctypes.data,
                        output['device'],
                        output['nbytes'],
                        self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream_handle,
                    )
                self._synchronize_current_stream()
                self._maybe_dump_trt_outputs()
                infer_end = time.monotonic()
                perf_infer += infer_end - infer_start

                post_start = time.monotonic()
                binding = self.trt_outputs[0]
                result, per_sample = flatten_trt_output(binding)
                num_predictions = per_sample or result.shape[0]
                for j in range(actual_b):
                    start = j * num_predictions
                    end = start + num_predictions
                    pred = result[start:end]
                    boxes = pred[:, :4]
                    scores = pred[:, 4 + self.person_class_id]
                    mask = scores >= self.confidence_threshold
                    boxes, scores = boxes[mask], scores[mask]

                    if boxes.size > 0:
                        h_orig, w_orig = chunk[j].shape[:2]
                        sw, sh = w_orig / target_w, h_orig / target_h

                        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * sw
                        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * sh
                        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * sw
                        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * sh
                        res_boxes = np.stack([x1, y1, x2, y2], axis=1)
                        all_boxes.append(res_boxes)
                        all_scores.append(scores)
                    else:
                        all_boxes.append(np.array([]))
                        all_scores.append(np.array([]))
                post_end = time.monotonic()
                perf_post += post_end - post_start

        self.last_perf['preprocess'] = perf_pre
        self.last_perf['infer'] = perf_infer
        self.last_perf['postprocess'] = perf_post
        return all_boxes, all_scores

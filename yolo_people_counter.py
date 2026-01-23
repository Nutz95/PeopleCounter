from ultralytics import YOLO
import numpy as np
import cv2
import os

class YoloPeopleCounter:
    def __init__(self, model_name='yolov26n', device='cpu', confidence_threshold=0.3, backend='torch'):
        """
        model_name: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'yolo11n', etc.
        device: 'cpu' or 'cuda'
        confidence_threshold: float, minimum confidence for detection
        """
        self.backend = (backend or 'torch').lower()
        self.device = device
        self.person_class_id = 0  # COCO: 0 = person
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.ov_compiled_model = None
        self.ov_input = None
        self.ov_output = None
        self.ov_input_shape = None
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_bindings = []
        self.trt_stream = None

        if self.backend == 'tensorrt_native':
            self.model_name = model_name
            import tensorrt as trt
            from cuda.bindings import runtime as cudart
            self.cudart = cudart
            
            self.trt_logger = trt.Logger(trt.Logger.INFO)
            with open(model_name, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.trt_engine = runtime.deserialize_cuda_engine(f.read())
            
            err, self.trt_stream = cudart.cudaStreamCreate()
            if int(err) != 0:
                raise RuntimeError(f"CUDA Stream Creation failed: {err}")
            
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

                shape = tuple(shape)
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

        elif self.backend == 'openvino_native':
            if os.path.isdir(model_name):
                xml_candidates = [p for p in os.listdir(model_name) if p.lower().endswith('.xml')]
                if xml_candidates:
                    model_name = os.path.join(model_name, xml_candidates[0])
            if not (os.path.isfile(model_name) and model_name.lower().endswith('.xml')):
                raise FileNotFoundError(f"OpenVINO XML not found for model: {model_name}")
            self.model_name = model_name
            import openvino as ov
            core = ov.Core()
            try:
                print(f"OpenVINO available devices: {core.available_devices}")
            except Exception:
                pass
            ov_model = core.read_model(model_name)
            self.ov_compiled_model = core.compile_model(ov_model, self.device)
            self.ov_input = self.ov_compiled_model.inputs[0]
            self.ov_output = self.ov_compiled_model.outputs[0]
            try:
                self.ov_input_shape = list(self.ov_input.get_shape())
            except Exception:
                self.ov_input_shape = None
            try:
                exec_devices = self.ov_compiled_model.get_property("EXECUTION_DEVICES")
                print(f"YOLO OpenVINO device: {exec_devices}")
            except Exception:
                pass
        else:
            if self.backend == 'openvino':
                if os.path.isfile(model_name) and model_name.lower().endswith('.xml'):
                    model_name = os.path.dirname(model_name)
            self.model_name = model_name
            self.model = YOLO(model_name, task='detect')
            if self.backend != 'openvino':
                self.model.to(device)

    def count_people(self, image, tile_size=640, draw_boxes=True, use_tiling=True):
        """
        Détection de personnes avec pipeline hiérarchique :
        1. Passe Globale (image entière redimensionnée)
        2. Passe Quadrants (si 4K)
        3. Passe Tuiles (Sliding window 640x640)
        """
        # Conversion RGB pour la cohérence des modèles YOLO
        image_rgb = image[..., ::-1].copy()
        
        if not use_tiling:
            # Inference sur image entière redimensionnée (beaucoup plus rapide)
            if self.backend == 'openvino_native':
                boxes, scores = self._ov_native_infer(image_rgb)
                keep = self._nms(boxes, scores, iou_threshold=0.25, inclusion_threshold=0.45)
                total_count = len(keep)
                image_out = image.copy() if draw_boxes else None
                if draw_boxes:
                    for i in keep:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif self.backend == 'tensorrt_native':
                boxes, scores = self._trt_native_infer(image_rgb)
                keep = self._nms(boxes, scores, iou_threshold=0.25, inclusion_threshold=0.45)
                total_count = len(keep)
                image_out = image.copy() if draw_boxes else None
                if draw_boxes:
                    for i in keep:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                results = self.model.predict(image_rgb, verbose=False, imgsz=tile_size)
                r = results[0]
                total_count = 0
                image_out = image.copy() if draw_boxes else None
                if hasattr(r, 'boxes'):
                    person_mask = (r.boxes.cls.cpu().numpy() == self.person_class_id) & (r.boxes.conf.cpu().numpy() >= self.confidence_threshold)
                    total_count = np.sum(person_mask)
                    if draw_boxes:
                        for i, is_person in enumerate(person_mask):
                            if is_person:
                                box = r.boxes.xyxy[i].cpu().numpy().astype(int)
                                cv2.rectangle(image_out, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            if draw_boxes:
                return int(total_count), image_out
            return int(total_count)

        # Si use_tiling=True, on combine une pyramide de détection Hiérarchique
        h, w = image_rgb.shape[:2]
        all_boxes_list = []
        all_scores_list = []

        # --- ÉTAPE 1 : COLLECTE DE TOUTES LES TUILES ---
        tiles_to_infer = []
        tile_metadata = [] # (x0, y0, scale_w, scale_h)

        # 1.1 Passe Globale
        tiles_to_infer.append(image_rgb)
        tile_metadata.append((0, 0, 1.0, 1.0))

        # 1.2 Passe Quadrants
        quad_h, quad_w = h // 2, w // 2
        for i in range(2):
            for j in range(2):
                y0, x0 = i * quad_h, j * quad_w
                quad = image_rgb[y0:y0+quad_h, x0:x0+quad_w]
                tiles_to_infer.append(quad)
                tile_metadata.append((x0, y0, quad_w/tile_size, quad_h/tile_size))

        # 1.3 Passe Tuiles (Sliding Window)
        overlap_ratio = 0.2
        step = int(tile_size * (1 - overlap_ratio))
        y_coords = list(range(0, max(1, h - tile_size + step), step))
        if y_coords[-1] + tile_size < h: y_coords.append(h - tile_size)
        x_coords = list(range(0, max(1, w - tile_size + step), step))
        if x_coords[-1] + tile_size < w: x_coords.append(w - tile_size)

        for y0 in y_coords:
            for x0 in x_coords:
                tile = image_rgb[y0:y0+tile_size, x0:x0+tile_size]
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    tile = cv2.copyMakeBorder(tile, 0, tile_size-tile.shape[0], 0, tile_size-tile.shape[1], cv2.BORDER_CONSTANT, value=0)
                tiles_to_infer.append(tile)
                tile_metadata.append((x0, y0, 1.0, 1.0))

        # --- ÉTAPE 2 : INFÉRENCE BATCHÉE ---
        if self.backend == 'tensorrt_native':
            batch_boxes, batch_scores = self._trt_native_infer_batch(tiles_to_infer)
            for i, (boxes, scores) in enumerate(zip(batch_boxes, batch_scores)):
                if boxes.size > 0:
                    x0, y0, sw, sh = tile_metadata[i]
                    # On rescale les boxes si nécessaire (pour les quadrants)
                    if sw != 1.0 or sh != 1.0:
                        boxes[:, [0, 2]] *= sw
                        boxes[:, [1, 3]] *= sh
                    # On ajoute l'offset
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes_list.append(boxes)
                    all_scores_list.append(scores)
        else:
            # Fallback séquentiel pour OpenVINO/Torch (non encore batché ici)
            for i, tile in enumerate(tiles_to_infer):
                if self.backend == 'openvino_native':
                    boxes, scores = self._ov_native_infer(tile)
                else:
                    results = self.model.predict(tile, verbose=False, imgsz=tile_size)
                    mask = (results[0].boxes.cls.cpu().numpy() == self.person_class_id) & (results[0].boxes.conf.cpu().numpy() >= self.confidence_threshold)
                    boxes = results[0].boxes.xyxy[mask].cpu().numpy()
                    scores = results[0].boxes.conf[mask].cpu().numpy()
                
                if boxes.size > 0:
                    x0, y0, sw, sh = tile_metadata[i]
                    if self.backend == 'openvino_native' and (sw != 1.0 or sh != 1.0):
                        boxes[:, [0, 2]] *= sw
                        boxes[:, [1, 3]] *= sh
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes_list.append(boxes)
                    all_scores_list.append(scores)

        if not all_boxes_list:
            return (0, image.copy()) if draw_boxes else 0

        # Concaténation et NMS Hiérarchique
        all_boxes = np.concatenate(all_boxes_list, axis=0)
        all_scores = np.concatenate(all_scores_list, axis=0)
        keep = self._nms(all_boxes, all_scores, iou_threshold=0.25, inclusion_threshold=0.45)
        
        total_count = len(keep)
        image_out = image.copy() if draw_boxes else None
        if draw_boxes:
            for i in keep:
                x1, y1, x2, y2 = all_boxes[i].astype(int)
                cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if draw_boxes:
            return int(total_count), image_out
        return int(total_count)
        
        return (int(total_count), image_out) if draw_boxes else int(total_count)

    def _ov_native_infer(self, tile_rgb):
        img = tile_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        if self.ov_input_shape and len(self.ov_input_shape) == 4:
            _, _, target_h, target_w = self.ov_input_shape
            if img.shape[2] != target_h or img.shape[3] != target_w:
                resized = cv2.resize(tile_rgb, (target_w, target_h))
                img = resized.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[None, ...]
        result = self.ov_compiled_model([img])[self.ov_output]
        pred = result[0].T
        boxes = pred[:, :4]
        scores = pred[:, 4 + self.person_class_id]
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        if boxes.size == 0:
            return boxes, scores
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        return boxes, scores

    def _trt_native_infer(self, tile_rgb):
        import numpy as np
        h_orig, w_orig = tile_rgb.shape[:2]

        img = tile_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        
        # Check input shape
        target_w, target_h = w_orig, h_orig
        if self.trt_input_shape and len(self.trt_input_shape) == 4:
            _, _, target_h, target_w = self.trt_input_shape
            if img.shape[2] != target_h or img.shape[3] != target_w:
                resized = cv2.resize(tile_rgb, (target_w, target_h))
                img = resized.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[None, ...]

        # Copy input to device
        input_data = np.ascontiguousarray(img)
        # Use the actual size of input_data instead of self.trt_inputs[0]['nbytes'] (which might be max batch)
        self.cudart.cudaMemcpyAsync(self.trt_inputs[0]['device'], input_data.ctypes.data, input_data.nbytes, self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream)

        # IMPORTANT: Set input shape for dynamic engines
        self.trt_context.set_input_shape(self.trt_inputs[0]['name'], input_data.shape)

        # Execute
        self.trt_context.execute_async_v3(self.trt_stream)
        self.cudart.cudaStreamSynchronize(self.trt_stream)

        # Copy output back
        for output in self.trt_outputs:
            self.cudart.cudaMemcpyAsync(output['host'].ctypes.data, output['device'], output['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream)
        
        self.cudart.cudaStreamSynchronize(self.trt_stream)

        # Process output (assuming YOLO structure: [1, 84, 8400])
        result = self.trt_outputs[0]['host']
        pred = result[0].T
        boxes = pred[:, :4]
        scores = pred[:, 4 + self.person_class_id]
        
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        if boxes.size == 0:
            return boxes, scores

        # Scale back to original tile size
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
        return boxes, scores

    def _trt_native_infer_batch(self, batch_imgs):
        """
        Inférence YOLO en Batch pour TensorRT.
        batch_imgs: Liste d'images (tiles)
        """
        num_imgs = len(batch_imgs)
        if num_imgs == 0: return [], []
        
        # On suppose que toutes les images du batch font la même taille (imgsz de YOLO)
        target_h, target_w = self.trt_input_shape[2], self.trt_input_shape[3]
        max_batch = self.trt_input_shape[0] if self.trt_input_shape[0] > 0 else num_imgs
        
        all_boxes, all_scores = [], []
        
        # On traite par morceaux de taille supportée par l'engine
        from concurrent.futures import ThreadPoolExecutor
        
        # On définit une fonction pour le pré-processing d'une tuile
        def process_tile(tile, target_h, target_w):
            if tile.shape[0] != target_h or tile.shape[1] != target_w:
                tile = cv2.resize(tile, (target_w, target_h))
            tile = tile.astype(np.float32)
            tile *= (1.0 / 255.0)
            return np.transpose(tile, (2, 0, 1))

        for i in range(0, num_imgs, max_batch):
            chunk = batch_imgs[i:i+max_batch]
            actual_b = len(chunk)
            
            # Preprocess Batch (optimisé en parallèle pour le CPU)
            processed = np.zeros((actual_b, 3, target_h, target_w), dtype=np.float32)
            
            with ThreadPoolExecutor(max_workers=min(8, actual_b)) as executor:
                futures = [executor.submit(process_tile, chunk[j], target_h, target_w) for j in range(actual_b)]
                for j, future in enumerate(futures):
                    processed[j] = future.result()
            
            input_data = np.ascontiguousarray(processed)
            
            # On vérifie si on peut vraiment envoyer le batch
            if self.trt_input_shape[0] > 0 and actual_b > self.trt_input_shape[0]:
                print(f"[WARN] Batch size {actual_b} exceeds engine max {self.trt_input_shape[0]}. Falling back to sequential.")
                for j in range(actual_b):
                    single_boxes, single_scores = self._trt_native_infer(chunk[j])
                    all_boxes.append(single_boxes)
                    all_scores.append(single_scores)
            else:
                # Real batch inference
                # print(f"[DEBUG] YOLO: Executing batch of size {actual_b}")
                self.cudart.cudaMemcpyAsync(self.trt_inputs[0]['device'], input_data.ctypes.data, input_data.nbytes, self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream)
                # Définir explicitement la forme du batch actuel
                self.trt_context.set_input_shape(self.trt_inputs[0]['name'], input_data.shape)
                self.trt_context.execute_async_v3(self.trt_stream)
                self.cudart.cudaStreamSynchronize(self.trt_stream)

                for output in self.trt_outputs:
                    self.cudart.cudaMemcpyAsync(output['host'].ctypes.data, output['device'], output['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream)
                self.cudart.cudaStreamSynchronize(self.trt_stream)

                # Process Output
                result = self.trt_outputs[0]['host'] # [Batch, 84, 8400]
                for j in range(actual_b):
                    pred = result[j].T
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

        return all_boxes, all_scores

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.3, inclusion_threshold=0.5):
        """
        NMS Hiérarchique optimisé pour le Multi-Scale.
        Utilise une pondération par la surface pour privilégier les détections 
        'globales' (le tout) sur les détections de 'tuiles' (les parties), 
        tout en préservant les petits objets isolés (chevaliers au fond).
        """
        if boxes.size == 0:
            return []
            
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # HEURISTIQUE CLÉ : On trie par (Score * sqrt(Area))
        # Cela permet à une détection globale du corps (grande surface) de 'gagner' 
        # sur une détection locale d'une partie du corps (petite surface)
        # même si le score local est légèrement meilleur.
        importance = scores * np.sqrt(areas)
        order = importance.argsort()[::-1]
        
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            # IoU classique
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)
            
            # Ratio d'Inclusion (inter / min_area)
            min_area = np.minimum(areas[i], areas[order[1:]])
            io_min = inter / (min_area + 1e-6)
            
            # Suppression si l'un des deux critères est rempli
            mask = (iou > iou_threshold) | (io_min > inclusion_threshold)
            
            inds = np.where(~mask)[0]
            order = order[inds + 1]
            
        return keep

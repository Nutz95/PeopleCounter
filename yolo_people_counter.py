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
                shape = self.trt_engine.get_tensor_shape(name)
                dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
                size = trt.volume(shape)
                nbytes = size * np.dtype(dtype).itemsize
                
                # Allocate device memory
                err, device_ptr = cudart.cudaMalloc(nbytes)
                if int(err) != 0:
                    raise RuntimeError(f"CUDA Malloc failed: {err}")
                
                # Allocation host memory
                host_mem = np.empty(shape, dtype=dtype)
                
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

    def count_people(self, image, tile_size=640, draw_boxes=False, use_tiling=False):
        if not use_tiling:
            # Inference sur image entière redimensionnée (beaucoup plus rapide)
            if self.backend == 'openvino_native':
                boxes, scores = self._ov_native_infer(image)
                keep = self._nms(boxes, scores, iou_threshold=0.45)
                total_count = len(keep)
                image_out = image.copy() if draw_boxes else None
                if draw_boxes:
                    for i in keep:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif self.backend == 'tensorrt_native':
                boxes, scores = self._trt_native_infer(image)
                keep = self._nms(boxes, scores, iou_threshold=0.45)
                total_count = len(keep)
                image_out = image.copy() if draw_boxes else None
                if draw_boxes:
                    for i in keep:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                results = self.model.predict(image, verbose=False, imgsz=tile_size)
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
        h, w = image.shape[:2]
        all_boxes = []
        all_scores = []

        # --- ÉTAPE 1 : PASSE GLOBALE (Cibles Proches > 350px) ---
        if self.backend == 'openvino_native':
            g_boxes, g_scores = self._ov_native_infer(image)
        elif self.backend == 'tensorrt_native':
            g_boxes, g_scores = self._trt_native_infer(image)
        else:
            g_results = self.model.predict(image, verbose=False, imgsz=tile_size)
            g_boxes = g_results[0].boxes.xyxy.cpu().numpy()
            g_scores = g_results[0].boxes.conf.cpu().numpy()
            g_mask = (g_results[0].boxes.cls.cpu().numpy() == self.person_class_id) & (g_scores >= self.confidence_threshold)
            g_boxes, g_scores = g_boxes[g_mask], g_scores[g_mask]
        
        if len(g_boxes) > 0:
            all_boxes.append(g_boxes)
            all_scores.append(g_scores)

        # --- ÉTAPE 2 : PASSE QUADRANTS (Cibles Moyennes) ---
        quad_h, quad_w = h // 2, w // 2
        for i in range(2):
            for j in range(2):
                y0, x0 = i * quad_h, j * quad_w
                quad = image[y0:y0+quad_h, x0:x0+quad_w]
                quad_rgb = quad[..., ::-1]
                
                if self.backend == 'openvino_native':
                    q_boxes, q_scores = self._ov_native_infer(cv2.resize(quad_rgb, (tile_size, tile_size)))
                elif self.backend == 'tensorrt_native':
                    q_boxes, q_scores = self._trt_native_infer(cv2.resize(quad_rgb, (tile_size, tile_size)))
                else:
                    results = self.model.predict(quad_rgb, verbose=False, imgsz=tile_size)
                    q_boxes = results[0].boxes.xyxy.cpu().numpy()
                    q_scores = results[0].boxes.conf.cpu().numpy()
                    q_mask = (results[0].boxes.cls.cpu().numpy() == self.person_class_id) & (q_scores >= self.confidence_threshold)
                    q_boxes, q_scores = q_boxes[q_mask], q_scores[q_mask]

                if len(q_boxes) > 0:
                    if self.backend in ['openvino_native', 'tensorrt_native']:
                        q_boxes[:, [0, 2]] *= (quad_w / tile_size)
                        q_boxes[:, [1, 3]] *= (quad_h / tile_size)
                    q_boxes[:, [0, 2]] += x0
                    q_boxes[:, [1, 3]] += y0
                    all_boxes.append(q_boxes)
                    all_scores.append(q_scores)

        # --- ÉTAPE 3 : PASSE PAR TUILES (Cibles Éloignées / Petites) ---
        overlap_ratio = 0.2
        step = int(tile_size * (1 - overlap_ratio))
        y_coords = list(range(0, max(1, h - tile_size + step), step))
        if y_coords[-1] + tile_size < h: y_coords.append(h - tile_size)
        x_coords = list(range(0, max(1, w - tile_size + step), step))
        if x_coords[-1] + tile_size < w: x_coords.append(w - tile_size)

        for y0 in y_coords:
            for x0 in x_coords:
                tile = image[y0:y0+tile_size, x0:x0+tile_size]
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    tile = cv2.copyMakeBorder(tile, 0, tile_size-tile.shape[0], 0, tile_size-tile.shape[1], cv2.BORDER_CONSTANT, value=0)
                
                tile_rgb = tile[..., ::-1]
                if self.backend == 'openvino_native':
                    boxes, scores = self._ov_native_infer(tile_rgb)
                elif self.backend == 'tensorrt_native':
                    boxes, scores = self._trt_native_infer(tile_rgb)
                else:
                    results = self.model.predict(tile_rgb, verbose=False, imgsz=tile_size)
                    r = results[0]
                    boxes, scores = [], []
                    if hasattr(r, 'boxes'):
                        mask = (r.boxes.cls.cpu().numpy() == self.person_class_id) & (r.boxes.conf.cpu().numpy() >= self.confidence_threshold)
                        boxes = r.boxes.xyxy[mask].cpu().numpy()
                        scores = r.boxes.conf[mask].cpu().numpy()

                if len(boxes) > 0:
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes.append(boxes)
                    all_scores.append(scores)

        if not all_boxes:
            return (0, image.copy()) if draw_boxes else 0

        # Concaténation et NMS Hiérarchique
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        keep = self._nms(all_boxes, all_scores, iou_threshold=0.35, inclusion_threshold=0.6)
        
        total_count = len(keep)
        image_out = image.copy() if draw_boxes else None
        if draw_boxes:
            for i in keep:
                x1, y1, x2, y2 = all_boxes[i].astype(int)
                cv2.rectangle(image_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
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

        img = tile_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        
        # Check input shape
        if self.trt_input_shape and len(self.trt_input_shape) == 4:
            _, _, target_h, target_w = self.trt_input_shape
            if img.shape[2] != target_h or img.shape[3] != target_w:
                resized = cv2.resize(tile_rgb, (target_w, target_h))
                img = resized.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[None, ...]

        # Copy input to device
        input_data = np.ascontiguousarray(img)
        self.cudart.cudaMemcpyAsync(self.trt_inputs[0]['device'], input_data.ctypes.data, self.trt_inputs[0]['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream)

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

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.3, inclusion_threshold=0.5):
        """
        NMS Robuste optimisé pour la détection multi-échelle (Global, Quads, Tiles).
        Gère les chevauchements entre les différents niveaux de la pyramide.
        """
        if boxes.size == 0:
            return []
            
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        order = scores.argsort()[::-1]
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
            
            # IoU classique pour les doublons à une même échelle
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)
            
            # Ratio d'Inclusion : Supprime les boîtes (ex: tête) incluses dans une plus grande (ex: corps)
            min_area = np.minimum(areas[i], areas[order[1:]])
            io_min = inter / (min_area + 1e-6)
            
            # Suppression si l'IoU est élevé OU si une boîte est presque totalement incluse dans l'autre
            mask = (iou > iou_threshold) | (io_min > inclusion_threshold)
            
            inds = np.where(~mask)[0]
            order = order[inds + 1]
            
        return keep

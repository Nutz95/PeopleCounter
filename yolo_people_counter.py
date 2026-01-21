from ultralytics import YOLO
import numpy as np
import cv2
import os

class YoloPeopleCounter:
    def __init__(self, model_name='yolov26n', device='cpu', confidence_threshold=0.25, backend='torch'):
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

        if self.backend == 'openvino_native':
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

    def count_people(self, image, tile_size=640, draw_boxes=False):
        h, w = image.shape[:2]
        tiles = []
        tile_coords = []
        pads = []
        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                tile = image[y0:y0+tile_size, x0:x0+tile_size]
                pad_bottom = tile_size - tile.shape[0]
                pad_right = tile_size - tile.shape[1]
                if pad_bottom > 0 or pad_right > 0:
                    tile = cv2.copyMakeBorder(tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
                tile_rgb = tile[..., ::-1]
                tiles.append(tile_rgb)
                tile_coords.append((x0, y0))
                pads.append((pad_right, pad_bottom))
        total_count = 0
        image_out = image.copy() if draw_boxes else None
        if self.backend == 'openvino_native':
            for idx, tile in enumerate(tiles):
                x0, y0 = tile_coords[idx]
                pad_right, pad_bottom = pads[idx]
                boxes, scores = self._ov_native_infer(tile)
                keep = self._nms(boxes, scores, iou_threshold=0.45)
                total_count += len(keep)
                if draw_boxes:
                    for i in keep:
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        if x2 <= tile_size - pad_right and y2 <= tile_size - pad_bottom:
                            x1 = min(x1, tile_size - pad_right - 1)
                            x2 = min(x2, tile_size - pad_right - 1)
                            y1 = min(y1, tile_size - pad_bottom - 1)
                            y2 = min(y2, tile_size - pad_bottom - 1)
                            x1 += x0
                            x2 += x0
                            y1 += y0
                            y2 += y0
                            cv2.rectangle(image_out, (x1, y1), (x2, y2), (0,255,0), 2)
        else:
            # Traitement batché sur GPU
            if self.backend == 'openvino':
                results = []
                for tile in tiles:
                    results.extend(self.model.predict([tile], verbose=False))
            else:
                results = self.model.predict(tiles, verbose=False)
            for idx, r in enumerate(results):
                x0, y0 = tile_coords[idx]
                pad_right, pad_bottom = pads[idx]
                if hasattr(r, 'boxes'):
                    boxes = r.boxes
                    if hasattr(boxes, 'cls') and hasattr(boxes, 'conf'):
                        person_mask = (boxes.cls.cpu().numpy() == self.person_class_id) & (boxes.conf.cpu().numpy() >= self.confidence_threshold)
                        total_count += np.sum(person_mask)
                        if draw_boxes and hasattr(boxes, 'xyxy'):
                            for i, is_person in enumerate(person_mask):
                                if is_person:
                                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                                    x1, y1, x2, y2 = box
                                    if x2 <= tile_size - pad_right and y2 <= tile_size - pad_bottom:
                                        x1 = min(x1, tile_size - pad_right - 1)
                                        x2 = min(x2, tile_size - pad_right - 1)
                                        y1 = min(y1, tile_size - pad_bottom - 1)
                                        y2 = min(y2, tile_size - pad_bottom - 1)
                                        x1 += x0
                                        x2 += x0
                                        y1 += y0
                                        y2 += y0
                                        cv2.rectangle(image_out, (x1, y1), (x2, y2), (0,255,0), 2)
        if draw_boxes:
            return int(total_count), image_out
        else:
            return int(total_count)

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

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.45):
        if boxes.size == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
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
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

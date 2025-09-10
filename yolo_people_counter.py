from ultralytics import YOLO
import numpy as np
import cv2

class YoloPeopleCounter:
    def __init__(self, model_name='yolov8n', device='cpu', confidence_threshold=0.25):
        """
        model_name: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'yolo11n', etc.
        device: 'cpu' or 'cuda'
        confidence_threshold: float, minimum confidence for detection
        """
        self.model_name = model_name
        self.model = YOLO(model_name, task='detect')
        self.model.to(device)
        self.device = device
        self.person_class_id = 0  # COCO: 0 = person
        self.confidence_threshold = confidence_threshold

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
        # Traitement batché sur GPU
        results = self.model.predict(tiles, verbose=False)
        total_count = 0
        image_out = image.copy() if draw_boxes else None
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

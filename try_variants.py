from ultralytics import YOLO
import sys

variants = ["yolo12n-seg.pt", "yolov12n-seg.pt", "yolo12n_seg.pt", "yolo12s-seg.pt", "yolov12s-seg.pt", "yolo12s_seg.pt"]
for v in variants:
    print(f"Trying {v}...")
    try:
        model = YOLO(v)
        print(f"SUCCESS: {v} is available")
        break
    except Exception as e:
        print(f"FAILED {v}: {e}")

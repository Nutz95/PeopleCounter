from ultralytics import YOLO
import logging

# Disable logging to avoid clutter
logging.getLogger("ultralytics").setLevel(logging.ERROR)

models_to_try = [
    "yolo12n-seg.pt",
    "yolo12s-seg.pt",
    "yolo12m-seg.pt",
    "yolo12l-seg.pt",
    "yolo12x-seg.pt",
    "yolo12n-segment.pt",
    "yolo12s-segment.pt",
    "yolo12n_seg.pt",
    "yolo12s_seg.pt",
]

for model_name in models_to_try:
    print(f"Trying {model_name}...")
    try:
        model = YOLO(model_name)
        print(f"  Result: Found! {model_name}")
    except Exception as e:
        print(f"  Result: Not found. Error: {str(e)[:100]}...")

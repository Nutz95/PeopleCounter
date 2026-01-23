from ultralytics import YOLO
import sys

models = ["yolo12n-seg.pt", "yolo12s-seg.pt", "yolo12m-seg.pt"]
for m in models:
    print(f"Checking {m}...")
    try:
        model = YOLO(m)
        print(f"Successfully loaded {m}")
    except Exception as e:
        print(f"Failed to load {m}: {e}")

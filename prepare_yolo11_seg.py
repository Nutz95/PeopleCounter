from ultralytics import YOLO
import torch
import os
import sys
from convert_onnx_to_trt import build_engine

def main():
    # Gamme complète des modèles segmentations
    models = ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"]

    for m_name in models:
        print(f"\n--- Processing {m_name} ---")
        # Ensure CUDA is visible to torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available in current process.")
        
        # Download
        model = YOLO(m_name)
        
        onnx_name = m_name.replace(".pt", ".onnx")
        engine_name = m_name.replace(".pt", ".engine")
        
        if not os.path.exists(onnx_name):
            print(f"Exporting {m_name} to ONNX...")
            model.export(format="onnx", dynamic=True, opset=12, device='cpu')
        
        if os.path.exists(engine_name):
            print(f"Skipping: {engine_name} already exists.")
            continue

        if os.path.exists(onnx_name):
            print(f"Converting {onnx_name} to TensorRT...")
            success = build_engine(onnx_name, engine_name, max_batch_size=32)
            if success:
                print(f"SUCCESS: {engine_name} created.")
            else:
                print(f"FAILED: {engine_name} conversion.")
        else:
            print(f"Error: {onnx_name} not found after export.")

if __name__ == "__main__":
    main()

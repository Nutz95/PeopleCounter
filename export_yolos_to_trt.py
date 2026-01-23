import os
import glob
import subprocess
import sys

def export_yolos():
    # Find all .pt files in current directory
    pt_files = glob.glob("*.pt")
    print(f"Found YOLO models: {pt_files}")
    
    python_path = sys.executable

    for pt in pt_files:
        print(f"\n--- Exporting {pt} to ONNX (CPU) ---")
        try:
            onnx_path = pt.replace(".pt", ".onnx")
            engine_path = pt.replace(".pt", ".engine")

            # Export ONNX via a separate process to keep CUDA state clean
            export_cmd = [
                python_path, "-c",
                f"from ultralytics import YOLO; model = YOLO('{pt}'); model.export(format='onnx', dynamic=True, opset=12, device='cpu')"
            ]
            subprocess.run(export_cmd)
            
            # Now convert ONNX to TRT using the custom script
            # We use a batch size of 32 instead of 64 for robustness with large models
            print(f"Converting {onnx_path} to {engine_path} (Batch=32)...")
            subprocess.run([python_path, "convert_onnx_to_trt.py", onnx_path, engine_path, "32"])
            
            print(f"Process for {pt} COMPLETED.")
        except Exception as e:
            print(f"FAILED to process {pt}: {e}")

if __name__ == "__main__":
    export_yolos()

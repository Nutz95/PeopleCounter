import os
import glob
import subprocess
import sys

def export_yolos(models_dir="models"):
    # Find all .pt files under models_dir
    os.makedirs(models_dir, exist_ok=True)
    pt_files = glob.glob(os.path.join(models_dir, "*.pt"))
    print(f"Found YOLO models in {models_dir}: {pt_files}")
    
    python_path = sys.executable

    for pt_path in pt_files:
        pt = os.path.basename(pt_path)
        print(f"\n--- Exporting {pt} to ONNX (CPU) ---")
        try:
            onnx_path = os.path.join(models_dir, pt.replace(".pt", ".onnx"))
            engine_path = os.path.join(models_dir, pt.replace(".pt", ".engine"))

            # Export ONNX via a separate process to keep CUDA state clean
            export_cmd = [
                python_path, "-c",
                (
                    "from ultralytics import YOLO; "
                    f"model = YOLO('{pt}'); model.export(format='onnx', dynamic=True, opset=12, device='cpu'); "
                    f"print('exported {pt} to onnx')"
                )
            ]
            # Run export with working directory = models_dir so outputs land there
            subprocess.run(export_cmd, cwd=models_dir)
            
            # Now convert ONNX to TRT using the custom script
            print(f"Converting {onnx_path} to {engine_path} (Batch=32)...")
            subprocess.run([python_path, os.path.join(sys.path[0], "convert_onnx_to_trt.py"), onnx_path, engine_path, "32"])
            
            print(f"Process for {pt} COMPLETED.")
        except Exception as e:
            print(f"FAILED to process {pt}: {e}")

if __name__ == "__main__":
    export_yolos(models_dir=os.environ.get('MODELS_DIR', 'models'))

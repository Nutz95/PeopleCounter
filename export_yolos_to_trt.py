import glob
import os
import shutil
import subprocess
import sys

def export_yolos(models_dir="models"):
    # Find all .pt files under models_dir and subfolders
    os.makedirs(models_dir, exist_ok=True)
    pt_files = glob.glob(os.path.join(models_dir, "*.pt")) + glob.glob(os.path.join(models_dir, "**", "*.pt"))
    print(f"Found YOLO models: {pt_files}")
    
    python_path = sys.executable

    for pt_path in pt_files:
        pt = os.path.basename(pt_path)
        print(f"\n--- Exporting {pt} to ONNX (CPU) ---")
        try:
            # Place outputs in the correct structured folders
            onnx_dir = os.path.join(models_dir, "onnx")
            engine_dir = os.path.join(models_dir, "tensorrt")
            os.makedirs(onnx_dir, exist_ok=True)
            os.makedirs(engine_dir, exist_ok=True)
            
            onnx_path = os.path.join(onnx_dir, pt.replace(".pt", ".onnx"))
            engine_path = os.path.join(engine_dir, pt.replace(".pt", ".engine"))

            generated_onnx = os.path.join(models_dir, pt.replace(".pt", ".onnx"))

            if os.path.exists(engine_path):
                print(f"{engine_path} already exists, skipping export/conversion")
                continue

            # Export ONNX via a separate process to keep CUDA state clean
            export_cmd = [
                python_path, "-c",
                (
                    "from ultralytics import YOLO; "
                    f"model = YOLO('{pt_path}'); model.export(format='onnx', dynamic=True, opset=12, device='cpu'); "
                    f"print('exported {pt} to onnx')"
                )
            ]
            # Run export with working directory = models_dir so outputs land there
            need_export = not os.path.exists(onnx_path)
            if need_export:
                subprocess.run(export_cmd, cwd=models_dir)
                if os.path.exists(generated_onnx):
                    shutil.move(generated_onnx, onnx_path)
            else:
                print(f"{onnx_path} already exists, skipping export step")
            
            # Now convert ONNX to TRT using the custom script
            print(f"Converting {onnx_path} to {engine_path} (Batch=32)...")
            subprocess.run([python_path, os.path.join(sys.path[0], "convert_onnx_to_trt.py"), onnx_path, engine_path, "32"])
            
            print(f"Process for {pt} COMPLETED.")
        except Exception as e:
            print(f"FAILED to process {pt}: {e}")

if __name__ == "__main__":
    export_yolos(models_dir=os.environ.get('MODELS_DIR', 'models'))

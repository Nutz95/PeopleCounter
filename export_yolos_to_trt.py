import glob
import os
import shutil
import subprocess
import sys

def _ensure_dir(path, mode=0o775):
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, mode)
    except PermissionError:
        pass
    return path


def export_yolos(models_dir="models"):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(script_dir, models_dir)
    models_dir = os.path.abspath(models_dir)
    _ensure_dir(models_dir)
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
            _ensure_dir(onnx_dir)
            _ensure_dir(engine_dir)
            
            onnx_path = os.path.join(onnx_dir, pt.replace(".pt", ".onnx"))
            engine_path = os.path.join(engine_dir, pt.replace(".pt", ".engine"))

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
            # Run export from repo root to avoid Ultralytics creating nested models/
            need_export = not os.path.exists(onnx_path)
            if need_export:
                subprocess.run(export_cmd, cwd=script_dir)
                generated_onnx = _locate_generated_asset(models_dir, pt, ".onnx")
                if generated_onnx:
                    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                    shutil.move(generated_onnx, onnx_path)
            else:
                print(f"{onnx_path} already exists, skipping export step")
            
            # Now convert ONNX to TRT using the custom script
            print(f"Converting {onnx_path} to {engine_path} (Batch=32)...")
            subprocess.run([python_path, os.path.join(sys.path[0], "convert_onnx_to_trt.py"), onnx_path, engine_path, "32"])
            generated_engine = _locate_generated_asset(models_dir, pt, ".engine")
            if generated_engine:
                os.makedirs(os.path.dirname(engine_path), exist_ok=True)
                shutil.move(generated_engine, engine_path)
            
            print(f"Process for {pt} COMPLETED.")
        except Exception as e:
            print(f"FAILED to process {pt}: {e}")

    _clean_redundant_dir(models_dir)

def _locate_generated_asset(models_dir, pt_filename, suffix):
    name = pt_filename.replace(".pt", suffix)
    pattern = os.path.join(models_dir, "**", name)
    subdir_map = {
        '.onnx': 'onnx',
        '.engine': 'tensorrt',
    }
    target_subdir = os.path.join(models_dir, subdir_map.get(suffix, '')) if suffix in subdir_map else None
    for candidate in sorted(glob.glob(pattern, recursive=True)):
        candidate_abs = os.path.abspath(candidate)
        if target_subdir and candidate_abs.startswith(os.path.abspath(target_subdir)):
            continue
        if os.path.isdir(candidate_abs):
            continue
        return candidate_abs
    return None

def _clean_redundant_dir(models_dir):
    redundant_models_dir = os.path.join(models_dir, "models")
    if os.path.isdir(redundant_models_dir):
        try:
            shutil.rmtree(redundant_models_dir)
            print(f"Removed redundant directory {redundant_models_dir}")
        except Exception as cleanup_exc:
            print(f"Warning: unable to remove {redundant_models_dir}: {cleanup_exc}")

    for sub in ("onnx", "tensorrt"):
        nested = os.path.join(models_dir, sub, "models")
        if os.path.isdir(nested):
            try:
                shutil.rmtree(nested)
                print(f"Removed nested directory {nested}")
            except Exception as cleanup_exc:
                print(f"Warning: unable to remove {nested}: {cleanup_exc}")

if __name__ == "__main__":
    export_yolos(models_dir=os.environ.get('MODELS_DIR', 'models'))

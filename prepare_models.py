import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENDORS = ROOT / 'vendors'
LWCC_DIR = VENDORS / 'lwcc'

# Clone and install lwcc from git if LWCC_GIT_URL is set and lwcc missing
LWCC_GIT_URL = os.environ.get('LWCC_GIT_URL') or 'https://github.com/Nutz95/lwcc.git'

def run(cmd, **kwargs):
    print(f"[prepare_models] $ {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)

if not VENDORS.exists():
    VENDORS.mkdir(parents=True, exist_ok=True)

if LWCC_GIT_URL and not LWCC_DIR.exists():
    print(f"[prepare_models] Cloning lwcc from {LWCC_GIT_URL} into {LWCC_DIR}")
    try:
        run(f"git clone {LWCC_GIT_URL} {LWCC_DIR}")
    except Exception as e:
        print(f"[prepare_models] git clone failed: {e}. Falling back to ZIP download.")
        try:
            # Download the main branch zip archive and extract; try 'main' then 'master'
            import urllib.request
            import zipfile
            from io import BytesIO
            import shutil

            tried = []
            for branch in ('main', 'master'):
                zip_url = LWCC_GIT_URL.rstrip('.git') + f'/archive/refs/heads/{branch}.zip'
                tried.append(zip_url)
                try:
                    print(f"[prepare_models] Attempting to download archive {zip_url}")
                    data = urllib.request.urlopen(zip_url, timeout=30).read()
                    z = zipfile.ZipFile(BytesIO(data))
                    # Extract to a temp dir then move
                    temp_dir = VENDORS / 'lwcc_tmp'
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    z.extractall(path=str(temp_dir))
                    # The archive usually contains a single top-level folder
                    entries = list(temp_dir.iterdir())
                    if entries:
                        top = entries[0]
                        top.rename(LWCC_DIR)
                        print(f"[prepare_models] Extracted lwcc to {LWCC_DIR}")
                        break
                except Exception as ex_branch:
                    print(f"[prepare_models] Archive download for {branch} failed: {ex_branch}")
            else:
                print(f"[prepare_models] All archive attempts failed: {tried}")
        except Exception as ex:
            print(f"[prepare_models] Failed to download/extract lwcc zip: {ex}")

# If vendors/lwcc exists, install editable local package
if LWCC_DIR.exists():
    print(f"[prepare_models] Installing local lwcc (editable)")
    run(f"pip install --no-deps -e {LWCC_DIR}")
else:
    print("[prepare_models] vendors/lwcc not present; skipping local install. Set LWCC_GIT_URL to auto-clone.")

# Ensure models directory exists
models_dir = ROOT / 'models'
models_dir.mkdir(exist_ok=True)

# Optionally, user can list YOLO models to pre-download via YOLO_PREPARE env var (comma separated)
yolo_list = os.environ.get('YOLO_PREPARE', '')
if yolo_list:
    try:
        from ultralytics import YOLO
        for name in [x.strip() for x in yolo_list.split(',') if x.strip()]:
            print(f"[prepare_models] Ensuring YOLO model available: {name}")
            # This will download model weights if needed
            try:
                model = YOLO(name)
                print(f"[prepare_models] YOLO {name} ready")
            except Exception as e:
                print(f"[prepare_models] Warning: failed to prepare {name}: {e}")
    except Exception as e:
        print(f"[prepare_models] ultralytics not available: {e}")

print("[prepare_models] Done.")

# Additional automatic prep steps (weights + density exports + OpenVINO/TRT conversions)
try:
    # Download LWCC weights into ~/.lwcc/weights
    print("[prepare_models] Running download_lwcc_weights.py")
    run(f"python3 download_lwcc_weights.py")
except Exception as e:
    print(f"[prepare_models] Warning: download_lwcc_weights.py failed: {e}")
    
# Ensure weights are in /root/.lwcc/weights as some scripts expect that path
try:
    import shutil
    src = Path('/') / '.lwcc' / 'weights'
    dst = Path('/root') / '.lwcc' / 'weights'
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[prepare_models] Moving weights from {src} to {dst}")
        shutil.move(str(src), str(dst))
    elif src.exists() and dst.exists():
        # Move individual files
        for f in src.glob('*'):
            shutil.move(str(f), str(dst))
        try:
            src.rmdir()
        except Exception:
            pass
except Exception as e:
    print(f"[prepare_models] Warning: failed to relocate weights: {e}")

try:
    # Export density models to ONNX
    print("[prepare_models] Running export_density_to_onnx.py")
    run(f"python3 export_density_to_onnx.py")
except Exception as e:
    print(f"[prepare_models] Warning: export_density_to_onnx.py failed: {e}")

# Convert any generated LWCC ONNX models to TensorRT if CUDA available
try:
    onnx_dir = ROOT / 'models' / 'onnx'
    trt_dir = ROOT / 'models' / 'tensorrt'
    onnx_dir.mkdir(parents=True, exist_ok=True)
    trt_dir.mkdir(parents=True, exist_ok=True)
    has_cuda = False
    try:
        import shutil, subprocess
        if shutil.which('nvidia-smi'):
            res = subprocess.run(['nvidia-smi','-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            has_cuda = (res.returncode == 0)
    except Exception:
        has_cuda = False

    if has_cuda:
        for onnx in onnx_dir.glob('*.onnx'):
            engine = trt_dir / (onnx.stem + '.engine')
            if not engine.exists():
                print(f"[prepare_models] Converting {onnx} -> {engine}")
                try:
                    run(f"python3 convert_onnx_to_trt.py {onnx} {engine} 32")
                except Exception as ce:
                    print(f"[prepare_models] Warning: convert_onnx_to_trt failed for {onnx}: {ce}")
    else:
        print("[prepare_models] CUDA not available in container; skipping ONNX->TensorRT conversion.")
except Exception as e:
    print(f"[prepare_models] Warning: ONNX->TensorRT conversion step failed: {e}")

try:
    # Convert PTH to OpenVINO IR
    print("[prepare_models] Running convert_pth_to_openvino.py")
    run(f"python3 convert_pth_to_openvino.py")
except Exception as e:
    print(f"[prepare_models] Warning: convert_pth_to_openvino.py failed: {e}")

try:
    # Prepare YOLO models (ONNX/OpenVINO/TensorRT)
    print("[prepare_models] Running prepare_yolo_all.py")
    run(f"python3 prepare_yolo_all.py")
except Exception as e:
    print(f"[prepare_models] Warning: prepare_yolo_all.py failed: {e}")

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


def _ensure_directory(path, mode=0o775):
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    try:
        resolved.chmod(mode)
    except PermissionError:
        pass
    return resolved

_ensure_directory(VENDORS)

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
    # Add to sys.path so we can import it immediately in this process
    sys.path.append(str(LWCC_DIR))
    # Still install it for other processes
    run(f"pip install --no-deps -e {LWCC_DIR}")

# Ensure models directory exists
models_dir = ROOT / 'models'
_ensure_directory(models_dir)

# Point LWCC to use a persistent weights directory inside the workspace
# This prevents re-downloading inside the container
LWCC_WEIGHTS_DIR = models_dir / 'lwcc_weights'
_ensure_directory(LWCC_WEIGHTS_DIR)
os.environ['LWCC_WEIGHTS_PATH'] = str(LWCC_WEIGHTS_DIR)

# Also try to set for Python's current process
try:
    import lwcc.util.functions as lwcc_funcs
    # some versions of lwcc might use a global or need a patch
    lwcc_funcs.PAIRS = lwcc_funcs.PAIRS # just ensuring it's imported
except Exception as e:
    print(f"[prepare_models] Optional LWCC setup skipped or failed: {e}")
    pass

# Optionally, user can list YOLO models to pre-download via YOLO_PREPARE env var (comma separated)
# Defaults to yolo26 series if not specified
yolo_list = os.environ.get('YOLO_PREPARE', 'yolo26n.pt,yolo26s.pt,yolo26m.pt,yolo26l.pt,yolo26x.pt,yolo26n-seg.pt,yolo26s-seg.pt,yolo26m-seg.pt,yolo26l-seg.pt,yolo26x-seg.pt')
if yolo_list:
    try:
        from ultralytics import YOLO
        for name in [x.strip() for x in yolo_list.split(',') if x.strip()]:
            # Always look in models/pt OR models/
            target_pt = models_dir / name
            alt_pt = models_dir / "pt" / name
            
            if target_pt.exists() or alt_pt.exists():
                print(f"[prepare_models] YOLO {name} already exists.")
                continue

            print(f"[prepare_models] Downloading YOLO model: {name}")
            try:
                # Force download to ROOT then move
                model = YOLO(name)
                # Ultralytics often leaves it in current dir
                if os.path.exists(name):
                    import shutil
                    # Prefer models/pt/
                    pt_dir = models_dir / "pt"
                    pt_dir.mkdir(exist_ok=True)
                    shutil.move(name, str(pt_dir / name))
                print(f"[prepare_models] YOLO {name} ready in {models_dir}")
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
    
# Ensure weights are in models/lwcc_weights for persistence
try:
    import shutil
    srcs = [Path('/') / '.lwcc' / 'weights', Path('/root') / '.lwcc' / 'weights']
    dst = ROOT / 'models' / 'lwcc_weights'
    dst.mkdir(parents=True, exist_ok=True)
    
    for src in srcs:
        if src.exists() and src.resolve() != dst.resolve():
            print(f"[prepare_models] Moving weights from {src} to {dst}")
            for f in src.glob('*.pth'):
                shutil.move(str(f), str(dst / f.name))
except Exception as e:
    print(f"[prepare_models] Warning: failed to relocate weights: {e}")

try:
    # Export density models to ONNX
    print("[prepare_models] Running export_density_to_onnx.py")
    run(f"python3 export_density_to_onnx.py")
except Exception as e:
    print(f"[prepare_models] Warning: export_density_to_onnx.py failed: {e}")

try:
    # Export YOLO models to TRT
    print("[prepare_models] Running export_yolos_to_trt.py")
    run(f"python3 export_yolos_to_trt.py")
except Exception as e:
    print(f"[prepare_models] Warning: export_yolos_to_trt.py failed: {e}")

# Convert any generated LWCC ONNX models to TensorRT if CUDA available
try:
    onnx_dir = ROOT / 'models' / 'onnx'
    trt_dir = ROOT / 'models' / 'tensorrt'
    onnx_dir.mkdir(parents=True, exist_ok=True)
    trt_dir.mkdir(parents=True, exist_ok=True)

    for onnx in onnx_dir.glob('*.onnx'):
        if "yolo" in onnx.name.lower():
            continue
        engine = trt_dir / (onnx.stem + '.engine')
        if not engine.exists():
            print(f"[prepare_models] Converting {onnx.name} -> {engine.name} (Batch 8)")
            try:
                # Use batch 8 (Original app setting) for density to match 2x2 tiling or multi-camera
                run(f"python3 convert_onnx_to_trt.py {onnx} {engine} 8")
            except Exception as ce:
                print(f"[prepare_models] Warning: convert_onnx_to_trt failed for {onnx}: {ce}")
except Exception as e:
    print(f"[prepare_models] Warning: LWCC ONNX->TRT conversion failed: {e}")

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

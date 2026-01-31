from ultralytics import YOLO
import os
import sys
from convert_onnx_to_trt import build_engine
import shutil
import subprocess

def main():
    # Only YOLO v26 models (skip v11/v12 as requested)
    models = [
        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
        "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt"
    ]
    
    # Use subdirectory structure
    root = os.getcwd()
    pt_dir = os.path.join(root, "models", "pt")
    onnx_dir = os.path.join(root, "models", "onnx")
    trt_dir = os.path.join(root, "models", "tensorrt")
    ov_dir = os.path.join(root, "models", "openvino")
    
    for d in [pt_dir, onnx_dir, trt_dir, ov_dir]:
        os.makedirs(d, exist_ok=True)

    for m_name in models:
        print(f"\n--- Processing {m_name} ---")
        pt_path = os.path.join(pt_dir, m_name)
        onnx_path = os.path.join(onnx_dir, m_name.replace(".pt", ".onnx"))
        engine_path = os.path.join(trt_dir, m_name.replace(".pt", ".engine"))
        
        # OpenVINO directory name used by Ultralytics
        ov_dirname_src = m_name.replace(".pt", "_openvino_model")
        ov_subdir_dst = os.path.join(ov_dir, ov_dirname_src)

        # Download/Fetch PT
        if not os.path.exists(pt_path):
            print(f"Downloading {m_name}...")
            model = YOLO(m_name)
            # Move to pt_dir
            downloaded = os.path.join(root, m_name)
            if os.path.exists(downloaded):
                shutil.move(downloaded, pt_path)
        else:
            model = YOLO(pt_path)

        # Export ONNX (Idempotent)
        if not os.path.exists(onnx_path):
            print(f"Exporting {m_name} to ONNX (opset 18)...")
            model.export(format="onnx", dynamic=True, opset=18)
            
            # Ultralytics exports to same folder as .pt or current dir
            src_onnx_local = m_name.replace(".pt", ".onnx")
            src_onnx_pt = pt_path.replace(".pt", ".onnx")
            
            if os.path.exists(src_onnx_local):
                shutil.move(src_onnx_local, onnx_path)
            elif os.path.exists(src_onnx_pt):
                shutil.move(src_onnx_pt, onnx_path)
            
            # Move .data file if it exists
            for src_onnx in [src_onnx_local, src_onnx_pt]:
                src_data = src_onnx + ".data"
                if os.path.exists(src_data):
                    shutil.move(src_data, onnx_path + ".data")
        else:
            print(f"ONNX already exists: {onnx_path}")

        # Export OpenVINO (Idempotent)
        if not os.path.exists(ov_subdir_dst):
            print(f"Exporting {m_name} to OpenVINO...")
            model.export(format="openvino")
            
            ov_source_local = m_name.replace(".pt", "_openvino_model")
            ov_source_pt = pt_path.replace(".pt", "_openvino_model")
            
            ov_source = None
            if os.path.exists(ov_source_local): ov_source = ov_source_local
            elif os.path.exists(ov_source_pt): ov_source = ov_source_pt

            if ov_source:
                if os.path.exists(ov_subdir_dst):
                    shutil.rmtree(ov_subdir_dst)
                shutil.move(ov_source, ov_subdir_dst)
        else:
            print(f"OpenVINO already exists in {ov_subdir_dst}")

        # Export TensorRT (Idempotent)
        has_cuda = False
        try:
            if shutil.which("nvidia-smi"):
                res = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                has_cuda = (res.returncode == 0)
        except Exception:
            has_cuda = False

        if not os.path.exists(engine_path):
            if has_cuda:
                print(f"Converting {onnx_path} to TensorRT...")
                success = build_engine(onnx_path, engine_path, max_batch_size=32)
                if success:
                    print(f"SUCCESS: {engine_path}")
            else:
                print("Skipping TensorRT (No CUDA)")
        else:
            print(f"Engine already exists: {engine_path}")

if __name__ == "__main__":
    main()

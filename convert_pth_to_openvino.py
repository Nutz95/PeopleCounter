import os
import sys
# Try relative path first
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "external", "lwcc"))
import torch
import onnx
from pathlib import Path
from lwcc.util.functions import weights_check
from lwcc.models.CSRNet import make_model as make_csrnet
from lwcc.models.SFANet import make_model as make_sfanet
from lwcc.models.Bay import make_model as make_bay
from lwcc.models.DMCount import make_model as make_dmcount

# OpenVINO Model Optimizer import (assume mo is in PATH or use absolute path)
# Example: mo_path = r"C:\Program Files (x86)\Intel\openvino_2023\tools\model_optimizer\mo.py"

MODELS = [
    ("CSRNet", "SHA"),
    ("CSRNet", "SHB"),
    ("SFANet", "SHB"),
    ("Bay", "QNRF"),
    ("Bay", "SHA"),
    ("Bay", "SHB"),
    ("DM-Count", "QNRF"),
    ("DM-Count", "SHA"),
    ("DM-Count", "SHB"),
]

def convert_pth_to_onnx(model_name, model_weights, pth_path, onnx_path):
    if os.path.isfile(onnx_path):
        print(f"[INFO] ONNX file already exists: {onnx_path}")
        return True
    
    print(f"[INFO] Loading {model_name} weights from {pth_path}")
    model = None
    # On utilise 544x960 car 544 est un multiple de 16 (requis par SFANet)
    target_h, target_w = 544, 960
    dummy_input = torch.randn(1, 3, target_h, target_w)
    if model_name == "CSRNet":
        model = make_csrnet(model_weights)
    elif model_name == "SFANet":
        model = make_sfanet(model_weights)
    elif model_name == "Bay":
        model = make_bay(model_weights)
    elif model_name == "DM-Count":
        model = make_dmcount(model_weights)
    else:
        return False
    
    model.eval()
    print(f"[INFO] Exporting to ONNX: {onnx_path} (opset 18, resolution {target_h}x{target_w})")
    
    # Pour éviter les erreurs avec les nouvelles versions de PyTorch (Dynamo), 
    # on désactive temporairement le tracking dynamique si ça pose problème ou on utilise fixed shapes.
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            opset_version=18, 
            input_names=['input'], 
            output_names=['output'],
            # On retire dynamic_axes pour SFANet si nécessaire, ou on tente un export plus "safe"
            dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}} if model_name != "SFANet" else None
        )
    except Exception as e:
        print(f"[WARN] Standard export failed, trying fixed shape export for {model_name}: {e}")
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            opset_version=18, 
            input_names=['input'], 
            output_names=['output']
        )
    return True

def convert_onnx_to_openvino(onnx_path, output_dir):
    xml_path = os.path.join(output_dir, os.path.splitext(os.path.basename(onnx_path))[0] + '.xml')
    if os.path.exists(xml_path):
        print(f"[INFO] OpenVINO XML already exists: {xml_path}")
        return
    
    print(f"[INFO] Converting {onnx_path} to OpenVINO IR...")
    try:
        import openvino as ov
        # Ensure we are passing the path as a string to avoid any Confusion
        core = ov.Core()
        ov_model = ov.convert_model(str(onnx_path))
        ov.save_model(ov_model, str(xml_path))
        print(f"[INFO] SUCCESS: {xml_path}")
    except Exception as e:
        print(f"[ERROR] OpenVINO conversion failed: {e}")

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.environ.get('LWCC_WEIGHTS_PATH', os.path.join(str(Path.home()), ".lwcc", "weights"))
    onnx_dir = os.path.join(repo_root, "models", "onnx")
    ov_dir = os.path.join(repo_root, "models", "openvino")
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(ov_dir, exist_ok=True)
    
    for model_name, model_weights in MODELS:
        # Standardize filenames: lower() and replace - with _
        filename = f"{model_name}_{model_weights}".lower().replace("-", "_")
        pth_path = os.path.join(weights_dir, f"{model_name}_{model_weights}.pth")
        onnx_path = os.path.join(onnx_dir, f"{filename}.onnx")
        
        if os.path.exists(pth_path):
            if convert_pth_to_onnx(model_name, model_weights, pth_path, onnx_path):
                convert_onnx_to_openvino(onnx_path, ov_dir)
        else:
            print(f"[WARN] Weight file not found: {pth_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

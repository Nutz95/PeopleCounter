import os
import sys
sys.path.append(r"E:\AI\lwcc")
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

# Dummy input shapes for each model (adjust as needed)
DUMMY_INPUT_SHAPE = {
    "CSRNet": (1, 3, 1920, 1072),
    "SFANet": (1, 3, 1920, 1072),
    "Bay": (1, 3, 1920, 1072),
    "DM-Count": (1, 3, 1920, 1072),
}

def convert_pth_to_onnx(model_name, model_weights, pth_path, onnx_path):
    if os.path.isfile(onnx_path):
        print(f"[INFO] ONNX file already exists: {onnx_path}, skipping export.")
        return
    print(f"[INFO] Loading model {model_name} with weights {pth_path}")
    model = None
    dummy_input = torch.randn(1, 3, 1072, 1920)  # Multiple de 16 pour tous les modèles
    if model_name == "CSRNet":
        model = make_csrnet(model_weights)
    elif model_name == "SFANet":
        model = make_sfanet(model_weights)
    elif model_name == "Bay":
        model = make_bay(model_weights)
    elif model_name == "DM-Count":
        model = make_dmcount(model_weights)
    else:
        print(f"[WARN] Unknown model: {model_name}")
        return
    model.eval()
    print(f"[INFO] Exporting to ONNX: {onnx_path}")
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
    print(f"[INFO] ONNX export done for {onnx_path}")

def convert_onnx_to_openvino(onnx_path, output_dir):
    print(f"[INFO] Converting {onnx_path} to OpenVINO IR in {output_dir}")
    try:
        import openvino as ov
        ir_model = ov.convert_model(onnx_path)
        xml_path = os.path.join(output_dir, os.path.splitext(os.path.basename(onnx_path))[0] + '.xml')
        ov.save_model(ir_model, xml_path)
        print(f"[INFO] OpenVINO IR saved to {xml_path}")
    except Exception as e:
        print(f"[ERROR] OpenVINO conversion failed for {onnx_path}: {e}")

def main():
    home = str(Path.home())
    weights_dir = os.path.join(home, ".lwcc", "weights")
    output_dir = os.path.join(home, ".lwcc", "openvino")
    os.makedirs(output_dir, exist_ok=True)
    for model_name, model_weights in MODELS:
        pth_path = os.path.join(weights_dir, f"{model_name}_{model_weights}.pth")
        onnx_path = os.path.join(output_dir, f"{model_name}_{model_weights}.onnx")
        if os.path.isfile(pth_path):
            convert_pth_to_onnx(model_name, model_weights, pth_path, onnx_path)
            convert_onnx_to_openvino(onnx_path, output_dir)
        else:
            print(f"[WARN] {pth_path} not found, skipping.")

if __name__ == "__main__":
    main()

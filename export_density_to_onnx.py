import os
import sys
# Try relative path first, then absolute from sys.argv or env
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "external", "lwcc"))
import torch
from pathlib import Path
from lwcc.models.DMCount import make_model as make_dmcount

def export_dmcount_qnrf():
    model_name = "DM-Count"
    model_weights = "QNRF"
    lwcc_weights_dir = os.environ.get('LWCC_WEIGHTS_PATH', os.path.join(root, "models", "lwcc_weights"))
    weights_path = os.path.join(lwcc_weights_dir, f"{model_name}_{model_weights}.pth")
    
    onnx_dir = os.path.join(root, "models", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "dm_count_qnrf.onnx")

    if os.path.exists(onnx_path):
        print(f"ONNX already exists: {onnx_path}")
        return

    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return

    model = make_dmcount(model_weights)
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    # Optionally strip common checkpoints prefixes before loading
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    # DUMMY INPUT with Batch support (B, C, H, W)
    # Target tile size is 640x720 to match the planned density tiling on 4K (6 columns × 3 rows) without overlap.
    target_h, target_w = 720, 640
    dummy_input = torch.randn(1, 3, target_h, target_w)
    
    print(f"Exporting DM-Count QNRF to {onnx_path} (opset 18, resolution {target_h}x{target_w})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Exported {onnx_path} successfully.")

if __name__ == "__main__":
    export_dmcount_qnrf()

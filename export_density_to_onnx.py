import os
import sys
sys.path.append(r"E:\AI\lwcc")
import torch
from pathlib import Path
from lwcc.models.DMCount import make_model as make_dmcount

def export_dmcount_qnrf():
    model_name = "DM-Count"
    model_weights = "QNRF"
    weights_path = os.path.join(str(Path.home()), ".lwcc", "weights", f"{model_name}_{model_weights}.pth")
    
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return

    model = make_dmcount(model_weights)
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    # Nettoyage optionnel de certains préfixes si nécessaire
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()

    # DUMMY INPUT with Batch support (B, C, H, W)
    # We use a 1080p-ish size for the ONNX export, but we make it dynamic
    dummy_input = torch.randn(1, 3, 1080, 1920)
    onnx_path = "dm_count_qnrf.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"Exported {onnx_path} successfully.")

if __name__ == "__main__":
    export_dmcount_qnrf()

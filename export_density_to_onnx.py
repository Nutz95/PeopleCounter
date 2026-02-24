import os
import sys
import argparse
# Try relative path first, then absolute from sys.argv or env
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "external", "lwcc"))
import torch
from pathlib import Path
from lwcc.models.DMCount import make_model as make_dmcount


def export_dmcount_qnrf(target_h: int = 720, target_w: int = 640):
    model_name = "DM-Count"
    model_weights = "QNRF"
    lwcc_weights_dir = os.environ.get('LWCC_WEIGHTS_PATH', os.path.join(root, "models", "lwcc_weights"))
    weights_path = os.path.join(lwcc_weights_dir, f"{model_name}_{model_weights}.pth")

    onnx_dir = os.path.join(root, "models", "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Resolution-specific filename: dm_count_qnrf.onnx for the default 640×720,
    # dm_count_qnrf_WxH.onnx for other resolutions.
    if target_w == 640 and target_h == 720:
        onnx_filename = "dm_count_qnrf.onnx"
    else:
        onnx_filename = f"dm_count_qnrf_{target_w}x{target_h}.onnx"
    onnx_path = os.path.join(onnx_dir, onnx_filename)

    if os.path.exists(onnx_path):
        print(f"ONNX already exists: {onnx_path}")
        return onnx_path

    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return None

    model = make_dmcount(model_weights)
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, target_h, target_w)

    print(f"Exporting DM-Count QNRF to {onnx_path} (opset 18, resolution {target_w}×{target_h})...")
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
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    print(f"Exported {onnx_path} successfully.")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DM-Count QNRF to ONNX.")
    parser.add_argument("--target-w", type=int, default=640,
                        help="Model input width  (must be multiple of 16). Default: 640")
    parser.add_argument("--target-h", type=int, default=720,
                        help="Model input height (must be multiple of 16). Default: 720")
    args = parser.parse_args()

    if args.target_w % 16 != 0 or args.target_h % 16 != 0:
        print(f"Error: target dimensions must be multiples of 16 "
              f"(got {args.target_w}×{args.target_h}).")
        sys.exit(1)

    export_dmcount_qnrf(target_h=args.target_h, target_w=args.target_w)

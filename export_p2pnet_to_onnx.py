#!/usr/bin/env python3
"""Export P2PNet (.pth) to ONNX.

This exporter is designed to run from the PeopleCounter workspace while loading
P2PNet source code from an external checkout (e.g. ``models/p2pnet-src``).
It avoids ambiguous ``from models import ...`` imports by loading the P2PNet
package explicitly from the provided source path.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.onnx

try:
    import torchvision
except Exception:  # pragma: no cover
    torchvision = None  # type: ignore[assignment]


class Args:
    """Mock args object for build_model."""
    def __init__(self, backbone='vgg16_bn', row=2, line=2):
        self.backbone = backbone
        self.row = row
        self.line = line
        self.point_loss_coef = 1.0
        self.eos_coef = 0.1


def _load_build_model(p2pnet_src: Path):
    """Load ``build_model`` from <p2pnet_src>/models/__init__.py safely."""
    module_path = p2pnet_src / "models" / "__init__.py"
    module_dir = module_path.parent
    if not module_path.exists():
        raise FileNotFoundError(f"P2PNet source not found: {module_path}")

    # Compatibility shim for upstream P2PNet util.misc:
    # it does `float(torchvision.__version__[:3]) < 0.7`, which mis-parses
    # modern versions like 0.19/0.20 as 0.1/0.2 and triggers deprecated
    # imports (`_new_empty_tensor`). Force the check to take the modern branch.
    if torchvision is not None:
        try:
            _v = str(getattr(torchvision, "__version__", ""))
            if _v.startswith("0.1") or _v.startswith("0.2"):
                torchvision.__version__ = "0.70.0"
        except Exception:
            pass

    # Ensure local package imports like "models.p2pnet" and "util.misc" resolve
    sys.path.insert(0, str(p2pnet_src))

    spec = importlib.util.spec_from_file_location(
        "p2pnet_models",
        str(module_path),
        submodule_search_locations=[str(module_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register package in sys.modules before exec so relative imports like
    # "from .p2pnet import build" resolve to p2pnet_models.p2pnet.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    build_model = getattr(module, "build_model", None)
    if build_model is None:
        raise RuntimeError("build_model not found in P2PNet models package")
    return build_model


def _patch_backbone_pretrained_flag(p2pnet_src: Path) -> None:
    """Disable hardcoded pretrained=True in upstream backbone for export.

    Upstream CrowdCounting-P2PNet expects local VGG checkpoint files at an
    internal absolute path. For inference export with a full checkpoint
    (SHTechA.pth), pretrained backbone init is unnecessary and breaks.
    """
    backbone_py = p2pnet_src / "models" / "backbone.py"
    if not backbone_py.exists():
        return

    text = backbone_py.read_text(encoding="utf-8")
    patched = text.replace("pretrained=True", "pretrained=False")
    if patched != text:
        backbone_py.write_text(patched, encoding="utf-8")
        print(f"Patched backbone pretrained flag for export: {backbone_py}")


def export_to_onnx(
    model_path: str,
    output_path: str,
    p2pnet_src: str,
    height: int = 1088,
    width: int = 1920,
) -> None:
    """
    Export P2PNet model to ONNX format.
    
    Args:
        model_path: Path to trained .pth model
        output_path: Path to save ONNX model
        height: Input height (must be multiple of 128)
        width: Input width (must be multiple of 128)
    """
    # VGG-based P2PNet requires feature-map friendly spatial sizes.
    # In practice multiples of 16 are sufficient for this model and match
    # our production density path (e.g. 1920x1088).
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be multiples of 16")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model from P2PNet source checkout
    p2pnet_src_path = Path(p2pnet_src).resolve()
    print(f"Using P2PNet source: {p2pnet_src_path}")
    _patch_backbone_pretrained_flag(p2pnet_src_path)
    build_model = _load_build_model(p2pnet_src_path)

    print(f"Building P2PNet model...")
    args = Args()
    model = build_model(args, training=False)
    model = model.to(device)
    
    # Load pretrained weights
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create dummy input (batch_size=1 for ONNX)
    print(f"Creating dummy input tensor ({height}x{width})...")
    dummy_input = torch.randn(1, 3, height, width, device=device)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['images'],
        output_names=['pred_logits', 'pred_points'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'pred_logits': {0: 'batch_size'},
            'pred_points': {0: 'batch_size'},
        },
        opset_version=18,
        do_constant_folding=True,
        verbose=True,
    )
    
    print(f"✓ Successfully exported to {output_path}")
    print(f"  Input:  [batch, 3, {height}, {width}]")
    print(f"  Output: pred_logits [batch, N_points, 2]")
    print(f"          pred_points [batch, N_points, 2]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export P2PNet to ONNX')
    parser.add_argument('--model_path', required=True, help='Path to P2PNet .pth weights')
    parser.add_argument('--output_path', required=True, help='Output path for ONNX model')
    parser.add_argument(
        '--p2pnet-src',
        default=os.environ.get('P2PNET_SRC', 'models/p2pnet-src'),
        help='Path to CrowdCounting-P2PNet source checkout',
    )
    parser.add_argument('--height', type=int, default=1088, help='Input height (multiple of 16)')
    parser.add_argument('--width', type=int, default=1920, help='Input width (multiple of 16)')
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        p2pnet_src=args.p2pnet_src,
        height=args.height,
        width=args.width,
    )

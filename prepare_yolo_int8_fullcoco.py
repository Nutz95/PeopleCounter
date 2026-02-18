#!/usr/bin/env python3
"""
INT8 TensorRT Engine Builder - FULL COCO calibration

Uses Ultralytics default COCO dataset (5000 val2017 images) for proper calibration.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    exit(1)


def build_int8_engine(
    model_path: Path,
    output_path: Path,
    batch_size: int = 32,
    img_size: int = 640,
    workspace: int = 8,
) -> None:
    """Build INT8 TensorRT engine with FULL COCO calibration."""
    
    print("🔧 Building INT8 TensorRT engine with FULL COCO dataset...")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}")
    print(f"   Workspace: {workspace}GB")
    print(f"   Calibration: FULL COCO val2017 (5000 images)")
    print(f"")
    print(f"⏱️  This will take 10-20 minutes (downloading + calibration)...")
    print(f"📦 Ultralytics will download COCO val2017 automatically (~1GB)")
    print(f"")
    
    # Load model
    model = YOLO(str(model_path))
    
    # Export with INT8 quantization
    # Ultralytics downloads COCO automatically and uses all val images
    model.export(
        format='engine',
        imgsz=img_size,
        int8=True,
        data='coco.yaml',  # ✅ Use default COCO (will download if needed)
        batch=batch_size,
        workspace=workspace,
        simplify=True,
        verbose=True,
    )
    
    # Move to final location
    default_output = Path(model_path).with_suffix('.engine')
    if default_output.exists() and default_output != Path(output_path):
        print(f"📦 Moving {default_output} → {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(default_output), str(output_path))
    
    print(f"✅ INT8 engine saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build INT8 TensorRT engine with full COCO calibration")
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt model")
    parser.add_argument("--output", type=Path, required=True, help="Output .engine path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workspace", type=int, default=8, help="Workspace size in GB")
    
    args = parser.parse_args()
    
    build_int8_engine(
        model_path=args.model,
        output_path=args.output,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        workspace=args.workspace,
    )

#!/usr/bin/env python3
"""
INT8 TensorRT Engine Builder - Optimized version

Downloads ONLY COCO val2017 (1GB) instead of full dataset (18GB+).
Uses reasonable batch size for faster calibration.
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


def download_coco_val2017_minimal(output_dir: Path, num_images: int = 300) -> Path:
    """Download minimal COCO val2017 for calibration using urllib."""
    import urllib.request
    import zipfile
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images" / "val2017"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already have enough images
    existing = list(images_dir.glob("*.jpg"))
    if len(existing) >= num_images:
        print(f"✅ Already have {len(existing)} calibration images in {images_dir}")
        return output_dir
    
    print(f"📥 Downloading COCO val2017 images (~1GB for {num_images} images)...")
    
    # Download val2017 images zip (1GB)
    val_zip_url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_file = output_dir / "val2017.zip"
    
    if not zip_file.exists():
        print(f"Downloading from {val_zip_url}...")
        print("This may take a few minutes (~1GB)...")
        
        # Download with progress
        def reporthook(blocknum, blocksize, totalsize):
            percent = min(100, blocknum * blocksize * 100 / totalsize)
            if blocknum % 100 == 0:  # Update every 100 blocks
                print(f"  Progress: {percent:.1f}% ({blocknum*blocksize/(1024*1024):.1f}MB / {totalsize/(1024*1024):.1f}MB)", end='\r')
        
        urllib.request.urlretrieve(val_zip_url, str(zip_file), reporthook=reporthook)
        print(f"\n✅ Downloaded {zip_file.stat().st_size / (1024*1024):.1f}MB")
    
    # Extract only N images (not all 5000!)
    print(f"Extracting {num_images} images...")
    with zipfile.ZipFile(zip_file) as zf:
        # List all image files and extract first N
        image_members = [m for m in zf.namelist() if m.endswith('.jpg')][:num_images]
        
        for i, member in enumerate(image_members):
            if i % 50 == 0:
                print(f"  Extracted {i}/{num_images} images...")
            
            # Extract to images/val2017/
            target_path = images_dir / Path(member).name
            with zf.open(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())
    
    print(f"✅ Extracted {len(list(images_dir.glob('*.jpg')))} images to {images_dir}")
    
    # Create minimal coco.yaml for Ultralytics
    yaml_file = output_dir / "coco_minimal.yaml"
    yaml_file.write_text(f"""# COCO minimal dataset for INT8 calibration
path: {output_dir.absolute()}
train: images/val2017  # Use val for calibration (no training)
val: images/val2017

names:
  0: person
  # (other classes omitted for brevity)
""")
    
    return output_dir


def clean_cache(cache_patterns: list[str] = None) -> None:
    """Clean TensorRT and Ultralytics cache files."""
    if cache_patterns is None:
        cache_patterns = [
            "*.cache",
            "*.trt",
            ".tensorrt_cache",
        ]
    
    print("🧹 Cleaning cache files...")
    for pattern in cache_patterns:
        for cache_file in Path(".").rglob(pattern):
            print(f"  Removing {cache_file}")
            cache_file.unlink()
    
    # Also clean Ultralytics cache in home
    ultralytics_cache = Path.home() / ".cache" / "ultralytics"
    if ultralytics_cache.exists():
        print(f"  Removing {ultralytics_cache}")
        shutil.rmtree(ultralytics_cache, ignore_errors=True)


def build_int8_engine(
    model_path: str,
    output_path: str,
    calibration_dataset: Path,
    img_size: int = 640,
    batch_size: int = 16,  # Reasonable batch for calibration
    workspace: int = 4,
) -> None:
    """Build INT8 TensorRT engine with custom calibration dataset."""
    print(f"📦 Loading model {model_path}...")
    model = YOLO(model_path)
    
    yaml_file = calibration_dataset / "coco_minimal.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"Dataset config not found: {yaml_file}")
    
    print(f"🔧 Building INT8 TensorRT engine...")
    print(f"   Image size: {img_size}x{img_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Workspace: {workspace}GB")
    print(f"   Calibration dataset: {yaml_file}")
    print(f"")
    print(f"⏱️  This will take 5-15 minutes...")
    
    # Export with INT8 quantization (fixed batch size)
    model.export(
        format='engine',
        imgsz=img_size,
        int8=True,
        data=str(yaml_file),  # Use our minimal dataset
        batch=batch_size,
        workspace=workspace,
        simplify=True,
        verbose=True,
    )
    
    # Move to final location
    default_output = Path(model_path).with_suffix('.engine')
    if default_output.exists() and default_output != Path(output_path):
        print(f"📦 Moving {default_output} → {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(default_output), output_path)
    
    print(f"✅ INT8 engine saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build INT8 TensorRT engine (optimized)")
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--output", required=True, help="Output .engine path")
    parser.add_argument("--calibration-images", type=int, default=300, help="Number of calibration images")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for calibration")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace (GB)")
    parser.add_argument("--clean-cache", action="store_true", help="Clean cache before building")
    parser.add_argument("--skip-download", action="store_true", help="Skip COCO download (use existing)")
    
    args = parser.parse_args()
    
    # Clean cache if requested
    if args.clean_cache:
        clean_cache()
    
    # Download calibration dataset
    calibration_dir = Path("./coco_val_minimal")
    if not args.skip_download:
        download_coco_val2017_minimal(calibration_dir, args.calibration_images)
    
    # Build INT8 engine
    build_int8_engine(
        model_path=args.model,
        output_path=args.output,
        calibration_dataset=calibration_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workspace=args.workspace,
    )


if __name__ == "__main__":
    main()

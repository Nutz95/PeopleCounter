#!/usr/bin/env python3
"""
Convert P2PNet ONNX model to TensorRT engine.

Usage:
    python convert_p2pnet_to_trt.py \
        --onnx_path models/onnx/p2pnet_1920x1088.onnx \
        --output_path models/tensorrt/p2pnet_1920x1088_fp16.engine \
        --precision fp16
"""

import argparse
import sys
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    print("ERROR: TensorRT not installed. Install with: pip install tensorrt")
    sys.exit(1)

import torch
import numpy as np


def build_engine(onnx_model_path, output_engine_path, precision='fp16', batch_size=1, height=1088, width=1920):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_model_path: Path to ONNX model
        output_engine_path: Path to save TRT engine
        precision: 'fp16' or 'fp32'
        batch_size: Batch size for optimization
        height: Input image height
        width: Input image width
    """
    
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Set builder config
    config = builder.create_builder_config()
    
    # Set precision
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building with FP16 precision")
    else:
        print("Building with FP32 precision")
    
    # Set optimization profiles (for dynamic shapes)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        (1, 3, height, width),           # min shape
        (batch_size, 3, height, width),  # opt shape
        (batch_size, 3, height, width)   # max shape
    )
    config.add_optimization_profile(profile)
    
    # Parse ONNX
    print(f"Loading ONNX model from {onnx_model_path}...")
    parser = trt.OnnxParser(builder, logger)
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Build engine
    print("Building TensorRT engine...")
    engine = builder.build_serialized_network(config)
    if engine is None:
        print("ERROR: Failed to build engine")
        return False
    
    # Save engine
    output_path = Path(output_engine_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving engine to {output_engine_path}...")
    with open(output_engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"✓ Successfully built TensorRT engine")
    print(f"  Input:  [batch, 3, {height}, {width}]")
    print(f"  Precision: {precision}")
    print(f"  Output file: {output_engine_path}")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert P2PNet ONNX to TensorRT')
    parser.add_argument('--onnx_path', required=True, help='Path to ONNX model')
    parser.add_argument('--output_path', required=True, help='Output path for TRT engine')
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16', help='Precision: fp16 or fp32')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for optimization')
    parser.add_argument('--height', type=int, default=1088, help='Input height')
    parser.add_argument('--width', type=int, default=1920, help='Input width')
    
    args = parser.parse_args()
    
    success = build_engine(
        onnx_model_path=args.onnx_path,
        output_engine_path=args.output_path,
        precision=args.precision,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
    )
    
    sys.exit(0 if success else 1)

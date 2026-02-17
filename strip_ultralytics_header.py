#!/usr/bin/env python3
"""
Strip Ultralytics JSON header from TensorRT engine files.

Ultralytics wraps engine files with:
- 4 bytes: uint32 length of JSON header
- N bytes: JSON metadata
- Rest: actual TensorRT engine

This script extracts only the TensorRT engine part.
"""
import argparse
import struct
import shutil
from pathlib import Path


def strip_ultralytics_header(input_path: Path, output_path: Path = None):
    """Strip Ultralytics JSON header from engine file."""
    if output_path is None:
        # Backup original
        backup_path = input_path.with_suffix(input_path.suffix + ".ultralytics")
        shutil.copy(input_path, backup_path)
        print(f"📦 Backup created: {backup_path}")
        output_path = input_path
    
    with open(input_path, 'rb') as f:
        # Read first 4 bytes (uint32 little-endian)
        header_len_bytes = f.read(4)
        if len(header_len_bytes) < 4:
            raise ValueError(f"File too small: {input_path}")
        
        header_len = struct.unpack('<I', header_len_bytes)[0]
        print(f"📄 JSON header length: {header_len} bytes")
        
        # Read the JSON header (just to validate)
        json_bytes = f.read(header_len)
        if len(json_bytes) < header_len:
            raise ValueError(f"Incomplete JSON header in {input_path}")
        
        print(f"📋 JSON: {json_bytes[:100].decode('utf-8', errors='ignore')}...")
        
        # Read remaining bytes (the actual TensorRT engine)
        engine_bytes = f.read()
        print(f"🔧 TensorRT engine size: {len(engine_bytes)} bytes ({len(engine_bytes) / 1024 / 1024:.1f} MB)")
    
    # Write stripped engine
    with open(output_path, 'wb') as f:
        f.write(engine_bytes)
    
    print(f"✅ Stripped engine saved: {output_path}")
    print(f"   Original size: {input_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Stripped size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Removed: {4 + header_len} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip Ultralytics JSON header from TensorRT engine")
    parser.add_argument("input", type=Path, help="Input .engine file")
    parser.add_argument("-o", "--output", type=Path, help="Output .engine file (default: overwrite input)")
    
    args = parser.parse_args()
    
    strip_ultralytics_header(args.input, args.output)

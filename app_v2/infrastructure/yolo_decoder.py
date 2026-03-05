"""yolo_decoder.py — backward-compatibility shim.

The decoder has been split into format-specific subclasses following the
Template Method (SOLID/SRP) pattern:
  YoloV8Decoder  (yolo_v8_decoder.py) — all standard YOLOv8 / ultralytics models
  YoloV5Decoder  (yolo_v5_decoder.py) — YOLO-CROWD (decoder_format="yolov5")
  YoloDecoderBase (yolo_decoder_base.py) — shared abstract base

This module re-exports ``YoloDecoder`` as an alias for ``YoloV8Decoder`` so
that existing callers and tests continue to work unchanged.

Instantiation in model classes uses a factory:
    _fmt = params.get("decoder_format", "yolov8")
    decoder = YoloV5Decoder(...) if _fmt == "yolov5" else YoloDecoder(...)
"""
from __future__ import annotations

from app_v2.infrastructure.yolo_v8_decoder import YoloV8Decoder as YoloDecoder

__all__ = ["YoloDecoder"]

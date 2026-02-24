"""Fusion strategy implementations.

Each strategy lives in its own module following the Open/Closed principle:
add a new strategy by creating a new file, not by modifying existing ones.
"""
from app_v2.core.strategies.base import FusionStrategy
from app_v2.core.strategies.simple import SimpleFusionStrategy
from app_v2.core.strategies.raw_stream import RawStreamFusionStrategy

__all__ = ["FusionStrategy", "SimpleFusionStrategy", "RawStreamFusionStrategy"]

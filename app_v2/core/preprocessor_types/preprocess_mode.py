from __future__ import annotations

from enum import Enum


class PreprocessMode(str, Enum):
    """Supported preprocess strategies."""

    GLOBAL = "global"
    TILES = "tiles"

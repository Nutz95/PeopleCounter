from __future__ import annotations

from dataclasses import dataclass

from app_v2.core.preprocessor_types.preprocess_mode import PreprocessMode


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Describes one model input preprocessing strategy."""

    model_name: str
    target_width: int
    target_height: int
    mode: PreprocessMode
    overlap: float = 0.0
    # Optional: size of each tile crop in the source frame before resize to
    # target_width × target_height.  0 means equal to target (no pre-resize).
    # Used for density tiles: e.g. source 1920×1080 → model input 640×720.
    source_tile_width: int = 0
    source_tile_height: int = 0

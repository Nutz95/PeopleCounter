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

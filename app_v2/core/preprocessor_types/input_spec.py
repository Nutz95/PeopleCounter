from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Describes one model input preprocessing strategy."""

    model_name: str
    target_width: int
    target_height: int
    mode: str
    overlap: float = 0.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from app_v2.core.preprocessor_types.preprocess_task import PreprocessTask


@dataclass(frozen=True, slots=True)
class PreprocessPlan:
    """Set of preprocessing tasks for one model on a given frame."""

    model_name: str
    frame_width: int
    frame_height: int
    tasks: tuple[PreprocessTask, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

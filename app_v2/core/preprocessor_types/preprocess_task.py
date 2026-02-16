from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PreprocessTask:
    """A single crop/resize task derived from an InputSpec."""

    model_name: str
    task_index: int
    source_x: int
    source_y: int
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Describes one model input preprocessing strategy."""

    model_name: str
    target_width: int
    target_height: int
    mode: str
    overlap: float = 0.0


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


@dataclass(frozen=True, slots=True)
class PreprocessPlan:
    """Set of preprocessing tasks for one model on a given frame."""

    model_name: str
    frame_width: int
    frame_height: int
    tasks: tuple[PreprocessTask, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreprocessOutput:
    """Materialized preprocessing output consumed by inference backends."""

    frame_id: int
    plans: Mapping[str, PreprocessPlan]
    model_inputs: Mapping[str, Sequence[Any]]

    def flatten_inputs(self, model_name: str | None = None) -> list[Any]:
        """Return model inputs as a flat list compatible with `InferenceModel.infer`."""
        if model_name is not None:
            return list(self.model_inputs.get(model_name, ()))

        flattened: list[Any] = []
        for key in sorted(self.model_inputs.keys()):
            flattened.extend(self.model_inputs[key])
        return flattened

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.preprocessor_types.preprocess_plan import PreprocessPlan


@dataclass(frozen=True, slots=True)
class PreprocessOutput:
    """Materialized preprocessing output consumed by inference backends."""

    frame_id: int
    plans: Mapping[str, PreprocessPlan]
    model_inputs: Mapping[str, Sequence[Any]]
    telemetry: FrameTelemetry | None = None

    def flatten_inputs(self, model_name: str | None = None) -> list[Any]:
        """Return model inputs as a flat list compatible with `InferenceModel.infer`."""
        if model_name is not None:
            return list(self.model_inputs.get(model_name, ()))

        flattened: list[Any] = []
        for key in sorted(self.model_inputs.keys()):
            flattened.extend(self.model_inputs[key])
        return flattened

from __future__ import annotations

from typing import Any, Sequence

from app_v2.application.gpu_preprocess_planner import GpuPreprocessPlanner
from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.core.preprocess_types import PreprocessOutput
from app_v2.core.preprocessor import Preprocessor


class GpuPreprocessor(Preprocessor):
    """GPU-first preprocess facade that builds plans and inference-compatible inputs."""

    def __init__(
        self,
        registry: InputSpecRegistry | None = None,
        planner: GpuPreprocessPlanner | None = None,
    ) -> None:
        self._registry = registry or InputSpecRegistry()
        self._planner = planner or GpuPreprocessPlanner()

    def configure(self, metadata: dict[str, Any]) -> None:
        self._registry.configure(metadata)

    def build_output(self, frame_id: int, frame: Any) -> PreprocessOutput:
        frame_width = int(getattr(frame, "width"))
        frame_height = int(getattr(frame, "height"))

        plans = {}
        model_inputs = {}
        for spec in self._registry.all_specs():
            plan = self._planner.build_plan(frame_width, frame_height, spec)
            plans[spec.model_name] = plan
            model_inputs[spec.model_name] = tuple(frame for _ in plan.tasks)

        return PreprocessOutput(frame_id=frame_id, plans=plans, model_inputs=model_inputs)

    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        output = self.build_output(frame_id, frame)
        return output.flatten_inputs()

from __future__ import annotations

from typing import Any, Sequence

from app_v2.application.gpu_preprocess_planner import GpuPreprocessPlanner
from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.preprocessor_types import GpuTensor, PreprocessOutput
from app_v2.core.preprocessor import Preprocessor
from app_v2.kernels.preprocess import run_letterbox_kernel, run_tiling_kernel


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
        telemetry: FrameTelemetry | None = getattr(frame, "telemetry", None)
        if telemetry:
            telemetry.mark_stage_start("preprocess")
        for spec in self._registry.all_specs():
            plan = self._planner.build_plan(frame_width, frame_height, spec)
            plans[spec.model_name] = plan
            inputs: list[GpuTensor] = []
            for task in plan.tasks:
                kernel_stage = f"preprocess_kernel_{spec.model_name}_{task.task_index}"
                if telemetry:
                    telemetry.mark_stage_start(kernel_stage)
                tensor = (
                    run_letterbox_kernel(frame, task)
                    if spec.mode == "global"
                    else run_tiling_kernel(frame, task)
                )
                if telemetry:
                    telemetry.mark_stage_end(kernel_stage)
                inputs.append(tensor)
            model_inputs[spec.model_name] = tuple(inputs)
        if telemetry:
            telemetry.mark_stage_end("preprocess")

        return PreprocessOutput(frame_id=frame_id, plans=plans, model_inputs=model_inputs, telemetry=telemetry)

    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        output = self.build_output(frame_id, frame)
        return output.flatten_inputs()

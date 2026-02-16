from __future__ import annotations

from typing import Any, Sequence

from app_v2.application.gpu_preprocess_planner import GpuPreprocessPlanner
from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.preprocessor_types import GpuTensor, PreprocessOutput
from app_v2.core.preprocessor import Preprocessor
from app_v2.infrastructure.gpu_tensor_pool import GpuTensorPool
from app_v2.kernels.preprocess import resolve_frame_source_tensor, run_letterbox_kernel, run_tiling_kernel


class GpuPreprocessor(Preprocessor):
    """GPU-first preprocess facade that builds plans and inference-compatible inputs."""

    def __init__(
        self,
        registry: InputSpecRegistry | None = None,
        planner: GpuPreprocessPlanner | None = None,
        tensor_pool: GpuTensorPool | None = None,
    ) -> None:
        self._registry = registry or InputSpecRegistry()
        self._planner = planner or GpuPreprocessPlanner()
        self._pool = tensor_pool or GpuTensorPool()
        self._stream_by_model: dict[str, int] = {}

    def configure(self, metadata: dict[str, Any]) -> None:
        self._registry.configure(metadata)
        self._stream_by_model = self._build_stream_map(metadata)

    def build_output(self, frame_id: int, frame: Any) -> PreprocessOutput:
        frame_width = int(getattr(frame, "width"))
        frame_height = int(getattr(frame, "height"))

        plans = {}
        model_inputs = {}
        telemetry: FrameTelemetry | None = getattr(frame, "telemetry", None)
        if telemetry:
            telemetry.mark_stage_start("preprocess")
        source_tensor = resolve_frame_source_tensor(frame)
        for spec in self._registry.all_specs():
            model_stage = f"preprocess_model_{spec.model_name}"
            if telemetry:
                telemetry.mark_stage_start(model_stage)
            plan = self._planner.build_plan(frame_width, frame_height, spec)
            plans[spec.model_name] = plan
            inputs: list[GpuTensor] = []
            stream_id = self._stream_by_model.get(spec.model_name, 0)
            for task in plan.tasks:
                kernel_stage = f"preprocess_kernel_{spec.model_name}_{task.task_index}"
                if telemetry:
                    telemetry.mark_stage_start(kernel_stage)
                tensor = (
                    run_letterbox_kernel(frame, task, stream=stream_id, pool=self._pool, source_tensor=source_tensor)
                    if spec.mode == "global"
                    else run_tiling_kernel(frame, task, stream=stream_id, pool=self._pool, source_tensor=source_tensor)
                )
                if telemetry:
                    telemetry.mark_stage_end(kernel_stage)
                inputs.append(tensor)
            model_inputs[spec.model_name] = tuple(inputs)
            if telemetry:
                telemetry.mark_stage_end(model_stage)
        if telemetry:
            telemetry.add_metrics(self._pool.stats_snapshot().as_dict(), prefix="tensor_pool_")
            telemetry.mark_stage_end("preprocess")

        return PreprocessOutput(frame_id=frame_id, plans=plans, model_inputs=model_inputs, telemetry=telemetry)

    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        output = self.build_output(frame_id, frame)
        return output.flatten_inputs()

    @staticmethod
    def _build_stream_map(metadata: dict[str, Any]) -> dict[str, int]:
        streams = metadata.get("streams", {})
        if not isinstance(streams, dict):
            return {}
        mapping: dict[str, int] = {}
        yolo_stream = int(streams.get("yolo", 0))
        density_stream = int(streams.get("density", 0))
        for model_name in metadata.get("preprocess", {}).keys():
            if str(model_name).startswith("yolo"):
                mapping[str(model_name)] = yolo_stream
            elif str(model_name).startswith("density") or str(model_name).startswith("lwcc"):
                mapping[str(model_name)] = density_stream
            else:
                mapping[str(model_name)] = 0
        return mapping

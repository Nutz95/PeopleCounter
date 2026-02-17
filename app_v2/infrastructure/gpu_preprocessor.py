from __future__ import annotations

from typing import Any, Sequence

from app_v2.application.gpu_preprocess_planner import GpuPreprocessPlanner
from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.preprocessor_types import GpuTensor, PreprocessOutput
from app_v2.core.preprocessor import Preprocessor
from app_v2.infrastructure.gpu_tensor_pool import GpuTensorPool
from app_v2.infrastructure.preprocess_stream_manager import PreprocessStreamManager
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
        self._pool_is_external = tensor_pool is not None
        self._pool = tensor_pool or GpuTensorPool()
        self._stream_manager = PreprocessStreamManager()

    def configure(self, metadata: dict[str, Any]) -> None:
        self._registry.configure(metadata)
        self._stream_manager.configure(metadata)
        if not self._pool_is_external:
            self._pool = self._build_pool(metadata)

    def build_output(self, frame_id: int, frame: Any) -> PreprocessOutput:
        frame_width = int(getattr(frame, "width"))
        frame_height = int(getattr(frame, "height"))

        plans = {}
        model_inputs = {}
        telemetry: FrameTelemetry | None = getattr(frame, "telemetry", None)
        stream_metrics: dict[str, float] = {}
        used_streams: set[int] = set()
        if telemetry:
            telemetry.mark_stage_start("preprocess")
            frame_timestamp_ns = getattr(frame, "timestamp_ns", None)
            if frame_timestamp_ns is not None:
                telemetry.add_metrics({"frame_timestamp_ns": int(frame_timestamp_ns)})
        source_tensor = resolve_frame_source_tensor(frame)
        for spec in self._registry.all_specs():
            model_stage = f"preprocess_model_{spec.model_name}"
            stream_id = self._stream_manager.stream_for_model(spec.model_name)
            used_streams.add(stream_id)
            stream_metrics[f"preprocess_stream_model_{spec.model_name}"] = float(stream_id)
            cuda_stream_handle = self._stream_manager.stream_handle(stream_id)
            if cuda_stream_handle is not None:
                stream_metrics[f"preprocess_stream_cuda_model_{spec.model_name}"] = float(cuda_stream_handle)
            if telemetry:
                telemetry.mark_stage_start(model_stage)
            plan = self._planner.build_plan(frame_width, frame_height, spec)
            plans[spec.model_name] = plan
            inputs: list[GpuTensor] = []
            with self._stream_manager.stream_context(stream_id):
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

        self._stream_manager.synchronize_streams(used_streams)
        if telemetry:
            telemetry.add_metrics(self._pool.stats_snapshot().as_dict(), prefix="tensor_pool_")
            telemetry.add_metrics(stream_metrics)
            telemetry.mark_stage_end("preprocess")
            self._add_parallelism_metrics(telemetry, model_inputs)

        return PreprocessOutput(frame_id=frame_id, plans=plans, model_inputs=model_inputs, telemetry=telemetry)

    def process(self, frame_id: int, frame: Any) -> Sequence[Any]:
        output = self.build_output(frame_id, frame)
        return output.flatten_inputs()

    @staticmethod
    def _build_pool(metadata: dict[str, Any]) -> GpuTensorPool:
        pool_config = metadata.get("tensor_pool", {})
        if not isinstance(pool_config, dict):
            return GpuTensorPool(max_per_key=64)
        max_per_key = int(pool_config.get("max_per_key", 64))
        return GpuTensorPool(max_per_key=max_per_key)

    @staticmethod
    def _add_parallelism_metrics(telemetry: FrameTelemetry, model_inputs: dict[str, Sequence[GpuTensor]]) -> None:
        snapshot = telemetry.snapshot()
        model_metrics: list[float] = []
        for model_name in model_inputs.keys():
            model_key = f"preprocess_model_{model_name}_ms"
            if model_key in snapshot:
                model_metrics.append(float(snapshot[model_key]))

        if not model_metrics:
            telemetry.add_metrics(
                {
                    "preprocess_model_sum_ms": 0.0,
                    "preprocess_model_max_ms": 0.0,
                    "preprocess_critical_path_ms": 0.0,
                    "preprocess_serial_overhead_ms": 0.0,
                    "preprocess_parallel_efficiency": 0.0,
                }
            )
            return

        model_sum = float(sum(model_metrics))
        model_max = float(max(model_metrics))
        preprocess_total = float(snapshot.get("preprocess_ms", 0.0))
        bridge_ms = float(snapshot.get("preprocess_nv12_bridge_ms", 0.0))
        critical_path = bridge_ms + model_max
        serial_overhead = max(0.0, preprocess_total - critical_path)

        telemetry.add_metrics(
            {
                "preprocess_model_sum_ms": model_sum,
                "preprocess_model_max_ms": model_max,
                "preprocess_critical_path_ms": critical_path,
                "preprocess_serial_overhead_ms": serial_overhead,
                "preprocess_parallel_efficiency": model_sum / model_max if model_max > 0.0 else 0.0,
            }
        )

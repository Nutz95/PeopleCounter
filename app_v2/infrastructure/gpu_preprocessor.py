from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

from app_v2.application.gpu_preprocess_planner import GpuPreprocessPlanner
from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.preprocessor_types import GpuTensor, PreprocessOutput
from app_v2.core.preprocessor import Preprocessor
from app_v2.infrastructure.gpu_tensor_pool import GpuTensorPool
from app_v2.kernels.preprocess import resolve_frame_source_tensor, run_letterbox_kernel, run_tiling_kernel

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


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
        self._stream_by_model: dict[str, int] = {}
        self._cuda_streams: dict[int, Any] = {}

    def configure(self, metadata: dict[str, Any]) -> None:
        self._registry.configure(metadata)
        self._stream_by_model = self._build_stream_map(metadata)
        self._cuda_streams = {}
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
        source_tensor = resolve_frame_source_tensor(frame)
        for spec in self._registry.all_specs():
            model_stage = f"preprocess_model_{spec.model_name}"
            stream_id = self._stream_by_model.get(spec.model_name, 0)
            used_streams.add(stream_id)
            stream_metrics[f"preprocess_stream_model_{spec.model_name}"] = float(stream_id)
            cuda_stream_handle = self._cuda_stream_handle(stream_id)
            if cuda_stream_handle is not None:
                stream_metrics[f"preprocess_stream_cuda_model_{spec.model_name}"] = float(cuda_stream_handle)
            if telemetry:
                telemetry.mark_stage_start(model_stage)
            plan = self._planner.build_plan(frame_width, frame_height, spec)
            plans[spec.model_name] = plan
            inputs: list[GpuTensor] = []
            with self._preprocess_stream_context(stream_id):
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

        self._synchronize_preprocess_streams(used_streams)
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
    def _build_stream_map(metadata: dict[str, Any]) -> dict[str, int]:
        streams = metadata.get("streams", {})
        if not isinstance(streams, dict):
            return {}
        mapping: dict[str, int] = {}
        yolo_stream = int(streams.get("yolo", 0))
        yolo_global_preprocess_stream = int(streams.get("yolo_global_preprocess", yolo_stream))
        yolo_tiles_preprocess_stream = int(streams.get("yolo_tiles_preprocess", yolo_stream))
        density_stream = int(streams.get("density", 0))
        density_preprocess_stream = int(streams.get("density_preprocess", density_stream))
        for model_name in metadata.get("preprocess", {}).keys():
            normalized_name = str(model_name)
            if normalized_name == "yolo_global":
                mapping[normalized_name] = yolo_global_preprocess_stream
            elif normalized_name.startswith("yolo_tiles"):
                mapping[normalized_name] = yolo_tiles_preprocess_stream
            elif str(model_name).startswith("density") or str(model_name).startswith("lwcc"):
                mapping[normalized_name] = density_preprocess_stream
            else:
                mapping[normalized_name] = 0
        return mapping

    @staticmethod
    def _build_pool(metadata: dict[str, Any]) -> GpuTensorPool:
        pool_config = metadata.get("tensor_pool", {})
        if not isinstance(pool_config, dict):
            return GpuTensorPool(max_per_key=64)
        max_per_key = int(pool_config.get("max_per_key", 64))
        return GpuTensorPool(max_per_key=max_per_key)

    def _preprocess_stream_context(self, stream_id: int) -> Any:
        stream = self._get_or_create_cuda_stream(stream_id)
        if stream is None:
            return nullcontext()
        assert torch is not None
        return torch.cuda.stream(stream)

    def _get_or_create_cuda_stream(self, stream_id: int) -> Any | None:
        if torch is None or not torch.cuda.is_available():
            return None
        normalized_stream_id = int(stream_id)
        if normalized_stream_id == 0:
            return torch.cuda.default_stream()
        existing = self._cuda_streams.get(normalized_stream_id)
        if existing is not None:
            return existing
        created = torch.cuda.Stream()
        self._cuda_streams[normalized_stream_id] = created
        return created

    def _synchronize_preprocess_streams(self, stream_ids: set[int]) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        for stream_id in sorted(stream_ids):
            stream = self._get_or_create_cuda_stream(stream_id)
            if stream is not None:
                stream.synchronize()

    def _cuda_stream_handle(self, stream_id: int) -> int | None:
        stream = self._get_or_create_cuda_stream(stream_id)
        if stream is None:
            return None
        handle = getattr(stream, "cuda_stream", None)
        if handle is None:
            return None
        return int(handle)

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

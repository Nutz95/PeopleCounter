from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Sequence

from app_v2.application.gpu_preprocess_planner import GpuPreprocessPlanner
from app_v2.application.input_spec_registry import InputSpecRegistry
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.preprocessor_types import GpuTensor, PreprocessMode, PreprocessOutput
from app_v2.core.preprocessor import Preprocessor
from app_v2.core.telemetry_keys import (
    FRAME_TIMESTAMP_NS,
    PREPROCESS_CRITICAL_PATH_MS,
    PREPROCESS_MODEL_MAX_MS,
    PREPROCESS_MODEL_SUM_MS,
    PREPROCESS_PARALLEL_EFFICIENCY,
    PREPROCESS_SERIAL_OVERHEAD_MS,
    preprocess_model_metric_key,
    preprocess_model_stage_name,
    preprocess_stream_cuda_model_key,
    preprocess_stream_model_key,
)
from app_v2.infrastructure.gpu_tensor_pool import GpuTensorPool
from app_v2.infrastructure.preprocess_stream_manager import PreprocessStreamManager
from app_v2.kernels.preprocess import (
    resolve_frame_source_tensor,
    run_letterbox_kernel_fused,
    run_tiling_kernel_fused_batch,
)


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
        # Persistent thread pool — created once, reused across all frames.
        # Avoids the ~4 ms per-frame overhead of creating/destroying threads
        # inside build_output().  Max workers = max supported specs (2 here).
        self._executor: ThreadPoolExecutor | None = None

    def configure(self, metadata: dict[str, Any]) -> None:
        self._registry.configure(metadata)
        self._stream_manager.configure(metadata)
        if not self._pool_is_external:
            self._pool = self._build_pool(metadata)
        # (Re-)create the persistent executor sized to the number of specs.
        num_specs = len(list(self._registry.all_specs()))
        if num_specs > 1:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=num_specs)
        else:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self._executor = None

    def _dispatch_one_spec(
        self,
        spec: Any,
        frame: Any,
        frame_width: int,
        frame_height: int,
        source_tensor: Any,
    ) -> tuple[str, Any, tuple[Any, ...], int, dict[str, float], float]:
        """Dispatch preprocess for one model spec on its dedicated CUDA stream.

        Returns (model_name, plan, inputs, stream_id, stream_metrics, elapsed_ms).
        Designed to run in a background thread — each thread sets its own CUDA
        stream context.  The tensor pool and stream manager are already
        thread-safe at this point (streams pre-created in configure()).
        """
        stream_id = self._stream_manager.stream_for_model(spec.model_name)
        stream_metrics: dict[str, float] = {
            preprocess_stream_model_key(spec.model_name): float(stream_id),
        }
        cuda_stream_handle = self._stream_manager.stream_handle(stream_id)
        if cuda_stream_handle is not None:
            stream_metrics[preprocess_stream_cuda_model_key(spec.model_name)] = float(cuda_stream_handle)

        t_start_ns = time.monotonic_ns()
        plan = self._planner.build_plan(frame_width, frame_height, spec)
        inputs: list[GpuTensor] = []
        with self._stream_manager.stream_context(stream_id):
            if spec.mode is PreprocessMode.TILES:
                inputs = run_tiling_kernel_fused_batch(
                    frame, list(plan.tasks), stream=stream_id, pool=self._pool
                )
            else:
                for task in plan.tasks:
                    tensor = run_letterbox_kernel_fused(
                        frame, task, stream=stream_id, pool=self._pool, source_tensor=source_tensor
                    )
                    inputs.append(tensor)
        elapsed_ms = (time.monotonic_ns() - t_start_ns) / 1_000_000.0
        return spec.model_name, plan, tuple(inputs), stream_id, stream_metrics, elapsed_ms

    def build_output(self, frame_id: int, frame: Any) -> PreprocessOutput:
        frame_width = int(getattr(frame, "width"))
        frame_height = int(getattr(frame, "height"))

        telemetry: FrameTelemetry | None = getattr(frame, "telemetry", None)
        if telemetry:
            telemetry.mark_stage_start("preprocess")
            frame_timestamp_ns = getattr(frame, "timestamp_ns", None)
            if frame_timestamp_ns is not None:
                telemetry.add_metrics({FRAME_TIMESTAMP_NS: int(frame_timestamp_ns)})

        # For raw NV12 frames, skip the full-frame decode: each fused kernel
        # handles its own crop/letterbox directly from the Y/UV device pointers.
        is_nv12 = getattr(frame, "device_ptr_y", None) is not None
        source_tensor = None if is_nv12 else resolve_frame_source_tensor(frame)

        specs = list(self._registry.all_specs())
        plans: dict[str, Any] = {}
        model_inputs: dict[str, tuple[Any, ...]] = {}
        used_streams: set[int] = set()
        merged_stream_metrics: dict[str, float] = {}
        per_model_ms: dict[str, float] = {}

        if len(specs) > 1:
            # Parallel dispatch using the persistent executor (no thread-creation overhead).
            executor = self._executor
            if executor is None:
                # Fallback: executor not yet created (e.g. configure() not called).
                executor = ThreadPoolExecutor(max_workers=len(specs))
            t_dispatch_start = time.monotonic_ns()
            futures: list[Future[Any]] = [
                executor.submit(self._dispatch_one_spec, spec, frame, frame_width, frame_height, source_tensor)
                for spec in specs
            ]
            results = [f.result() for f in futures]
            t_dispatch_end = time.monotonic_ns()
        else:
            t_dispatch_start = time.monotonic_ns()
            results = [
                self._dispatch_one_spec(specs[0], frame, frame_width, frame_height, source_tensor)
            ]
            t_dispatch_end = time.monotonic_ns()

        for model_name, plan, inputs, stream_id, stream_metrics, elapsed_ms in results:
            plans[model_name] = plan
            model_inputs[model_name] = inputs
            used_streams.add(stream_id)
            merged_stream_metrics.update(stream_metrics)
            per_model_ms[model_name] = elapsed_ms

        t_sync_start = time.monotonic_ns()
        # Record events on each preprocess stream instead of blocking the CPU.
        # The inference layer will call stream.wait_event() to create a GPU-side
        # ordering dependency, so the host never stalls on preprocess completion.
        cuda_events = self._stream_manager.record_events(used_streams)
        t_sync_end = time.monotonic_ns()

        if telemetry:
            # Record per-model stage timings collected from threads
            for model_name, elapsed_ms in per_model_ms.items():
                model_stage = preprocess_model_stage_name(model_name)
                telemetry.add_metrics({model_stage + "_ms": elapsed_ms})
            telemetry.add_metrics(self._pool.stats_snapshot().as_dict(), prefix="tensor_pool_")
            telemetry.add_metrics(merged_stream_metrics)
            # Sub-breakdown metrics to diagnose serial overhead sources
            dispatch_ms = (t_dispatch_end - t_dispatch_start) / 1_000_000.0
            sync_ms = (t_sync_end - t_sync_start) / 1_000_000.0
            telemetry.add_metrics({
                "preprocess_dispatch_ms": dispatch_ms,
                "preprocess_sync_ms": sync_ms,
            })
            telemetry.mark_stage_end("preprocess")
            self._add_parallelism_metrics(telemetry, model_inputs)

        return PreprocessOutput(frame_id=frame_id, plans=plans, model_inputs=model_inputs, telemetry=telemetry, cuda_events=cuda_events)

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
            model_key = preprocess_model_metric_key(model_name)
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
                PREPROCESS_MODEL_SUM_MS: model_sum,
                PREPROCESS_MODEL_MAX_MS: model_max,
                PREPROCESS_CRITICAL_PATH_MS: critical_path,
                PREPROCESS_SERIAL_OVERHEAD_MS: serial_overhead,
                PREPROCESS_PARALLEL_EFFICIENCY: model_sum / model_max if model_max > 0.0 else 0.0,
            }
        )

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
import sys
import time
from typing import Any

from app_v2.core.strategies import FusionStrategy
from app_v2.core.result_publisher import ResultPublisher
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.telemetry_keys import (
    DECODE_MODEL_SUM_MS,
    END_TO_END_MS,
    FRAME_TIMESTAMP_NS,
    FUSION_WAIT_MS,
    INFERENCE_MODEL_MAX_MS,
    INFERENCE_MODEL_SUM_MS,
    OVERLAY_LAG_MS,
    decode_model_metric_key,
    inference_model_metric_key,
)
from logger.filtered_logger import LogChannel, info as log_info, debug as log_debug

_PERF_LOG: bool = os.environ.get("PERF_LOG", "0").strip() not in ("", "0", "false", "no")


class ResultAggregator:
    """Collects per-model answers and applies the configured fusion strategy."""

    def __init__(self, fusion_strategy: FusionStrategy, publisher: ResultPublisher) -> None:
        self.fusion_strategy = fusion_strategy
        self.publisher = publisher
        self._buffers: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._telemetry: dict[int, FrameTelemetry] = {}
        self._release_hooks: dict[int, list[Any]] = defaultdict(list)
        self._collect_started_ns: dict[int, int] = {}

    def collect(self, frame_id: int, payload: dict[str, Any]) -> None:
        """Store partial payloads and publish when fusion strategy says so."""
        self._collect_started_ns.setdefault(frame_id, time.time_ns())
        self._buffers[frame_id].append(payload)
        collected = self._buffers[frame_id]
        if self.fusion_strategy.should_publish(frame_id, [frame_id for _ in collected]):
            merged = self.fusion_strategy.merge(collected)
            payload = merged if isinstance(merged, Sequence) else [merged]
            telemetry = self._telemetry.pop(frame_id, None)
            publish_ns = time.time_ns()
            _t0 = time.perf_counter_ns() if _PERF_LOG else 0
            payload_list = [self._strip_internal_metrics(entry) for entry in list(payload)]
            _t_strip = time.perf_counter_ns() if _PERF_LOG else 0
            if telemetry:
                telemetry.add_metrics(self._build_publication_metrics(frame_id, list(payload), publish_ns, telemetry))
                _t_metrics = time.perf_counter_ns() if _PERF_LOG else 0
                payload_list.append({"telemetry": telemetry.snapshot()})
                _t_snapshot = time.perf_counter_ns() if _PERF_LOG else 0
            else:
                _t_metrics = _t_strip
                _t_snapshot = _t_strip
            self.publisher.publish(frame_id, payload_list)
            _t_publish = time.perf_counter_ns() if _PERF_LOG else 0
            if _PERF_LOG:
                _ms = lambda a, b: f"{(b - a) / 1e6:.2f}"  # noqa: E731
                print(
                    f"[PERF2] f={frame_id}"
                    f" strip={_ms(_t0, _t_strip)}"
                    f" metrics={_ms(_t_strip, _t_metrics)}"
                    f" snapshot={_ms(_t_metrics, _t_snapshot)}"
                    f" publish={_ms(_t_snapshot, _t_publish)}"
                    f" total={_ms(_t0, _t_publish)}ms",
                    file=sys.stderr, flush=True,
                )
            self._run_release_hooks(frame_id)
            log_debug(
                LogChannel.GLOBAL,
                f"Published frame {frame_id} with {len(payload_list)} payloads via {self.fusion_strategy.strategy_type.value}",
            )
            self._buffers.pop(frame_id, None)
            self._collect_started_ns.pop(frame_id, None)

    def attach_telemetry(self, frame_id: int, telemetry: FrameTelemetry | None) -> None:
        if telemetry:
            self._telemetry[frame_id] = telemetry

    def discard_frame(self, frame_id: int) -> None:
        """Release all resources for a frame that had no inference payloads.

        Used in passthrough mode (no active models) so the ring-buffer slot is
        freed immediately instead of leaking until the aggregator is destroyed.
        """
        self._telemetry.pop(frame_id, None)
        self._buffers.pop(frame_id, None)
        self._collect_started_ns.pop(frame_id, None)
        self._run_release_hooks(frame_id)

    def attach_release_hook(self, frame_id: int, hook: Any) -> None:
        if callable(hook):
            self._release_hooks[frame_id].append(hook)

    def _run_release_hooks(self, frame_id: int) -> None:
        hooks = self._release_hooks.pop(frame_id, [])
        for hook in hooks:
            hook()

    def _build_publication_metrics(
        self,
        frame_id: int,
        payload: Sequence[dict[str, Any]],
        publish_ns: int,
        telemetry: FrameTelemetry,
    ) -> dict[str, float]:
        collect_started_ns = self._collect_started_ns.get(frame_id, publish_ns)
        fusion_wait_ms = max(0.0, (publish_ns - collect_started_ns) / 1_000_000.0)

        model_done_ns: list[int] = []
        for item in payload:
            value = item.get("_inference_done_ns")
            if isinstance(value, int):
                model_done_ns.append(value)
        overlay_lag_ms = 0.0
        if model_done_ns:
            overlay_lag_ms = max(0.0, (max(model_done_ns) - min(model_done_ns)) / 1_000_000.0)

        snapshot = telemetry.snapshot()
        frame_timestamp_ns = snapshot.get(FRAME_TIMESTAMP_NS)
        end_to_end_ms = 0.0
        if isinstance(frame_timestamp_ns, (int, float)):
            end_to_end_ms = max(0.0, (publish_ns - int(frame_timestamp_ns)) / 1_000_000.0)

        inference_metrics = self._collect_inference_metrics(payload)

        metrics: dict[str, float] = {
            FUSION_WAIT_MS: fusion_wait_ms,
            OVERLAY_LAG_MS: overlay_lag_ms,
            END_TO_END_MS: end_to_end_ms,
        }
        metrics.update(inference_metrics)
        return metrics

    @staticmethod
    def _collect_inference_metrics(payload: Sequence[dict[str, Any]]) -> dict[str, float]:
        per_model: dict[str, float] = {}
        infer_values: list[float] = []
        decode_values: list[float] = []
        for item in payload:
            model_name = item.get("model")
            infer_value = item.get("inference_ms")
            if isinstance(infer_value, (int, float)):
                v = float(infer_value)
                infer_values.append(v)
                if isinstance(model_name, str) and model_name:
                    key = inference_model_metric_key(model_name)
                    per_model[key] = max(per_model.get(key, 0.0), v)
            decode_value = item.get("decode_ms")
            if isinstance(decode_value, (int, float)):
                dv = float(decode_value)
                decode_values.append(dv)
                if isinstance(model_name, str) and model_name:
                    dkey = decode_model_metric_key(model_name)
                    per_model[dkey] = max(per_model.get(dkey, 0.0), dv)

        if infer_values:
            per_model[INFERENCE_MODEL_SUM_MS] = sum(infer_values)
            per_model[INFERENCE_MODEL_MAX_MS] = max(infer_values)
        if decode_values:
            per_model[DECODE_MODEL_SUM_MS] = sum(decode_values)
        return per_model

    @staticmethod
    def _strip_internal_metrics(payload: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(payload)
        sanitized.pop("_inference_done_ns", None)
        # Safety net: remove any nested CUDA tensors that would cause
        # json.dumps(default=str) to trigger GPU→CPU sync for each tensor value.
        if "prediction" in sanitized and isinstance(sanitized["prediction"], dict):
            pred = sanitized["prediction"]
            if any(k in pred for k in ("outputs", "output_tensors", "inputs", "input_tensor")):
                sanitized["prediction"] = {
                    k: v for k, v in pred.items()
                    if k not in ("outputs", "output_tensors", "inputs", "input_tensor")
                }
        return sanitized
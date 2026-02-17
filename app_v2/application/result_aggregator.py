from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import time
from typing import Any

from app_v2.core.fusion_strategy import FusionStrategy
from app_v2.core.result_publisher import ResultPublisher
from app_v2.core.frame_telemetry import FrameTelemetry
from app_v2.core.telemetry_keys import (
    END_TO_END_MS,
    FRAME_TIMESTAMP_NS,
    FUSION_WAIT_MS,
    INFERENCE_MODEL_MAX_MS,
    INFERENCE_MODEL_SUM_MS,
    OVERLAY_LAG_MS,
    inference_model_metric_key,
)
from logger.filtered_logger import LogChannel, info as log_info


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
            payload_list = [self._strip_internal_metrics(entry) for entry in list(payload)]
            if telemetry:
                telemetry.add_metrics(self._build_publication_metrics(frame_id, list(payload), publish_ns, telemetry))
                payload_list.append({"telemetry": telemetry.snapshot()})
            self.publisher.publish(frame_id, payload_list)
            self._run_release_hooks(frame_id)
            log_info(
                LogChannel.GLOBAL,
                f"Published frame {frame_id} with {len(payload_list)} payloads via {self.fusion_strategy.strategy_type.value}",
            )
            self._buffers.pop(frame_id, None)
            self._collect_started_ns.pop(frame_id, None)

    def attach_telemetry(self, frame_id: int, telemetry: FrameTelemetry | None) -> None:
        if telemetry:
            self._telemetry[frame_id] = telemetry

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
        values: list[float] = []
        for item in payload:
            metric_value = item.get("yolo_inference_ms", item.get("inference_ms"))
            if not isinstance(metric_value, (int, float)):
                continue
            value = float(metric_value)
            values.append(value)
            model_name = item.get("model")
            if isinstance(model_name, str) and model_name:
                key = inference_model_metric_key(model_name)
                per_model[key] = max(per_model.get(key, 0.0), value)

        if values:
            per_model[INFERENCE_MODEL_SUM_MS] = sum(values)
            per_model[INFERENCE_MODEL_MAX_MS] = max(values)
        return per_model

    @staticmethod
    def _strip_internal_metrics(payload: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(payload)
        sanitized.pop("_inference_done_ns", None)
        return sanitized
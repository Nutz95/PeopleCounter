from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any, Dict

try:
    import torch
except Exception:  # pragma: no cover - cuda bindings missing on host
    torch = None  # type: ignore[assignment]


@dataclass
class _StageRecord:
    start: Any | None = None
    end: Any | None = None


class FrameTelemetry:
    """GPU-aware telemetry recorder attached to each frame."""

    def __init__(self, frame_id: int, stream_name: str | None = None) -> None:
        self.frame_id = frame_id
        self.stream_name = stream_name
        self._records: Dict[str, _StageRecord] = defaultdict(_StageRecord)
        self._metrics: Dict[str, Any] = {}
        self._gpu_timing_enabled = bool(torch is not None and torch.cuda.is_available())

    def mark_stage_start(self, stage: str, stream: int = 0) -> None:
        record = self._records[stage]
        record.start = self._create_event(stream)

    def mark_stage_end(self, stage: str, stream: int = 0) -> None:
        record = self._records[stage]
        record.end = self._create_event(stream)

    def snapshot(self) -> Dict[str, Any]:
        durations: Dict[str, Any] = {}
        for stage, record in self._records.items():
            if record.start is None or record.end is None:
                continue
            durations[f"{stage}_ms"] = self._compute_duration(record)
        durations.update(self._metrics)
        if self.stream_name:
            durations["stream_name"] = self.stream_name
        durations["frame_id"] = float(self.frame_id)
        durations["gpu_timing_enabled"] = self._gpu_timing_enabled
        return durations

    def add_metrics(self, metrics: Dict[str, Any], *, prefix: str = "") -> None:
        for key, value in metrics.items():
            name = f"{prefix}{key}" if prefix else key
            self._metrics[name] = value

    def _compute_duration(self, record: _StageRecord) -> float:
        # CPU wall-clock fallback (no CUDA or no GPU timing).
        if isinstance(record.start, float) and isinstance(record.end, float):
            return (record.end - record.start) * 1000.0
        if not self._gpu_timing_enabled:
            return 0.0
        assert torch is not None
        assert hasattr(record.start, "elapsed_time")
        # elapsed_time() implicitly syncs only the two events involved — no global
        # GPU stall.  Replacing the previous record.end.synchronize() + elapsed_time()
        # pair removes a full device sync from every published frame's hot path.
        try:
            return float(record.start.elapsed_time(record.end))
        except RuntimeError:
            # Events not yet recorded (e.g. stage was skipped) — return 0.
            return 0.0

    @staticmethod
    def _create_event(stream: int) -> Any | None:
        del stream
        if torch is None or not torch.cuda.is_available():
            return time.perf_counter()  # CPU wall-clock fallback
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream())
        return event
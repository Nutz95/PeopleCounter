from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
        if self.stream_name:
            durations["stream_name"] = self.stream_name
        durations["frame_id"] = float(self.frame_id)
        durations["gpu_timing_enabled"] = self._gpu_timing_enabled
        return durations

    def _compute_duration(self, record: _StageRecord) -> float:
        if not self._gpu_timing_enabled:
            return 0.0
        assert torch is not None
        assert hasattr(record.start, "elapsed_time")
        assert hasattr(record.end, "synchronize")
        record.end.synchronize()
        return float(record.start.elapsed_time(record.end))

    @staticmethod
    def _create_event(stream: int) -> Any | None:
        del stream
        if torch is None or not torch.cuda.is_available():
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream())
        return event
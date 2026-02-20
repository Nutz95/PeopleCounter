from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import ContextManager, Dict, Iterator

from logger.filtered_logger import LogChannel, debug as log_debug


class PerformanceTracker:
    """Collects per-frame timing budgets so we can compare fusion strategies."""

    def __init__(self) -> None:
        self._starts: Dict[int, Dict[str, float]] = defaultdict(dict)
        self._history: Dict[int, Dict[str, float]] = defaultdict(dict)

    def start(self, frame_id: int, stage: str) -> None:
        """Mark the beginning of a stage for the provided frame_id."""
        self._starts[frame_id][stage] = time.perf_counter()

    def stop(self, frame_id: int, stage: str) -> float | None:
        """Record the elapsed time for a stage and log the duration."""
        start = self._starts.get(frame_id, {}).pop(stage, None)
        if start is None:
            return None
        duration = time.perf_counter() - start
        self._history[frame_id][stage] = duration
        log_debug(LogChannel.GLOBAL, f"Frame {frame_id} stage {stage} -> {duration * 1000:.2f} ms")
        return duration

    def get_summary(self, frame_id: int) -> Dict[str, float]:
        """Return the accumulated durations for all stages of a frame."""
        return dict(self._history.get(frame_id, {}))

    def clear(self, frame_id: int) -> None:
        """Drop stored timings for a frame once the metrics have been consumed."""
        self._starts.pop(frame_id, None)
        self._history.pop(frame_id, None)

    @contextmanager
    def stage(self, frame_id: int, stage: str) -> Iterator[None]:
        """Context manager that wraps the timing of a stage."""
        self.start(frame_id, stage)
        try:
            yield
        finally:
            self.stop(frame_id, stage)

import time
from collections import deque
from typing import Deque, Optional


class MetricsCollector:
    def __init__(self) -> None:
        self._timestamps: Deque[float] = deque(maxlen=128)
        self.last_capture_time: Optional[float] = None
        self.last_inference_time: Optional[float] = None

    def record_capture(self) -> None:
        now = time.time()
        self.last_capture_time = now
        self._timestamps.append(now)

    def record_inference(self) -> None:
        self.last_inference_time = time.time()

    def average_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        interval = self._timestamps[-1] - self._timestamps[0]
        if interval <= 0:
            return 0.0
        return len(self._timestamps) / interval

    def clear(self) -> None:
        self._timestamps.clear()
        self.last_capture_time = None
        self.last_inference_time = None

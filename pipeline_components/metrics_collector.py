import time
from collections import deque
from typing import Deque, Dict, Optional


class MetricsCollector:
    def __init__(self, history_len: int = 600) -> None:
        self.history_len = max(1, history_len)
        self._history: Deque[Dict[str, float]] = deque(maxlen=self.history_len)
        self.last_capture_time: Optional[float] = None
        self.last_inference_time: Optional[float] = None

    def append(self, yolo_value: float, density_value: float, average_value: float) -> None:
        entry = {
            "timestamp": time.time(),
            "yolo": float(yolo_value),
            "density": float(density_value),
            "average": float(average_value),
        }
        self._history.append(entry)

    def snapshot(self) -> list[Dict[str, float]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

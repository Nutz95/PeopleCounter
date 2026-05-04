from __future__ import annotations

import threading
import time


class FramePacer:
    def __init__(self, fps: int) -> None:
        self._interval_s = 1.0 / fps
        self._next_deadline = time.perf_counter()

    def reset(self) -> None:
        self._next_deadline = time.perf_counter()

    def wait_next(self, stop_event: threading.Event | None = None) -> None:
        self._next_deadline += self._interval_s
        remaining = self._next_deadline - time.perf_counter()
        if remaining <= 0:
            return
        if stop_event is not None:
            stop_event.wait(remaining)
        else:
            time.sleep(remaining)

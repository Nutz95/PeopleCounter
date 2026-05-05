from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class GpuMonitor:
    """Background GPU telemetry poller (pynvml, 1 Hz)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._available: bool = False
        self._last: dict[str, int] = {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 0}
        self._history: deque[int] = deque(maxlen=60)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="gpu-poller")
        self._thread.start()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if not self._available:
                return {"available": False}
            return {
                "available": True,
                "gpu_util": self._last.get("gpu_util", 0),
                "mem_used_mb": self._last.get("mem_used_mb", 0),
                "mem_total_mb": self._last.get("mem_total_mb", 0),
                "history_util": list(self._history),
            }

    def _poll_loop(self) -> None:
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml_ok = True
        except Exception:
            pynvml_ok = False
            handle = None

        while True:
            if pynvml_ok and handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    sample = {
                        "gpu_util": int(util.gpu),
                        "mem_used_mb": int(mem.used // (1024 * 1024)),
                        "mem_total_mb": int(mem.total // (1024 * 1024)),
                    }
                    with self._lock:
                        self._available = True
                        self._last = sample
                        self._history.append(sample["gpu_util"])
                except Exception:
                    pass
            time.sleep(1.0)

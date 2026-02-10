from threading import Lock


class FrameIdGenerator:
    def __init__(self, start: int = 1):
        self._lock = Lock()
        self._next_id = start

    def next_id(self) -> int:
        with self._lock:
            frame_id = self._next_id
            self._next_id += 1
        return frame_id

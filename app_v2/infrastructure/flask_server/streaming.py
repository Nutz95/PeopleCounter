from __future__ import annotations

import queue
import threading
from collections.abc import Iterator


# Minimal 1×1 black JPEG served when no frame has been pushed yet.
_PLACEHOLDER_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e"
    b"C  C\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
    b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4"
    b"\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00"
    b"\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142"
    b"\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18"
    b"\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85"
    b"\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3"
    b"\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba"
    b"\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8"
    b"\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4"
    b"\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb"
    b"\xd4P\x00\x00\x00\x1f\xff\xd9"
)


class SseHub:
    """Fan-out hub for Server-Sent Events clients."""

    def __init__(self) -> None:
        self._clients: list[queue.SimpleQueue[str]] = []
        self._lock = threading.Lock()

    def add_client(self) -> queue.SimpleQueue[str]:
        q: queue.SimpleQueue[str] = queue.SimpleQueue()
        with self._lock:
            self._clients.append(q)
        return q

    def remove_client(self, q: queue.SimpleQueue[str]) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def publish(self, data: str) -> None:
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(data)
            except Exception:
                pass


class MjpegFrameStore:
    """Latest-frame MJPEG store with sequence-based wakeup.

    Uses a monotonic sequence counter + condition variable instead of a shared
    Event.clear() pattern. This avoids missed notifications and keeps delivery
    stable when producers/consumers run at slightly different rates.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: bytes = _PLACEHOLDER_JPEG
        self._seq: int = 0

    def push(self, jpeg_bytes: bytes) -> None:
        with self._cond:
            self._frame = jpeg_bytes
            self._seq += 1
            self._cond.notify_all()

    def stream_iter(self) -> Iterator[bytes]:
        last_seq = -1
        while True:
            with self._cond:
                if self._seq == last_seq:
                    # Heartbeat: if no new frame arrived in 1 s, resend latest frame.
                    self._cond.wait(timeout=1.0)
                frame = self._frame
                last_seq = self._seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

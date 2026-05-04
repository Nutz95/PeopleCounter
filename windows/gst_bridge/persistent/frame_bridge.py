from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .frame_pacer import FramePacer
from .source_decoder import SourceDecoder
from .stream_encoder import StreamEncoder


class PersistentFrameBridge:
    def __init__(
        self,
        encoder: StreamEncoder,
        fps: int,
        black_frame: bytes,
        on_metrics: Callable[[float, float, float], None] | None = None,
    ) -> None:
        self._encoder = encoder
        self._fps = fps
        self._black_frame = black_frame
        self._on_metrics = on_metrics
        self._decoder: SourceDecoder | None = None
        self._decoder_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_since_report = 0
        self._freeze_since_report = 0
        self._report_started_at = time.monotonic()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="persistent-frame-bridge")
        self._thread.start()

    def switch(self, new_decoder: SourceDecoder) -> None:
        with self._decoder_lock:
            old_decoder = self._decoder
            self._decoder = new_decoder
        if old_decoder is not None:
            threading.Thread(target=old_decoder.stop, daemon=True, name="decoder-stop").start()

    def _loop(self) -> None:
        pacer = FramePacer(self._fps)
        while not self._stop.is_set():
            with self._decoder_lock:
                decoder = self._decoder
            if decoder is None:
                frame = self._black_frame
                freeze = False
            else:
                frame, freeze = decoder.get()
            if not self._encoder.submit(frame):
                self._stop.wait(0.1)
                pacer.reset()
                continue
            self._frames_since_report += 1
            if freeze:
                self._freeze_since_report += 1
            now = time.monotonic()
            elapsed = now - self._report_started_at
            if self._on_metrics is not None and elapsed >= 1.0:
                fps = self._frames_since_report / elapsed
                freeze_fps = self._freeze_since_report / elapsed
                self._on_metrics(fps, fps, freeze_fps)
                self._frames_since_report = 0
                self._freeze_since_report = 0
                self._report_started_at = now
            pacer.wait_next(self._stop)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._decoder_lock:
            decoder = self._decoder
            self._decoder = None
        if decoder is not None:
            decoder.stop()

import threading
from queue import Queue

class AsyncPushWorker:
    def __init__(self):
        self._queue = Queue(maxsize=8)  # Limite pour éviter l'explosion mémoire
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def push(self, jpeg_bytes, push_frame):
        self._queue.put((jpeg_bytes, push_frame))

    def _worker(self):
        while True:
            jpeg_bytes, push_frame = self._queue.get()
            try:
                push_frame(jpeg_bytes)
            except Exception as exc:
                import sys
                print(f"[AsyncPushWorker] push_frame failed: {exc}", file=sys.stderr, flush=True)
            self._queue.task_done()

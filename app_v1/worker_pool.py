from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Callable, List, Optional

from frame_tasks import FrameTask, QueueDepth, ResultTask
from logger.filtered_logger import LogChannel, warning


class WorkerPool:
    def __init__(
        self,
        group_name: str,
        worker_fn: Callable[[FrameTask], Optional[ResultTask]],
        queue_max_size: int = 8,
        worker_count: int = 1,
        channel: LogChannel = LogChannel.GLOBAL,
    ) -> None:
        self.group_name = group_name
        self._worker_fn = worker_fn
        self._queue = Queue(maxsize=queue_max_size if queue_max_size > 0 else 0)
        self._result_queue = Queue()
        self._stop_event = Event()
        self._workers: List[Thread] = []
        self._channel = channel
        self._worker_count = max(1, worker_count)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._workers = [
            Thread(target=self._worker_loop, name=f"{self.group_name}-worker-{idx}", daemon=True)
            for idx in range(self._worker_count)
        ]
        for worker in self._workers:
            worker.start()
        self._started = True

    def stop(self, wait: bool = True) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if wait:
            for worker in self._workers:
                worker.join(timeout=1.0)
        self._started = False

    def enqueue(self, task: FrameTask) -> bool:
        if not self._started:
            raise RuntimeError("WorkerPool must be started before enqueuing tasks.")
        try:
            self._queue.put_nowait(task)
            return True
        except Full:
            warning(self._channel, f"{self.group_name} queue is full; dropping frame {task.frame_id}.")
            return False

    def drain_results(self, max_results: Optional[int] = None) -> List[ResultTask]:
        drained: List[ResultTask] = []
        while max_results is None or len(drained) < max_results:
            try:
                drained.append(self._result_queue.get_nowait())
            except Empty:
                break
        return drained

    def queue_depth(self) -> QueueDepth:
        max_size = self._queue.maxsize if self._queue.maxsize > 0 else 0
        return QueueDepth(self.group_name, self._queue.qsize(), max_size)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame_task = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                result = self._worker_fn(frame_task)
                if result is not None:
                    self._result_queue.put(result)
            except Exception as exc:
                warning(self._channel, f"{self.group_name} worker error: {exc}")
            finally:
                self._queue.task_done()

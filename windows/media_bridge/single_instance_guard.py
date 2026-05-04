from __future__ import annotations

import os
import tempfile
from pathlib import Path

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - not available on Windows
    fcntl = None

try:
    import msvcrt  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - not available on POSIX
    msvcrt = None


class SingleInstanceGuard:
    def __init__(self, name: str) -> None:
        self.name = name
        self.lock_path = Path(tempfile.gettempdir()) / f"{name}.lock"
        self._handle = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.lock_path, "a+b")
        try:
            self._lock_handle()
            self._handle.seek(0)
            self._handle.truncate()
            self._handle.write(str(os.getpid()).encode("ascii"))
            self._handle.flush()
        except Exception:
            self.release()
            raise RuntimeError("PeopleCounter Media Bridge is already running. Close the existing instance before starting a new one.")

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            self._unlock_handle()
        finally:
            self._handle.close()
            self._handle = None

    def _lock_handle(self) -> None:
        if msvcrt is not None:
            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        raise RuntimeError("Unsupported platform for single-instance guard")

    def _unlock_handle(self) -> None:
        if msvcrt is not None:
            self._handle.seek(0)
            try:
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            return
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

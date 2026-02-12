from __future__ import annotations

from typing import Any


class SimpleStreamPool:
    """Minimal stream pool that keeps a handle per named stream key."""

    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}

    def acquire(self, key: str) -> Any:
        """Return the handle owned by key, creating a lightweight placeholder."""
        return self._handles.setdefault(key, f"stream:{key}")

    def release(self, key: str) -> None:
        """No-op release so the context stays compatible with TensorRTExecutionContext."""
        return None

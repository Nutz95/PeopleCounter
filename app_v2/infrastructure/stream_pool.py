from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class SimpleStreamPool:
    """Minimal stream pool that keeps a handle per named stream key."""

    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}

    def acquire(self, key: str) -> Any:
        """Return the handle owned by key, creating a lightweight placeholder."""
        if key in self._handles:
            return self._handles[key]
        if torch is not None and torch.cuda.is_available():
            self._handles[key] = torch.cuda.Stream()
        else:
            self._handles[key] = f"stream:{key}"
        return self._handles[key]

    def release(self, key: str) -> None:
        """No-op release so the context stays compatible with TensorRTExecutionContext."""
        return None

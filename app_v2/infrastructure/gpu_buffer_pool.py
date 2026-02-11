from __future__ import annotations

from typing import Any


class GPUBufferPool:
    """Pre-allocates GPU slices so multiple models can reuse data without extra copies."""

    def __init__(self) -> None:
        self._buffers: dict[str, Any] = {}

    def reserve(self, key: str, size: int) -> Any:
        """Reserve or reuse a buffer for the given key."""
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = f"gpu-buffer-{key}"  # placeholder for actual device memory
            self._buffers[key] = buffer
        return buffer

    def release(self, key: str) -> None:
        """Release ownership of the buffer so another model can reuse it."""
        self._buffers.pop(key, None)

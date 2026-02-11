from __future__ import annotations

from typing import Dict


class CudaStreamPool:
    """Manages dedicated CUDA streams for transfer, YOLO, and density inferences."""

    def __init__(self, stream_map: Dict[str, int]) -> None:
        self.stream_map = stream_map
        self.active_streams: Dict[str, int] = {}

    def acquire(self, key: str) -> int:
        """Return the stream handle assigned to key (e.g., transfer, yolo, density)."""
        if key not in self.stream_map:
            raise KeyError(f"Unknown CUDA stream key: {key}")
        stream_id = self.stream_map[key]
        self.active_streams[key] = stream_id
        return stream_id

    def release(self, key: str) -> None:
        """Drop ownership of a stream once the work is complete."""
        self.active_streams.pop(key, None)

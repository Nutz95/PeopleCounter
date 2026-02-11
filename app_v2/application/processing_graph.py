from __future__ import annotations

from typing import Any


class ProcessingGraph:
    """Describes dependencies between frame capture, preprocessing, inference, and fusion."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}

    def register(self, name: str, metadata: dict[str, Any]) -> None:
        """Add a node to the graph for observability."""
        self.nodes[name] = metadata

    def trace(self, frame_id: int) -> dict[str, Any]:
        """Return the current graph state for frame_id (useful for metadata publishing)."""
        return {"frame_id": frame_id, "nodes": list(self.nodes.keys())}

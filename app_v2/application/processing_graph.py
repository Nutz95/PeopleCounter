from __future__ import annotations

from typing import Any

from logger.filtered_logger import LogChannel, info as log_info


class ProcessingGraph:
    """Describes dependencies between frame capture, preprocessing, inference, and fusion."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        log_info(LogChannel.GLOBAL, "ProcessingGraph instantiated")

    def register(self, name: str, metadata: dict[str, Any]) -> None:
        """Add a node to the graph for observability."""
        self.nodes[name] = metadata
        log_info(LogChannel.GLOBAL, f"ProcessingGraph registered node {name}")

    def trace(self, frame_id: int) -> dict[str, Any]:
        """Return the current graph state for frame_id (useful for metadata publishing)."""
        return {"frame_id": frame_id, "nodes": list(self.nodes.keys())}

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from app_v2.core.fusion_strategy import FusionStrategy
from app_v2.core.result_publisher import ResultPublisher
from logger.filtered_logger import LogChannel, info as log_info


class ResultAggregator:
    """Collects per-model answers and applies the configured fusion strategy."""

    def __init__(self, fusion_strategy: FusionStrategy, publisher: ResultPublisher) -> None:
        self.fusion_strategy = fusion_strategy
        self.publisher = publisher
        self._buffers: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def collect(self, frame_id: int, payload: dict[str, Any]) -> None:
        """Store partial payloads and publish when fusion strategy says so."""
        self._buffers[frame_id].append(payload)
        collected = self._buffers[frame_id]
        if self.fusion_strategy.should_publish(frame_id, [frame_id for _ in collected]):
            merged = self.fusion_strategy.merge(collected)
            payload = merged if isinstance(merged, Sequence) else [merged]
            self.publisher.publish(frame_id, payload)
            log_info(
                LogChannel.GLOBAL,
                f"Published frame {frame_id} with {len(payload)} payloads via {self.fusion_strategy.strategy_type.value}",
            )
            self._buffers.pop(frame_id, None)
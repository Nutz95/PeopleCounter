"""SimpleFusionStrategy — wait for all models before publishing."""
from __future__ import annotations

from typing import Any, Sequence

from app_v2.enums import FusionStrategyType
from app_v2.core.strategies.base import FusionStrategy


class SimpleFusionStrategy(FusionStrategy):
    """Publishes once all expected models have reported for a given frame.

    ``expected_count`` is set to ``len(models)`` by the orchestrator so that
    all model outputs arrive in a single SSE event rather than one per model.
    """

    def __init__(
        self,
        strategy_type: FusionStrategyType = FusionStrategyType.ASYNC_OVERLAY,
        expected_count: int = 1,
    ) -> None:
        super().__init__(strategy_type)
        self.expected_count = expected_count

    def should_publish(self, frame_id: int, collected_ids: Sequence[int]) -> bool:
        return len(collected_ids) >= self.expected_count

    def merge(self, payloads: Sequence[Any]) -> Sequence[Any]:
        return list(payloads)

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from app_v2.enums import FusionStrategyType


class FusionStrategy(ABC):
    """Decides when independent model outputs should be published."""

    def __init__(self, strategy_type: FusionStrategyType) -> None:
        self.strategy_type = strategy_type

    @abstractmethod
    def should_publish(self, frame_id: int, collected_ids: Sequence[int]) -> bool:
        """Return True when the collected data for frame_id is ready to go."""

    @abstractmethod
    def merge(self, payloads: Sequence[Any]) -> Any:
        """Merge the raw payloads from each model before delivery."""


class SimpleFusionStrategy(FusionStrategy):
    """Default fusion strategy that publishes once all expected models have reported.

    ``expected_count`` controls how many ``collect()`` calls are needed before
    the aggregator publishes the merged frame result.  Set it to
    ``len(models)`` in the orchestrator so that all model outputs arrive in a
    single SSE event rather than one event per model.
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

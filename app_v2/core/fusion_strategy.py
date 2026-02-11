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
    """Default fusion strategy that publishes eagerly."""

    def __init__(self, strategy_type: FusionStrategyType = FusionStrategyType.ASYNC_OVERLAY) -> None:
        super().__init__(strategy_type)

    def should_publish(self, frame_id: int, collected_ids: Sequence[int]) -> bool:
        return True

    def merge(self, payloads: Sequence[Any]) -> Sequence[Any]:
        return list(payloads)

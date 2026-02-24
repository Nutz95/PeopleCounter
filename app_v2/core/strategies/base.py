"""Base contract for all fusion strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from app_v2.enums import FusionStrategyType


class FusionStrategy(ABC):
    """Decides when and how independent model outputs should be published."""

    def __init__(self, strategy_type: FusionStrategyType) -> None:
        self.strategy_type = strategy_type

    @abstractmethod
    def should_publish(self, frame_id: int, collected_ids: Sequence[int]) -> bool:
        """Return True when the collected data for *frame_id* is ready to ship."""

    @abstractmethod
    def merge(self, payloads: Sequence[Any]) -> Any:
        """Merge raw payloads from each model before delivery."""

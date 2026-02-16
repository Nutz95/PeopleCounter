from __future__ import annotations

from collections.abc import Callable
from threading import Condition
from typing import Protocol


class PoolSaturationPolicy(Protocol):
    """Defines how `GpuTensorPool` behaves when a key reaches capacity."""

    def wait_for_availability(
        self,
        *,
        cond: Condition,
        timeout_s: float | None,
        on_wait_sample: Callable[[float], None],
    ) -> bool:
        """Return True when caller should retry, False to abort acquisition."""

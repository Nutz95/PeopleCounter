from __future__ import annotations

from collections.abc import Callable
from threading import Condition
from time import monotonic


class BlockingSaturationPolicy:
    """Simple backpressure policy that blocks until a tensor becomes available."""

    def wait_for_availability(
        self,
        *,
        cond: Condition,
        timeout_s: float | None,
        on_wait_sample: Callable[[float], None],
    ) -> bool:
        started = monotonic()
        cond.wait(timeout=timeout_s)
        waited_ms = max(0.0, (monotonic() - started) * 1000.0)
        on_wait_sample(waited_ms)
        if timeout_s is not None and waited_ms >= timeout_s * 1000.0:
            return False
        return True

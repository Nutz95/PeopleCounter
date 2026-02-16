from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorPoolStats:
    hits: int
    misses: int
    allocations: int
    in_use: int
    available: int
    waits: int
    wait_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "allocations": self.allocations,
            "in_use": self.in_use,
            "available": self.available,
            "waits": self.waits,
            "wait_ms": self.wait_ms,
        }

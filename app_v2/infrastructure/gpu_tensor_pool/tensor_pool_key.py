from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorPoolKey:
    shape: tuple[int, ...]
    dtype: str
    memory_format: str
    stream: int
    device: str

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectShowMode:
    width: int
    height: int
    fps: int
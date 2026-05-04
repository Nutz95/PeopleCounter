from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DirectShowDevice:
    friendly_name: str
    device_type: str
    alternatives: list[str] = field(default_factory=list)

    @property
    def input_name(self) -> str:
        return self.friendly_name
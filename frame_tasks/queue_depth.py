from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class QueueDepth:
    group_name: str
    size: int
    max_size: int

    @property
    def is_saturated(self) -> bool:
        return self.size >= self.max_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group": self.group_name,
            "depth": self.size,
            "max_depth": self.max_size,
            "saturated": self.is_saturated,
        }

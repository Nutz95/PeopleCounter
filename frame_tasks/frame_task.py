from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class FrameTask:
    frame_id: int
    capture_ts: float
    frame: np.ndarray
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **items: Any) -> FrameTask:
        self.metadata.update(items)
        return self

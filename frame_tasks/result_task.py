from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ResultTask:
    frame_id: int
    capture_ts: float
    processed_ts: float
    yolo_count: Optional[int]
    frame_with_bbox: Optional[np.ndarray]
    density_result: Optional[Dict[str, Any]] = None
    heatmap_payload: Optional[Dict[str, Any]] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def add_stat(self, key: str, value: Any) -> ResultTask:
        self.stats[key] = value
        return self

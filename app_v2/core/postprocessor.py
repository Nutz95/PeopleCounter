from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Postprocessor(ABC):
    """Post-process inference outputs into consumable metadata for the publisher."""

    @abstractmethod
    def process(self, frame_id: int, outputs: dict[str, Any]) -> dict[str, Any]:
        """Return the decoded boxes, masks, or density maps."""

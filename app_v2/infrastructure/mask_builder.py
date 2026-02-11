from __future__ import annotations

from typing import Any


class MaskBuilder:
    """Responsible for building mask metadata that front-end overlays consume."""

    def __init__(self, target_resolution: tuple[int, int]) -> None:
        self.target_resolution = target_resolution

    def build(self, frame_id: int, mask_coeffs: Any) -> dict[str, Any]:
        """Generate metadata describing where to draw the mask for frame_id."""
        return {"frame_id": frame_id, "mask": mask_coeffs}

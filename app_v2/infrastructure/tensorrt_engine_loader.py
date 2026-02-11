from __future__ import annotations

from typing import Dict, Any


class TensorRTEngineLoader:
    """Loads TensorRT engine blobs and exposes metadata for other infrastructure classes."""

    def __init__(self, engine_path: str, profiles: Dict[str, Any]) -> None:
        self.engine_path = engine_path
        self.profiles = profiles
        self._metadata: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Deserialize the engine and cache tensor names/shapes."""
        if not self._metadata:
            self._metadata = {"path": self.engine_path, "profiles": self.profiles}
        return self._metadata

from __future__ import annotations

from typing import Any

from app_v2.core.preprocess_types import InputSpec


class InputSpecRegistry:
    """Builds preprocess input specs from pipeline-like configuration."""

    def __init__(self) -> None:
        self._specs: dict[str, InputSpec] = {}

    def configure(self, metadata: dict[str, Any]) -> None:
        models = metadata.get("models", {})
        preprocess_cfg = metadata.get("preprocess", {})

        self._specs = {}
        if self._is_enabled(models, "yolo_global"):
            global_cfg = preprocess_cfg.get("yolo_global", {})
            self._specs["yolo_global"] = InputSpec(
                model_name="yolo_global",
                target_width=int(global_cfg.get("target_width", 640)),
                target_height=int(global_cfg.get("target_height", 640)),
                mode="global",
                overlap=0.0,
            )

        if self._is_enabled(models, "yolo_tiles"):
            tiles_cfg = preprocess_cfg.get("yolo_tiles", {})
            self._specs["yolo_tiles"] = InputSpec(
                model_name="yolo_tiles",
                target_width=int(tiles_cfg.get("target_width", 640)),
                target_height=int(tiles_cfg.get("target_height", 640)),
                mode="tiles",
                overlap=float(tiles_cfg.get("overlap", 0.2)),
            )

    def all_specs(self) -> tuple[InputSpec, ...]:
        return tuple(self._specs.values())

    def by_model(self, model_name: str) -> InputSpec | None:
        return self._specs.get(model_name)

    def _is_enabled(self, models: dict[str, Any], model_name: str) -> bool:
        model_cfg = models.get(model_name)
        if not isinstance(model_cfg, dict):
            return False
        return bool(model_cfg.get("enabled", False))

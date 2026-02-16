from __future__ import annotations

from typing import Any

from app_v2.core.preprocessor_types import InputSpec


class InputSpecRegistry:
    """Builds preprocess input specs from pipeline-like configuration."""

    _REQUIRED_KEYS = (
        "target_width",
        "target_height",
        "mode",
        "overlap",
    )

    def __init__(self) -> None:
        self._specs: dict[str, InputSpec] = {}

    def configure(self, metadata: dict[str, Any]) -> None:
        models = metadata.get("models", {})
        preprocess_cfg = metadata.get("preprocess")
        if preprocess_cfg is None:
            raise ValueError("preprocess configuration is required")
        if not isinstance(preprocess_cfg, dict):
            raise ValueError("preprocess entries must be a mapping")

        self._specs = {}
        for model_name, spec_cfg in preprocess_cfg.items():
            self._validate_entry(model_name, spec_cfg)
            spec = self._build_spec(model_name, spec_cfg)
            if self._is_enabled(models, model_name):
                self._specs[model_name] = spec

    def all_specs(self) -> tuple[InputSpec, ...]:
        return tuple(self._specs.values())

    def by_model(self, model_name: str) -> InputSpec | None:
        return self._specs.get(model_name)

    def _is_enabled(self, models: dict[str, Any], model_name: str) -> bool:
        model_cfg = models.get(model_name)
        if not isinstance(model_cfg, dict):
            return False
        return bool(model_cfg.get("enabled", False))

    def _validate_entry(self, model_name: str, spec_cfg: Any) -> None:
        if not isinstance(spec_cfg, dict):
            raise ValueError(f"preprocess entry '{model_name}' must be a mapping")
        missing = [key for key in self._REQUIRED_KEYS if key not in spec_cfg]
        if missing:
            raise ValueError(
                f"preprocess entry '{model_name}' is missing required keys: {', '.join(missing)}"
            )

    def _build_spec(self, model_name: str, spec_cfg: dict[str, Any]) -> InputSpec:
        target_width = int(spec_cfg["target_width"])
        target_height = int(spec_cfg["target_height"])
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target dimensions must be positive integers")
        overlap = float(spec_cfg["overlap"])
        mode = str(spec_cfg["mode"])
        if not mode:
            raise ValueError("mode cannot be empty")
        return InputSpec(
            model_name=model_name,
            target_width=target_width,
            target_height=target_height,
            mode=mode,
            overlap=overlap,
        )

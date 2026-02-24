from __future__ import annotations

from typing import Any

from app_v2.core.preprocessor_types import InputSpec, PreprocessMode


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
        preprocess_branches = metadata.get("preprocess_branches", {})
        if preprocess_cfg is None:
            raise ValueError("preprocess configuration is required")
        if not isinstance(preprocess_cfg, dict):
            raise ValueError("preprocess entries must be a mapping")
        if preprocess_branches is not None and not isinstance(preprocess_branches, dict):
            raise ValueError("preprocess_branches must be a mapping when provided")

        self._specs = {}
        for model_name, spec_cfg in preprocess_cfg.items():
            self._validate_entry(model_name, spec_cfg)
            spec = self._build_spec(model_name, spec_cfg)
            if self._is_enabled(models, model_name) and self._is_branch_enabled(preprocess_branches, model_name):
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

    def _is_branch_enabled(self, preprocess_branches: dict[str, Any], model_name: str) -> bool:
        branch_key = self._branch_key(model_name)
        if branch_key is None:
            return True
        return bool(preprocess_branches.get(branch_key, True))

    @staticmethod
    def _branch_key(model_name: str) -> str | None:
        normalized = str(model_name)
        if normalized == "yolo_global":
            return "yolo_global_preprocess"
        if normalized.startswith("yolo_tiles"):
            return "yolo_tiles_preprocess"
        if normalized.startswith("density") or normalized.startswith("lwcc"):
            return "density_preprocess"
        return None

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
        mode = self._parse_mode(spec_cfg["mode"])
        source_tile_width  = int(spec_cfg.get("source_tile_width",  0))
        source_tile_height = int(spec_cfg.get("source_tile_height", 0))
        return InputSpec(
            model_name=model_name,
            target_width=target_width,
            target_height=target_height,
            mode=mode,
            overlap=overlap,
            source_tile_width=source_tile_width,
            source_tile_height=source_tile_height,
        )

    @staticmethod
    def _parse_mode(value: Any) -> PreprocessMode:
        mode_value = str(value).strip().lower()
        if not mode_value:
            raise ValueError("mode cannot be empty")
        try:
            return PreprocessMode(mode_value)
        except ValueError as exc:
            raise ValueError(f"Unsupported preprocess mode: {value}") from exc

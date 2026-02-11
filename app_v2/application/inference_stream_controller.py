from __future__ import annotations

from typing import Any, Dict, Iterable

from logger.filtered_logger import LogChannel, info as log_info


class InferenceStreamController:
    """Controls which inference streams are active and exposes engine paths."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._models = config.get("models", {})
        self._streams = config.get("inference_streams", {})
        self._enabled_models: Dict[str, Dict[str, Any]] = {
            name: data for name, data in self._models.items() if data.get("enabled", False)
        }
        log_info(LogChannel.GLOBAL, f"InferenceStreamController initialized with enabled models: {list(self._enabled_models.keys())}")
        log_info(LogChannel.GLOBAL, f"Stream availability: {self._streams}")

    def is_model_enabled(self, model_name: str) -> bool:
        """Return True when the named model (yolo_tiles, density, etc.) is configured for inference."""
        return model_name in self._enabled_models

    def get_engine_path(self, model_name: str) -> str | None:
        """Return the configured TensorRT engine path for the model."""
        model_cfg = self._enabled_models.get(model_name)
        if model_cfg is None:
            return None
        return model_cfg.get("engine")

    def enabled_models(self) -> Iterable[str]:
        """Yield the names of the models that will run."""
        return self._enabled_models.keys()

    def is_stream_available(self, stream_name: str) -> bool:
        """Return whether the requested inference stream is allowed to execute."""
        return bool(self._streams.get(stream_name, False))
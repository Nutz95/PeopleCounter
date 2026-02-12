from __future__ import annotations

from typing import Any

from app_v2.application.inference_stream_controller import InferenceStreamController
from app_v2.core.inference_model import InferenceModel
from app_v2.infrastructure.density_trt import DensityTRT
from app_v2.infrastructure.tensorrt_engine_loader import TensorRTEngineLoader
from app_v2.infrastructure.tensorrt_execution_context import TensorRTExecutionContext
from app_v2.infrastructure.yolo_global_trt import YoloGlobalTRT
from app_v2.infrastructure.yolo_tiling_trt import YoloTilingTRT
from app_v2.infrastructure.stream_pool import SimpleStreamPool
from logger.filtered_logger import LogChannel, warning as log_warning



_MODEL_REGISTRY: dict[str, tuple[type[InferenceModel], str]] = {
    "yolo_global": (YoloGlobalTRT, "yolo"),
    "yolo_tiles": (YoloTilingTRT, "yolo"),
    "density": (DensityTRT, "density"),
}


class ModelBuilder:
    """Constructs the enabled `InferenceModel` instances described in the pipeline config."""

    def __init__(
        self,
        config: dict[str, Any],
        inference_controller: InferenceStreamController,
        stream_pool: SimpleStreamPool,
    ) -> None:
        self._config = config
        self._controller = inference_controller
        self._stream_pool = stream_pool

    def build_models(self) -> list[InferenceModel]:
        models: list[InferenceModel] = []
        stream_ids = self._config.get("streams", {})
        for model_name in self._controller.enabled_models():
            engine_path = self._controller.get_engine_path(model_name)
            if engine_path is None:
                log_warning(LogChannel.GLOBAL, f"Engine path missing for {model_name}")
                continue
            registry_entry = _MODEL_REGISTRY.get(model_name)
            if registry_entry is None:
                log_warning(LogChannel.GLOBAL, f"No inference model registered for {model_name}")
                continue
            model_cls, stream_key = registry_entry
            stream_id = stream_ids.get(stream_key, 0)
            loader = TensorRTEngineLoader(engine_path, profiles={})
            context = TensorRTExecutionContext(loader, self._stream_pool)
            models.append(model_cls(context, stream_id))
        return models

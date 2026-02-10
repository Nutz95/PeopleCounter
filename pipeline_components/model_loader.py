import os
from typing import Any, Dict, Optional


class ModelLoader:
    def __init__(self, model_name: str, backend: str, configs: Optional[Dict[str, str]] = None) -> None:
        self.model_name = model_name
        self.backend = backend
        self.configs = configs or {}
        self.loaded_model: Optional[Any] = None

    def load(self) -> Any:
        if self.backend == 'trt':
            return self._load_trt()
        if self.backend == 'openvino':
            return self._load_openvino()
        return self._load_torch()

    def _load_trt(self) -> Any:
        engine_path = self.configs.get('LWCC_TRT_ENGINE') or 'dm_count.engine'
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        self.loaded_model = f"TensorRT:{engine_path}"
        return self.loaded_model

    def _load_openvino(self) -> Any:
        model_dir = self.configs.get('YOLO_OPENVINO_DIR') or '/opt/intel/openvino'
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"OpenVINO directory missing: {model_dir}")
        self.loaded_model = f"OpenVINO:{model_dir}/{self.model_name}"
        return self.loaded_model

    def _load_torch(self) -> Any:
        weights = self.configs.get('YOLO_MODEL') or self.model_name
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Torch weights not found: {weights}")
        self.loaded_model = f"Torch:{weights}"
        return self.loaded_model

    def get(self) -> Any:
        if self.loaded_model is None:
            return self.load()
        return self.loaded_model

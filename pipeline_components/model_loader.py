import importlib
import os
from typing import Any, Dict, Optional

from logger.filtered_logger import LogChannel, warning


class ModelLoader:
    def __init__(
        self,
        model_name: str = 'yolo26s-seg',
        backend: str = 'torch',
        yolo_conf: float = 0.65,
        density_threshold: int = 15,
        configs: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.yolo_conf = float(yolo_conf)
        self.density_threshold = int(density_threshold)
        self.configs = configs or {}

    def load(self, settings: Optional[Dict[str, str]] = None):
        from yolo_people_counter import YoloPeopleCounter
        from people_counter_processor import PeopleCounterProcessor
        import torch

        if settings is None:
            settings = {}

        yolo_model = settings.get('YOLO_MODEL') or self.model_name
        yolo_backend = (settings.get('YOLO_BACKEND') or self.backend).lower()
        yolo_device = settings.get('YOLO_DEVICE') or ('cuda' if getattr(torch, 'cuda', None) and torch.cuda.is_available() else 'cpu')
        density_backend = (settings.get('LWCC_BACKEND') or 'torch').lower()
        openvino_device = settings.get('OPENVINO_DEVICE', 'GPU')
        density_threshold = int(settings.get('DENSITY_THRESHOLD', self.density_threshold))

        def _module_available(module_name: str) -> bool:
            try:
                return importlib.util.find_spec(module_name) is not None
            except ImportError:
                return False

        if yolo_backend in ('tensorrt', 'tensorrt_native') and not _module_available('tensorrt'):
            warning(LogChannel.GLOBAL, "YOLO TensorRT backend requested but tensorrt package is missing; falling back to torch.")
            yolo_backend = 'torch'

        if yolo_backend == 'tensorrt_native' and not _module_available('cuda.bindings'):
            warning(LogChannel.GLOBAL, "YOLO TensorRT native backend requires cuda.bindings but it's missing; falling back to torch.")
            yolo_backend = 'torch'

        if density_backend == 'tensorrt' and not _module_available('cuda.bindings'):
            warning(LogChannel.GLOBAL, "Density TensorRT backend requested but cuda bindings are missing; falling back to torch.")
            density_backend = 'torch'

        yolo_counter = YoloPeopleCounter(
            model_name=yolo_model,
            device=yolo_device,
            confidence_threshold=self.yolo_conf,
            backend=yolo_backend,
        )
        processor = PeopleCounterProcessor(
            model_name=settings.get('DENSITY_MODEL', 'CSRNet'),
            backend=density_backend,
            openvino_device=openvino_device,
            density_threshold=density_threshold,
        )

        metadata = {
            'device': getattr(yolo_counter, 'device', yolo_device),
            'processor_device': getattr(processor, 'last_device', openvino_device),
        }
        return yolo_counter, processor, metadata

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

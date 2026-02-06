import glob
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Dict, Optional, Tuple

from people_counter_processor import PeopleCounterProcessor
from yolo_people_counter import YoloPeopleCounter
from yolo_seg_people_counter import YoloSegPeopleCounter


class ProfileManager:
    def __init__(self, configs_dir: Optional[str] = None) -> None:
        self.configs_dir = configs_dir or os.path.join(os.getcwd(), 'scripts', 'configs')
        self.default_settings = self._gather_base_settings()
        self.profile_settings = dict(self.default_settings)
        self.available_profiles = self._discover_profiles()
        self.available_profiles = self._filter_profiles_by_os(self.available_profiles)
        env_profile = os.environ.get('ACTIVE_PROFILE', '').strip()
        self.active_profile_name = env_profile or self._determine_default_profile()
        self.pending_profile_name = self.active_profile_name
        settings, _ = self._load_profile_settings(self.active_profile_name)
        self.pending_profile_settings = dict(settings)
        self.pending_profile_reload = True
        self._last_applied_profile = ''

    def _gather_base_settings(self) -> Dict[str, str]:
        return {
            'YOLO_BACKEND': os.environ.get('YOLO_BACKEND', 'torch'),
            'YOLO_DEVICE': os.environ.get('YOLO_DEVICE'),
            'YOLO_MODEL': os.environ.get('YOLO_MODEL', 'yolo26s-seg'),
            'YOLO_SEG': os.environ.get('YOLO_SEG', '0'),
            'YOLO_TILING': os.environ.get('YOLO_TILING', '1'),
            'DENSITY_TILING': os.environ.get('DENSITY_TILING', '1'),
            'DEBUG_TILING': os.environ.get('DEBUG_TILING', '0'),
            'EXTREME_DEBUG': os.environ.get('EXTREME_DEBUG', '0'),
            'YOLO_GLOBAL_TILE_ONLY': os.environ.get('YOLO_GLOBAL_TILE_ONLY', '0'),
            'DISABLE_DENSITY_INFERENCE': os.environ.get('DISABLE_DENSITY_INFERENCE', '0'),
            'LWCC_BACKEND': os.environ.get('LWCC_BACKEND', 'torch'),
            'LWCC_TRT_ENGINE': os.environ.get('LWCC_TRT_ENGINE', 'dm_count.engine'),
            'OPENVINO_DEVICE': os.environ.get('OPENVINO_DEVICE', 'GPU'),
            'YOLO_OPENVINO_DIR': os.environ.get('YOLO_OPENVINO_DIR') or '',
        }

    def _discover_profiles(self) -> Tuple[str, ...]:
        profiles = []
        if os.path.isdir(self.configs_dir):
            for path in sorted(glob.glob(os.path.join(self.configs_dir, '*.env'))):
                name = os.path.splitext(os.path.basename(path))[0]
                if name not in profiles:
                    profiles.append(name)
        return tuple(profiles)

    def _filter_profiles_by_os(self, profiles: Tuple[str, ...]) -> Tuple[str, ...]:
        if platform.system().lower() != 'linux':
            return profiles
        filtered = []
        for profile in profiles:
            path = os.path.join(self.configs_dir, f"{profile}.env")
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    content = fh.read().lower()
            except OSError:
                continue
            if 'openvino' in content:
                continue
            filtered.append(profile)
        return tuple(filtered) if filtered else profiles

    def _determine_default_profile(self) -> str:
        if not self.available_profiles:
            return ''
        preferred = 'rtx_extreme'
        for profile in self.available_profiles:
            if profile.lower() == preferred:
                return profile
        return self.available_profiles[0]

    def _load_profile_settings(self, profile_name: str) -> Tuple[Dict[str, str], bool]:
        settings = dict(self.default_settings)
        if not profile_name:
            return settings, True
        profile_path = os.path.join(self.configs_dir, f"{profile_name}.env")
        if not os.path.isfile(profile_path):
            print(f"[WARN] Profile '{profile_name}' not found, falling back to defaults.")
            return settings, False
        try:
            with open(profile_path, 'r', encoding='utf-8') as fh:
                for raw in fh:
                    line = raw.split('#', 1)[0].strip()
                    if not line:
                        continue
                    if line.lower().startswith('export '):
                        line = line[7:].strip()
                    if '=' not in line:
                        continue
                    key, val = line.split('=', 1)
                    key = key.strip()
                    value = val.strip().strip('"').strip("'")
                    if key:
                        settings[key] = value
        except Exception as exc:
            print(f"[WARN] Unable to read profile '{profile_name}': {exc}")
            return settings, False
        return settings, True

    def apply_profile(self, profile_name: Optional[str] = None) -> bool:
        if profile_name is not None:
            target = profile_name
        else:
            target = self.pending_profile_name or self.active_profile_name or ''
        settings, found = self._load_profile_settings(target)
        self.pending_profile_name = target
        self.pending_profile_settings = dict(settings)
        self.pending_profile_reload = True
        return found or not bool(target)

    def ensure_loaded(self) -> bool:
        if not self.pending_profile_reload:
            return False
        self.profile_settings = dict(self.pending_profile_settings or self.default_settings)
        self.active_profile_name = self.pending_profile_name or ''
        profile_changed = self.active_profile_name != self._last_applied_profile
        self.pending_profile_reload = False
        self._last_applied_profile = self.active_profile_name
        return profile_changed


class ModelLoader:
    def __init__(self, yolo_conf: float, density_threshold: int):
        self.yolo_conf = yolo_conf
        self.density_threshold = density_threshold

    def _resolve_model_path(self, model_name: str) -> str:
        if os.path.isabs(model_name):
            return model_name
        models_root = os.path.join(os.getcwd(), 'models')
        pt_dir = os.path.join(models_root, 'pt')
        candidate = os.path.join(pt_dir, model_name if model_name.endswith('.pt') else f"{model_name}.pt")
        return candidate if os.path.exists(candidate) else model_name

    def _prepare_tensorrt(self, backend: str, model_name: str) -> Tuple[str, str]:
        if backend != 'tensorrt_native':
            return backend, model_name
        models_root = os.path.join(os.getcwd(), 'models')
        tensorrt_dir = os.path.join(models_root, 'tensorrt')
        base_model = model_name
        if not base_model.endswith('.engine'):
            base_model = f"{base_model}.engine"
        if os.path.isabs(base_model):
            engine_path = base_model
        else:
            normalized = os.path.normpath(base_model)
            candidate = os.path.abspath(os.path.join(os.getcwd(), normalized))
            if candidate.startswith(os.path.abspath(models_root)):
                engine_path = candidate
            else:
                engine_path = os.path.join(tensorrt_dir, normalized)
        if not os.path.isfile(engine_path):
            base_name = os.path.basename(base_model).replace('.engine', '')
            onnx_candidate = os.path.join(models_root, 'onnx', f"{base_name}.onnx")
            if os.path.isfile(onnx_candidate):
                print(f"🔁 Converting ONNX -> TensorRT using convert_onnx_to_trt.py: {onnx_candidate}")
                try:
                    script_dir = os.path.abspath(os.path.dirname(__file__))
                    converter = os.path.join(script_dir, 'convert_onnx_to_trt.py')
                    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
                    subprocess.run(
                        [sys.executable, converter, onnx_candidate, engine_path, '32'],
                        check=True,
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"❌ Conversion ONNX -> TensorRT failed: {exc}")
            if os.path.isfile(engine_path):
                print(f"✅ Engine ready at {engine_path}")
                return backend, engine_path
            print(f"⚠️ Engine TensorRT not found at {engine_path}, using fallback weights.")
            return 'torch', model_name
        return backend, engine_path

    def _build_yolo_counter(self, settings: Dict[str, str]) -> Tuple[Optional[object], Dict[str, str]]:
        backend = settings.get('YOLO_BACKEND', 'torch').lower()
        yolo_model_input = settings.get('YOLO_MODEL', 'yolo26s-seg')
        yolo_seg = (settings.get('YOLO_SEG', '0') == '1') or ('-seg' in yolo_model_input.lower())
        backend, yolo_model = self._prepare_tensorrt(backend, yolo_model_input)
        if backend in ('openvino', 'openvino_native'):
            ov_dir = settings.get('YOLO_OPENVINO_DIR')
            if ov_dir and os.path.isdir(ov_dir):
                yolo_model = ov_dir
            else:
                candidate_dir = f"{yolo_model}_openvino_model"
                if os.path.isdir(candidate_dir):
                    yolo_model = candidate_dir
        model_candidate = self._resolve_model_path(yolo_model)
        yolo_device = settings.get('YOLO_DEVICE')
        if not yolo_device:
            if backend == 'openvino_native':
                yolo_device = 'GPU'
            elif backend == 'tensorrt_native':
                yolo_device = 'cuda'
            else:
                yolo_device = 'cpu'
        try:
            if yolo_seg:
                counter = YoloSegPeopleCounter(
                    model_path=model_candidate,
                    confidence_threshold=self.yolo_conf,
                    backend=backend,
                    device=yolo_device,
                )
                metadata = {
                    'backend': backend,
                    'device': yolo_device,
                    'segmentation': True,
                    'model': model_candidate,
                }
            else:
                counter = YoloPeopleCounter(
                    model_name=model_candidate,
                    device=yolo_device,
                    confidence_threshold=self.yolo_conf,
                    backend=backend,
                )
                metadata = {
                    'backend': backend,
                    'device': yolo_device,
                    'segmentation': False,
                    'model': model_candidate,
                }
            return counter, metadata
        except Exception as exc:
            print(
                f"[WARN] Impossible de charger YOLO ({backend}): {exc}, fallback vers torch+CPU."
            )
            fallback_counter = YoloPeopleCounter(
                model_name=model_candidate,
                device='cpu',
                confidence_threshold=self.yolo_conf,
                backend='torch',
            )
            fallback_metadata = {
                'backend': 'torch',
                'device': 'cpu',
                'segmentation': False,
                'model': model_candidate,
            }
            return fallback_counter, fallback_metadata

    def load(self, settings: Dict[str, str]) -> Tuple[object, PeopleCounterProcessor, Dict[str, str]]:
        processor = PeopleCounterProcessor(
            model_name="DM-Count",
            model_weights="QNRF",
            backend=settings.get('LWCC_BACKEND', 'torch'),
            openvino_device=settings.get('OPENVINO_DEVICE', 'GPU'),
            density_threshold=self.density_threshold,
        )
        yolo_counter, metadata = self._build_yolo_counter(settings)
        for key, val in settings.items():
            if val is not None:
                os.environ[key] = str(val)
        metadata['processor_device'] = getattr(processor, 'last_device', 'GPU')
        return yolo_counter, processor, metadata


class MetricsCollector:
    def __init__(self, history_len: int = 600) -> None:
        self.history_len = history_len
        self.history_data = []

    def append(self, yolo_count: float, density_count: float, average: float) -> None:
        entry = {
            'timestamp': time.time(),
            'yolo': float(yolo_count),
            'density': float(density_count),
            'average': float(average),
        }
        self.history_data.append(entry)
        if len(self.history_data) > self.history_len:
            self.history_data.pop(0)

    def snapshot(self):
        return list(self.history_data[-32:])

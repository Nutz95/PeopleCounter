import glob
import os
import platform
from typing import Dict, Optional, Tuple


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

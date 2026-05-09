from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app_v2.config import load_model_inference_config
from app_v2.enums import FusionStrategyType
from app_v2.infrastructure.flask_server.mode_registry import detect_mode_from_config


class RuntimeState:
    """Thread-safe runtime configuration and pending UI requests."""

    def __init__(self, initial_config: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        config = initial_config or {}

        self._active_mode: str = detect_mode_from_config(config)
        self._pending_mode: str | None = None
        self._available_modes: list[str] = self.compute_available_modes(config)

        self._active_sync_mode: str = self._sync_mode_from_strategy(
            str(config.get("fusion_strategy", FusionStrategyType.RAW_STREAM_WITH_METADATA.value))
        )
        self._pending_sync_mode: str | None = None
        self._sync_mode_labels: dict[str, str] = {
            "async": "Async (realtime video)",
            "sync": "Sync (video + inference aligned)",
        }

        density_config = config.get("density") or {}
        self._density_threshold: float = float(density_config.get("min_peak_weight", 0.05))
        self._pending_density_threshold: float | None = None

        model_inference_config = load_model_inference_config()
        crowd_global_config = model_inference_config.get("crowd_global") or {}
        crowd_tiles_config = model_inference_config.get("crowd_tiles") or {}
        self._crowd_confidence_by_mode: dict[str, float] = {
            "crowd_global": float(crowd_global_config.get("confidence_threshold", 0.25)),
            "crowd_tiles": float(crowd_tiles_config.get("confidence_threshold", 0.5)),
        }
        self._crowd_confidence: float = self._crowd_confidence_by_mode.get(self._active_mode, 0.25)
        self._pending_crowd_confidence: float | None = None

        video_stream_config = config.get("video_stream") or {}
        self._active_video_backend: str = str(video_stream_config.get("backend", "auto")).strip().lower()
        self._pending_video_backend: str | None = None

    def config_snapshot(self) -> dict[str, Any]:
        with self._lock:
            mode = self._active_mode
            return {
                "mode": mode,
                "pending_mode": self._pending_mode,
                "available_modes": list(self._available_modes),
                "density_threshold": self._density_threshold,
                "crowd_confidence": self._crowd_confidence_by_mode.get(mode, self._crowd_confidence),
                "crowd_confidence_by_mode": dict(self._crowd_confidence_by_mode),
                "sync_mode": self._active_sync_mode,
                "pending_sync_mode": self._pending_sync_mode,
                "sync_mode_labels": dict(self._sync_mode_labels),
                "sync_mode_options": ["async", "sync"],
                "video_backend": self._active_video_backend,
                "video_backend_options": ["auto", "cpu", "nvjpeg"],
            }

    def request_mode(self, requested: str) -> tuple[bool, dict[str, Any], int]:
        with self._lock:
            available = list(self._available_modes)
            current = self._active_mode
        if requested not in available:
            return False, {"ok": False, "error": f"mode '{requested}' not available"}, 400
        if requested == current:
            return True, {"ok": True, "mode": requested, "changed": False}, 200
        with self._lock:
            self._pending_mode = requested
        return True, {"ok": True, "mode": requested, "changed": True}, 200

    def request_sync_mode(self, requested: str) -> tuple[bool, dict[str, Any], int]:
        normalized_mode = (requested or "").strip().lower()
        if normalized_mode not in ("async", "sync"):
            return False, {"ok": False, "error": "mode must be 'async' or 'sync'"}, 400
        with self._lock:
            current_mode = self._active_sync_mode
        if normalized_mode == current_mode:
            return True, {"ok": True, "mode": normalized_mode, "changed": False}, 200
        with self._lock:
            self._pending_sync_mode = normalized_mode
        return True, {"ok": True, "mode": normalized_mode, "changed": True}, 200

    def set_density_threshold(self, value: float) -> float:
        clamped = max(0.0, min(1.0, float(value)))
        with self._lock:
            self._density_threshold = clamped
            self._pending_density_threshold = clamped
        return clamped

    def set_crowd_confidence(self, value: float) -> float:
        clamped = max(0.05, min(0.95, float(value)))
        with self._lock:
            self._crowd_confidence = clamped
            self._crowd_confidence_by_mode[self._active_mode] = clamped
            self._pending_crowd_confidence = clamped
        return clamped

    def get_and_clear_pending_mode(self) -> str | None:
        with self._lock:
            pending_mode = self._pending_mode
            self._pending_mode = None
            return pending_mode

    def get_and_clear_pending_sync_mode(self) -> str | None:
        with self._lock:
            pending_sync_mode = self._pending_sync_mode
            self._pending_sync_mode = None
            return pending_sync_mode

    def get_and_clear_pending_density_threshold(self) -> float | None:
        with self._lock:
            pending_density_threshold = self._pending_density_threshold
            self._pending_density_threshold = None
            return pending_density_threshold

    def get_and_clear_pending_crowd_confidence(self) -> float | None:
        with self._lock:
            pending_crowd_confidence = self._pending_crowd_confidence
            self._pending_crowd_confidence = None
            return pending_crowd_confidence

    def request_video_backend(self, requested: str) -> tuple[bool, dict[str, Any], int]:
        normalized_backend = (requested or "").strip().lower()
        if normalized_backend not in ("auto", "cpu", "nvjpeg"):
            return False, {"ok": False, "error": "backend must be 'auto', 'cpu' or 'nvjpeg'"}, 400
        with self._lock:
            current_backend = self._active_video_backend
        if normalized_backend == current_backend:
            return True, {"ok": True, "backend": normalized_backend, "changed": False}, 200
        with self._lock:
            self._pending_video_backend = normalized_backend
        return True, {"ok": True, "backend": normalized_backend, "changed": True}, 200

    def get_and_clear_pending_video_backend(self) -> str | None:
        with self._lock:
            pending_video_backend = self._pending_video_backend
            self._pending_video_backend = None
            return pending_video_backend

    def set_active_video_backend(self, backend: str) -> None:
        with self._lock:
            self._active_video_backend = backend

    def set_active_mode(self, mode: str) -> None:
        with self._lock:
            self._active_mode = mode

    def set_active_sync_mode(self, mode: str) -> None:
        with self._lock:
            self._active_sync_mode = mode

    def update_available_modes(self, config: dict[str, Any]) -> None:
        modes = self.compute_available_modes(config)
        with self._lock:
            self._available_modes = modes

    @staticmethod
    def _sync_mode_from_strategy(strategy: str) -> str:
        try:
            strategy_type = FusionStrategyType(strategy)
        except ValueError:
            return "async"
        if strategy_type == FusionStrategyType.RAW_STREAM_WITH_METADATA:
            return "async"
        return "sync"

    @staticmethod
    def compute_available_modes(config: dict[str, Any]) -> list[str]:
        models_config = config.get("models", {})
        project_root = Path(__file__).resolve().parents[3]

        def has_engine(model_name: str) -> bool:
            engine_path = models_config.get(model_name, {}).get("engine", "")
            if not engine_path:
                return False
            resolved_path = Path(engine_path)
            if not resolved_path.is_absolute():
                resolved_path = project_root / resolved_path
            return resolved_path.exists()

        available_engine_by_model = {
            name: has_engine(name)
            for name in ("yolo_global", "yolo_tiles", "density", "p2pnet", "crowd_global", "crowd_tiles")
        }

        available = ["passthrough"]
        if available_engine_by_model["density"]:
            available.append("density")
        if available_engine_by_model["p2pnet"]:
            available.append("p2pnet")
        if available_engine_by_model["yolo_global"]:
            available.append("yolo_global")
        if available_engine_by_model["yolo_tiles"]:
            available.append("yolo_tiles")
        if available_engine_by_model["crowd_global"]:
            available.append("crowd_global")
        if available_engine_by_model["crowd_tiles"]:
            available.append("crowd_tiles")
        return available

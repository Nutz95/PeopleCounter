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
        cfg = initial_config or {}

        self._active_mode: str = detect_mode_from_config(cfg)
        self._pending_mode: str | None = None
        self._available_modes: list[str] = self.compute_available_modes(cfg)

        self._active_sync_mode: str = self._sync_mode_from_strategy(
            str(cfg.get("fusion_strategy", FusionStrategyType.RAW_STREAM_WITH_METADATA.value))
        )
        self._pending_sync_mode: str | None = None
        self._sync_mode_labels: dict[str, str] = {
            "async": "Async (realtime video)",
            "sync": "Sync (video + inference aligned)",
        }

        dcfg = cfg.get("density") or {}
        self._density_threshold: float = float(dcfg.get("min_peak_weight", 0.05))
        self._pending_density_threshold: float | None = None

        model_inf = load_model_inference_config()
        crowd_global_cfg = model_inf.get("crowd_global") or {}
        crowd_tiles_cfg = model_inf.get("crowd_tiles") or {}
        self._crowd_confidence_by_mode: dict[str, float] = {
            "crowd_global": float(crowd_global_cfg.get("confidence_threshold", 0.25)),
            "crowd_tiles": float(crowd_tiles_cfg.get("confidence_threshold", 0.5)),
        }
        self._crowd_confidence: float = self._crowd_confidence_by_mode.get(self._active_mode, 0.25)
        self._pending_crowd_confidence: float | None = None

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
        req = (requested or "").strip().lower()
        if req not in ("async", "sync"):
            return False, {"ok": False, "error": "mode must be 'async' or 'sync'"}, 400
        with self._lock:
            current = self._active_sync_mode
        if req == current:
            return True, {"ok": True, "mode": req, "changed": False}, 200
        with self._lock:
            self._pending_sync_mode = req
        return True, {"ok": True, "mode": req, "changed": True}, 200

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
            m = self._pending_mode
            self._pending_mode = None
            return m

    def get_and_clear_pending_sync_mode(self) -> str | None:
        with self._lock:
            m = self._pending_sync_mode
            self._pending_sync_mode = None
            return m

    def get_and_clear_pending_density_threshold(self) -> float | None:
        with self._lock:
            v = self._pending_density_threshold
            self._pending_density_threshold = None
            return v

    def get_and_clear_pending_crowd_confidence(self) -> float | None:
        with self._lock:
            v = self._pending_crowd_confidence
            self._pending_crowd_confidence = None
            return v

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
            st = FusionStrategyType(strategy)
        except ValueError:
            return "async"
        if st == FusionStrategyType.RAW_STREAM_WITH_METADATA:
            return "async"
        return "sync"

    @staticmethod
    def compute_available_modes(config: dict[str, Any]) -> list[str]:
        models_cfg = config.get("models", {})
        project_root = Path(__file__).resolve().parents[3]

        def has_engine(model_name: str) -> bool:
            engine = models_cfg.get(model_name, {}).get("engine", "")
            if not engine:
                return False
            p = Path(engine)
            if not p.is_absolute():
                p = project_root / p
            return p.exists()

        have = {
            name: has_engine(name)
            for name in ("yolo_global", "yolo_tiles", "density", "crowd_global", "crowd_tiles")
        }

        available = ["passthrough"]
        if have["density"]:
            available.append("density")
        if have["yolo_global"]:
            available.append("yolo_global")
        if have["yolo_tiles"]:
            available.append("yolo_tiles")
        if have["crowd_global"]:
            available.append("crowd_global")
        if have["crowd_tiles"]:
            available.append("crowd_tiles")
        return available

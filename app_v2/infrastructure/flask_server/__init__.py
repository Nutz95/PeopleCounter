"""flask_server package — web UI and SSE publisher."""
from app_v2.infrastructure.flask_server.mode_registry import (
    _INFERENCE_MODES,
    _MODE_LABELS,
    _MODE_OVERLAYS,
    _PREPROCESS_BRANCH_MAP,
    detect_mode_from_config,
)
from app_v2.infrastructure.flask_server.server import FlaskStreamServer

__all__ = [
    "FlaskStreamServer",
    "_INFERENCE_MODES",
    "_MODE_LABELS",
    "_MODE_OVERLAYS",
    "_PREPROCESS_BRANCH_MAP",
    "detect_mode_from_config",
]

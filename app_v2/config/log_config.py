from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from logger.filtered_logger import configure_logger


_LOG_CONFIG_FILE = Path(__file__).parent / "log.yaml"


def load_log_config() -> dict[str, Any]:
    """Load the log configuration that defines active channels."""
    if not _LOG_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing log config: {_LOG_CONFIG_FILE}")
    with _LOG_CONFIG_FILE.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def apply_log_config() -> None:
    """Apply the log channel flags via the shared filtered logger."""
    config = load_log_config()
    channels: Dict[str, bool] = config.get("channels", {})
    configure_logger(
        extreme_debug=channels.get("global"),
        yolo_debug=channels.get("yolo"),
        density_debug=channels.get("density"),
    )

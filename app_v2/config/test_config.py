from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(__file__).resolve().parent
if _CONFIG_DIR.name == "__pycache__":
    _CONFIG_DIR = _CONFIG_DIR.parent
_CONFIG_PATH = _CONFIG_DIR / "test_config.yaml"


def load_test_config() -> dict[str, Any]:
    """Return the test-specific settings that supplement the NVDEC integration run."""
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}

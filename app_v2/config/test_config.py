from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).parent / "test_config.yaml"


def load_test_config() -> dict[str, Any]:
    """Return the test-specific settings that supplement the NVDEC integration run."""
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}

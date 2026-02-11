from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG_FILE = Path(__file__).parent / "pipeline.yaml"


def load_pipeline_config() -> dict[str, Any]:
    """Load the pipeline configuration that drives the v2 orchestrator and fusion strategy."""
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing pipeline config: {_CONFIG_FILE}")
    with _CONFIG_FILE.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuTensor:
    """Representation of a GPU-side tensor produced by preprocessing."""

    model_name: str
    task_index: int
    width: int
    height: int
    buffer_size: int
    device_ptr: int | None
    memory_format: str = "NV12"
    tensor_ref: Any | None = None

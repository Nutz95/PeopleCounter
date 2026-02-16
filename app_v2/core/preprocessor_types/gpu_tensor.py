from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
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
    memory_format: str = "RGB_NCHW_FP16"
    tensor_ref: Any | None = None
    lease_ref: Any | None = None

    def release(self) -> bool:
        releaser = getattr(self.lease_ref, "release", None)
        if callable(releaser):
            return bool(releaser())
        return False

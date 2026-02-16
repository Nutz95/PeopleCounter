from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from app_v2.infrastructure.gpu_tensor_pool.tensor_pool_key import TensorPoolKey


class GpuTensorLease:
    """Lease wrapper that returns tensors to the pool."""

    def __init__(self, token: int, key: TensorPoolKey, tensor: Any, releaser: Callable[[int], bool]) -> None:
        self._token = token
        self.key = key
        self.tensor = tensor
        self._releaser = releaser
        self._released = False
        self._lock = Lock()

    @property
    def token(self) -> int:
        return self._token

    def release(self) -> bool:
        with self._lock:
            if self._released:
                return False
            released = self._releaser(self._token)
            if released:
                self._released = True
            return released

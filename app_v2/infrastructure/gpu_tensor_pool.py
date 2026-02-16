from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Lock
from time import monotonic
from typing import Any, Protocol

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class TensorPoolKey:
    shape: tuple[int, ...]
    dtype: str
    memory_format: str
    stream: int
    device: str


@dataclass(frozen=True, slots=True)
class TensorPoolStats:
    hits: int
    misses: int
    allocations: int
    in_use: int
    available: int
    waits: int
    wait_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "allocations": self.allocations,
            "in_use": self.in_use,
            "available": self.available,
            "waits": self.waits,
            "wait_ms": self.wait_ms,
        }


class PoolSaturationPolicy(Protocol):
    """Defines how `GpuTensorPool` behaves when a key reaches capacity."""

    def wait_for_availability(
        self,
        *,
        cond: Condition,
        timeout_s: float | None,
        on_wait_sample: Callable[[float], None],
    ) -> bool:
        """Return True when caller should retry, False to abort acquisition."""


class BlockingSaturationPolicy:
    """Simple backpressure policy that blocks until a tensor becomes available."""

    def wait_for_availability(
        self,
        *,
        cond: Condition,
        timeout_s: float | None,
        on_wait_sample: Callable[[float], None],
    ) -> bool:
        started = monotonic()
        cond.wait(timeout=timeout_s)
        waited_ms = max(0.0, (monotonic() - started) * 1000.0)
        on_wait_sample(waited_ms)
        if timeout_s is not None and waited_ms >= timeout_s * 1000.0:
            return False
        return True


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


class GpuTensorPool:
    """Reusable GPU tensor pool keyed by shape/format/stream/device."""

    def __init__(
        self,
        *,
        max_per_key: int = 64,
        saturation_policy: PoolSaturationPolicy | None = None,
    ) -> None:
        if max_per_key < 1:
            raise ValueError("max_per_key must be >= 1")
        self._max_per_key = max_per_key
        self._policy = saturation_policy or BlockingSaturationPolicy()
        self._cond = Condition()
        self._free: dict[TensorPoolKey, deque[Any]] = defaultdict(deque)
        self._allocated_per_key: dict[TensorPoolKey, int] = defaultdict(int)
        self._in_use: dict[int, tuple[TensorPoolKey, Any]] = {}
        self._next_token = 1

        self._hits = 0
        self._misses = 0
        self._allocations = 0
        self._waits = 0
        self._wait_ms = 0.0

    def acquire(
        self,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        memory_format: str,
        stream: int,
        device: str = "cuda",
        timeout_s: float | None = None,
    ) -> GpuTensorLease:
        key = TensorPoolKey(
            shape=tuple(int(v) for v in shape),
            dtype=str(dtype),
            memory_format=memory_format,
            stream=int(stream),
            device=str(device),
        )

        with self._cond:
            while True:
                free_queue = self._free[key]
                if free_queue:
                    tensor = free_queue.popleft()
                    self._hits += 1
                    return self._checkout(key, tensor)

                if self._allocated_per_key[key] < self._max_per_key:
                    tensor = self._allocate(key, dtype)
                    self._misses += 1
                    self._allocations += 1
                    self._allocated_per_key[key] += 1
                    return self._checkout(key, tensor)

                can_retry = self._policy.wait_for_availability(
                    cond=self._cond,
                    timeout_s=timeout_s,
                    on_wait_sample=self._on_wait_sample,
                )
                if not can_retry:
                    raise TimeoutError(f"GpuTensorPool saturated for key={key}")

    def _checkout(self, key: TensorPoolKey, tensor: Any) -> GpuTensorLease:
        token = self._next_token
        self._next_token += 1
        self._in_use[token] = (key, tensor)
        return GpuTensorLease(token=token, key=key, tensor=tensor, releaser=self._release_token)

    def _allocate(self, key: TensorPoolKey, dtype: Any) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch is required to allocate GPU tensors")
        if key.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU-only preprocess tensors")
        return torch.empty(key.shape, dtype=dtype, device=key.device)

    def _release_token(self, token: int) -> bool:
        with self._cond:
            payload = self._in_use.pop(token, None)
            if payload is None:
                return False
            key, tensor = payload
            self._free[key].append(tensor)
            self._cond.notify_all()
            return True

    def _on_wait_sample(self, waited_ms: float) -> None:
        self._waits += 1
        self._wait_ms += waited_ms

    def stats_snapshot(self) -> TensorPoolStats:
        with self._cond:
            available = sum(len(queue) for queue in self._free.values())
            return TensorPoolStats(
                hits=self._hits,
                misses=self._misses,
                allocations=self._allocations,
                in_use=len(self._in_use),
                available=available,
                waits=self._waits,
                wait_ms=self._wait_ms,
            )

    def release(self, lease: GpuTensorLease) -> bool:
        return lease.release()

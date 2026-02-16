from __future__ import annotations

import threading
import time

import pytest

from app_v2.infrastructure.gpu_tensor_pool import GpuTensorPool


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU tensor pool tests")
def test_gpu_tensor_pool_reuses_tensor_for_same_key() -> None:
    pool = GpuTensorPool(max_per_key=2)

    lease1 = pool.acquire(
        shape=(3, 64, 64),
        dtype=torch.float16,
        memory_format="RGB_NCHW_FP16",
        stream=0,
        device="cuda",
        timeout_s=1.0,
    )
    ptr1 = int(lease1.tensor.data_ptr())
    assert pool.release(lease1)

    lease2 = pool.acquire(
        shape=(3, 64, 64),
        dtype=torch.float16,
        memory_format="RGB_NCHW_FP16",
        stream=0,
        device="cuda",
        timeout_s=1.0,
    )
    ptr2 = int(lease2.tensor.data_ptr())
    lease2.release()

    stats = pool.stats_snapshot().as_dict()
    assert ptr1 == ptr2
    assert stats["hits"] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU tensor pool tests")
def test_gpu_tensor_pool_blocks_then_unblocks_on_release() -> None:
    pool = GpuTensorPool(max_per_key=1)

    lease1 = pool.acquire(
        shape=(3, 32, 32),
        dtype=torch.float16,
        memory_format="RGB_NCHW_FP16",
        stream=0,
        device="cuda",
        timeout_s=1.0,
    )

    result: dict[str, object] = {}

    def _acquire_later() -> None:
        lease2 = pool.acquire(
            shape=(3, 32, 32),
            dtype=torch.float16,
            memory_format="RGB_NCHW_FP16",
            stream=0,
            device="cuda",
            timeout_s=2.0,
        )
        result["lease"] = lease2

    worker = threading.Thread(target=_acquire_later, daemon=True)
    worker.start()
    time.sleep(0.1)
    assert worker.is_alive(), "Second acquire should block while pool is saturated"

    lease1.release()
    worker.join(timeout=2.0)

    assert "lease" in result
    lease2 = result["lease"]
    assert hasattr(lease2, "release")
    getattr(lease2, "release")()

    stats = pool.stats_snapshot().as_dict()
    assert stats["waits"] >= 1

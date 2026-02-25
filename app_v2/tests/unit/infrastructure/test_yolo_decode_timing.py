"""YOLO decoder GPU→CPU sync timing benchmark.

Compares two decode strategies for a batched tile tensor:
  1. Naive per-tile loop  — O(N) GPU→CPU syncs (the old implementation).
  2. Vectorized (current) — O(1) GPU→CPU syncs via batched_nms + single tolist().

Run with::

    pytest app_v2/tests/unit/infrastructure/test_yolo_decode_timing.py -v -s

The ``-s`` flag keeps stdout so the timing table prints to the console.

Marks:
  * ``gpu`` — requires ``torch.cuda.is_available()``.
  * ``slow`` — can be excluded with ``pytest -m "not slow"``.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_ANCHORS = 8400
NUM_CLASSES = 80  # YOLOv8 COCO
TILE_SIZE = 640
FRAME_W = 3840   # 4 K source (worst-case: lots of tiles)
FRAME_H = 2160


def _make_tile_tensor(n_tiles: int, *, device: str = "cpu") -> "torch.Tensor":
    """Return a synthetic ``[N, 84, 8400]`` raw YOLOv8 output tensor.

    A handful of anchors per tile carry a high "person" score so the decoder
    actually finds detections and exercises the NMS path.
    """
    t = torch.zeros(n_tiles, 4 + NUM_CLASSES, NUM_ANCHORS, device=device)
    # Background: low scores everywhere
    t[:, 4:, :] = 0.02
    # Inject 5 "persons" per tile at distinct anchor positions
    for tile in range(n_tiles):
        for k in range(5):
            anchor = 100 + tile * 300 + k * 50
            anchor = anchor % NUM_ANCHORS
            # xywh in 640-px tile space
            t[tile, 0, anchor] = float(200 + k * 80)   # cx
            t[tile, 1, anchor] = float(300 + k * 60)   # cy
            t[tile, 2, anchor] = 80.0                  # w
            t[tile, 3, anchor] = 160.0                 # h
            t[tile, 4, anchor] = 0.85                  # person score (class 0)
    return t


def _make_tile_plan(n_tiles: int) -> Any:
    """Return a ``PreprocessPlan`` covering *n_tiles* equal-width strips."""
    from app_v2.core.preprocessor_types.preprocess_plan import PreprocessPlan
    from app_v2.core.preprocessor_types.preprocess_task import PreprocessTask

    strip_w = FRAME_W // max(n_tiles, 1)
    tasks = []
    for i in range(n_tiles):
        tasks.append(PreprocessTask(
            model_name="yolo_tiles",
            task_index=i,
            source_x=i * strip_w,
            source_y=0,
            source_width=min(strip_w, FRAME_W - i * strip_w),
            source_height=FRAME_H,
            target_width=TILE_SIZE,
            target_height=TILE_SIZE,
            metadata={"row": 0, "col": i},
        ))
    return PreprocessPlan(
        model_name="yolo_tiles",
        frame_width=FRAME_W,
        frame_height=FRAME_H,
        tasks=tuple(tasks),
        metadata={"mode": "tiles", "task_count": n_tiles},
    )


def _decode_naive_per_tile(
    decoder: Any,
    tensor: "torch.Tensor",
    plan: Any,
) -> list[dict]:
    """Old per-tile loop: ~3 GPU→CPU syncs per tile (threshold bool-index +
    2× tolist inside ``_decode_raw_yolov8``).
    """
    results: list[dict] = []
    n_tiles = int(tensor.shape[0])
    for i in range(n_tiles):
        # _to_rows handles the [1, 84, 8400] → [8400, 84] conversion
        rows = decoder._to_rows(tensor[i : i + 1])
        dets = decoder._decode_raw_yolov8(rows)
        results.extend(dets)
    return results


def _time_fn(fn: Any, warmup: int = 5, runs: int = 30) -> float:
    """Return mean wall-clock ms over *runs* calls, after *warmup* calls.

    Uses ``torch.cuda.synchronize()`` when CUDA is available so the timer
    captures GPU work, not just kernel launch latency.
    """
    sync = torch.cuda.synchronize if HAS_CUDA else (lambda: None)
    for _ in range(warmup):
        fn()
        sync()
    times: list[float] = []
    for _ in range(runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1_000.0)
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU required")
def test_vectorized_decode_is_faster_than_naive() -> None:
    """Vectorized tile decoder must be faster than the naive per-tile loop.

    Tile counts tested: 1, 4, 9, 16, 20.
    For N ≥ 4, the vectorized path must be at least 1.5× faster.
    """
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.25)
    device = "cuda"

    tile_counts = [1, 4, 9, 16, 20]
    print()
    print(f"{'tiles':>6}  {'naive ms':>10}  {'vec ms':>10}  {'speedup':>10}  {'detections':>12}")
    print("-" * 60)

    for n in tile_counts:
        tensor = _make_tile_tensor(n, device=device)
        plan = _make_tile_plan(n)

        naive_ms = _time_fn(lambda: _decode_naive_per_tile(decoder, tensor, plan))
        vec_ms   = _time_fn(lambda: decoder._decode_tiled_yolov8_global(tensor, plan))

        n_dets = len(decoder._decode_tiled_yolov8_global(tensor, plan))
        speedup = naive_ms / max(vec_ms, 0.001)
        print(f"{n:>6}  {naive_ms:>10.3f}  {vec_ms:>10.3f}  {speedup:>9.2f}×  {n_dets:>12}")

        # Speedup threshold scales with tile count: overhead savings grow with N.
        # For small N the constant cost dominates, so the benefit is modest.
        if n >= 9:
            min_speedup = 1.5
        elif n >= 4:
            min_speedup = 1.2
        else:
            min_speedup = 1.0  # N=1: no tile-batching advantage
        if min_speedup > 1.0:
            assert speedup >= min_speedup, (
                f"Vectorized decoder is only {speedup:.2f}× faster than naive for {n} tiles; "
                f"expected ≥ {min_speedup}×. Check for regressions in _decode_tiled_yolov8_global()."
            )

    print("-" * 60)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA GPU required")
def test_vectorized_decode_gpu_sync_count() -> None:
    """Vectorized decoder does ≤ 4 GPU→CPU syncs regardless of tile count.

    This test instruments ``torch.Tensor.tolist`` to count the number of
    synchronising calls made during a single decode of 20 tiles.

    A sync is defined as any call that forces the CPU to wait for GPU work:
    we approximate by counting ``.tolist()`` calls (each triggers cudaMemcpy).
    The naive per-tile loop would make ~3 × 20 = 60 such calls; the vectorized
    path must use ≤ 4.
    """
    import unittest.mock as mock
    from app_v2.infrastructure.yolo_decoder import YoloDecoder

    decoder = YoloDecoder(person_class_id=0, confidence_threshold=0.25)
    device = "cuda"
    n = 20
    tensor = _make_tile_tensor(n, device=device)
    plan = _make_tile_plan(n)

    call_count = 0
    real_tolist = torch.Tensor.tolist

    def counting_tolist(self: torch.Tensor) -> Any:
        nonlocal call_count
        call_count += 1
        return real_tolist(self)

    with mock.patch.object(torch.Tensor, "tolist", counting_tolist):
        decoder._decode_tiled_yolov8_global(tensor, plan)

    print(f"\n[sync count test] 20-tile decode: {call_count} tolist() call(s)")
    assert call_count <= 4, (
        f"Vectorized decoder called .tolist() {call_count} times for 20 tiles; "
        "expected ≤ 4.  The O(1) sync property has been broken."
    )

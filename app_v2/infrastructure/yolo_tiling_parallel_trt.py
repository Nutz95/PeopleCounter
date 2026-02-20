from __future__ import annotations

import concurrent.futures
import math
import time
from typing import Any, Sequence

from app_v2.core.inference_model import InferenceModel
from app_v2.infrastructure.yolo_decoder import YoloDecoder


class YoloTilingParallelTRT(InferenceModel):
    """TensorRT tiling model that splits tiles into N groups and processes them in parallel."""

    def __init__(
        self,
        contexts: list[Any],
        stream_id: int,
        groups: int = 2,
        inference_params: dict[str, Any] | None = None,
    ) -> None:
        self._contexts = contexts
        self._stream_id = stream_id
        self._name = "yolo_tiles"
        self._groups = groups
        self._inference_params = dict(inference_params or {})
        self._decoder = YoloDecoder(person_class_id=int(self._inference_params.get("person_class_id", 0)))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=groups, thread_name_prefix="yolo_tiles_group")

    @property
    def name(self) -> str:
        return self._name

    @property
    def stream_id(self) -> int:
        return self._stream_id

    def warm_up(self, batch_size: int) -> None:
        """Bind TensorRT profiles for the expected batch."""
        pass

    def infer(self, frame_id: int, inputs: Sequence[Any], *, preprocess_events: Sequence[Any] | None = None, tile_plan: Any | None = None) -> dict[str, Any]:
        """Execute tiled inference in parallel groups and return merged results + metrics."""
        start_ns = time.perf_counter_ns()
        
        # Partition tiles into groups
        tiles = list(inputs)
        tile_count = len(tiles)
        tiles_per_group = math.ceil(tile_count / self._groups)
        
        partitions: list[list[Any]] = []
        for g in range(self._groups):
            start_idx = g * tiles_per_group
            end_idx = min(start_idx + tiles_per_group, tile_count)
            if start_idx < tile_count:
                partitions.append(tiles[start_idx:end_idx])
            else:
                partitions.append([])
        
        # Execute groups in parallel
        futures = []
        for group_idx, partition in enumerate(partitions):
            if not partition:
                continue
            future = self._executor.submit(
                self._execute_group,
                frame_id=frame_id,
                group_idx=group_idx,
                tiles=partition,
                preprocess_events=list(preprocess_events or []),
            )
            futures.append((group_idx, future))
        
        # Collect results
        group_results: dict[int, dict[str, Any]] = {}
        for group_idx, future in futures:
            group_results[group_idx] = future.result()
        
        # Merge detections from all groups
        all_detections: list[Any] = []
        total_prepare_batch_ms = 0.0
        total_enqueue_ms = 0.0
        total_stream_sync_ms = 0.0
        total_decode_ms = 0.0
        max_group_ms = 0.0
        
        for group_idx in sorted(group_results.keys()):
            result = group_results[group_idx]
            all_detections.extend(result.get("detections", []))
            total_prepare_batch_ms += result.get("prepare_batch_ms", 0.0)
            total_enqueue_ms += result.get("enqueue_ms", 0.0)
            total_stream_sync_ms += result.get("stream_sync_ms", 0.0)
            total_decode_ms += result.get("decode_ms", 0.0)
            max_group_ms = max(max_group_ms, result.get("group_ms", 0.0))
        
        infer_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        
        return {
            "frame_id": frame_id,
            "model": self._name,
            "prediction": {"status": "ok"},
            "segmentation": None,
            "detections": all_detections,
            "inference_params": self._inference_params,
            "person_class_id": int(self._inference_params.get("person_class_id", 0)),
            "class_whitelist": list(self._inference_params.get("class_whitelist", [0])),
            "tile_count": tile_count,
            "groups": self._groups,
            "inference_ms": float(infer_ms),
            "prepare_batch_ms": float(total_prepare_batch_ms),
            "enqueue_ms": float(total_enqueue_ms),
            "stream_sync_ms": float(total_stream_sync_ms),
            "decode_ms": float(total_decode_ms),
            "max_group_ms": float(max_group_ms),
        }

    def _execute_group(self, frame_id: int, group_idx: int, tiles: list[Any], preprocess_events: list[Any] | None = None) -> dict[str, Any]:
        """Execute inference for one group of tiles on a dedicated TRT context."""
        stream_key = f"model:{self._name}:g{group_idx}"
        group_start_ns = time.perf_counter_ns()
        
        context = self._contexts[group_idx % len(self._contexts)]
        context.bind_stream(stream_key)
        try:
            raw_outputs = context.execute(
                {
                    "frame_id": frame_id,
                    "inputs": tiles,
                    "model": f"{self._name}_g{group_idx}",
                    "params": dict(self._inference_params),
                    "preprocess_events": list(preprocess_events or []),
                }
            )
            decode_start_ns = time.perf_counter_ns()
            decoded = self._decoder.process(frame_id, raw_outputs)
            decode_ms = (time.perf_counter_ns() - decode_start_ns) / 1_000_000.0
            group_ms = (time.perf_counter_ns() - group_start_ns) / 1_000_000.0
            
            return {
                "detections": decoded.get("detections", []),
                "prepare_batch_ms": float(raw_outputs.get("prepare_batch_ms", 0.0)),
                "enqueue_ms": float(raw_outputs.get("enqueue_ms", 0.0)),
                "stream_sync_ms": float(raw_outputs.get("stream_sync_ms", 0.0)),
                "decode_ms": float(decode_ms),
                "group_ms": float(group_ms),
                "tile_count": len(tiles),
            }
        finally:
            context.release_stream(stream_key)

    def close(self) -> None:
        """Release the CUDA resources and thread pool."""
        self._executor.shutdown(wait=True)

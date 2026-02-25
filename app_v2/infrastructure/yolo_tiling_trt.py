from __future__ import annotations

import sys
import time
from typing import Any, Sequence

from app_v2.core.inference_model import InferenceModel
from app_v2.infrastructure.yolo_decoder import YoloDecoder


class YoloTilingTRT(InferenceModel):
    """TensorRT tiling model that processes many 640x640 tiles concurrently."""

    def __init__(self, engine_context: Any, stream_id: int, inference_params: dict[str, Any] | None = None) -> None:
        self._context = engine_context
        self._stream_id = stream_id
        self._name = "yolo_tiles"
        self._inference_params = dict(inference_params or {})
        self._decoder = YoloDecoder(
            person_class_id=int(self._inference_params.get("person_class_id", 0)),
            confidence_threshold=float(self._inference_params.get("confidence_threshold", 0.25)),
        )
        self._decoder.seg_mask_enabled = bool(
            self._inference_params.get("seg_mask_enabled", False)
        )
        self._decoder.seg_mask_clip_to_bbox = bool(
            self._inference_params.get("seg_mask_clip_to_bbox", True)
        )

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
        """Execute tiled inference and return decoded placeholders + metrics.

        Auto-splits tiles into sub-batches when the full tile count exceeds the
        TRT engine's max batch size (profile 0).  This mirrors what
        ``YoloTilingParallelTRT`` does with groups so the same engine works in
        both modes regardless of the tile count.
        """
        tiles = list(inputs)
        max_batch = self._context.max_batch_size  # 0 = unknown / unconstrained
        if max_batch > 0 and len(tiles) > max_batch:
            return self._infer_chunked(frame_id, tiles, max_batch,
                                       preprocess_events=preprocess_events,
                                       tile_plan=tile_plan)
        return self._infer_batch(frame_id, tiles,
                                 preprocess_events=preprocess_events,
                                 tile_plan=tile_plan)

    def _infer_batch(self, frame_id: int, tiles: list[Any], *, preprocess_events: Sequence[Any] | None = None, tile_plan: Any | None = None) -> dict[str, Any]:
        """Run a single TRT execute call for tiles that fit within max_batch."""
        stream_key = f"model:{self._name}"
        start_ns = time.perf_counter_ns()
        self._context.bind_stream(stream_key)
        try:
            raw_outputs = self._context.execute(
                {
                    "frame_id": frame_id,
                    "inputs": tiles,
                    "model": self._name,
                    "params": dict(self._inference_params),
                    "preprocess_events": list(preprocess_events or []),
                }
            )
            if not isinstance(raw_outputs, dict) or raw_outputs.get("status") != "ok":
                reason = raw_outputs.get("reason", "unknown") if isinstance(raw_outputs, dict) else str(raw_outputs)
                print(
                    f"[yolo_tiles] TRT execute failed (batch={len(tiles)}): {reason}",
                    file=sys.stderr, flush=True,
                )
                return {
                    "frame_id": frame_id, "model": self._name, "prediction": {},
                    "segmentation": None, "detections": [], "seg_mask_raw": None,
                    "seg_mask_w": 0, "seg_mask_h": 0,
                    "inference_params": self._inference_params,
                    "person_class_id": int(self._inference_params.get("person_class_id", 0)),
                    "class_whitelist": list(self._inference_params.get("class_whitelist", [0])),
                    "tile_count": len(tiles), "inference_ms": 0.0,
                    "prepare_batch_ms": 0.0, "enqueue_ms": 0.0,
                    "stream_sync_ms": 0.0, "decode_ms": 0.0,
                }
            gpu_done_ns = time.perf_counter_ns()
            decode_start_ns = gpu_done_ns
            decoded = self._decoder.process(frame_id, raw_outputs, tile_plan=tile_plan)
            decode_ms = (time.perf_counter_ns() - decode_start_ns) / 1_000_000.0
            infer_ms = (gpu_done_ns - start_ns) / 1_000_000.0
            return {
                "frame_id": frame_id,
                "model": self._name,
                "prediction": {
                    k: v for k, v in raw_outputs.items()
                    if k not in ("outputs", "output_tensors", "inputs", "input_tensor")
                } if isinstance(raw_outputs, dict) else {},
                "segmentation": raw_outputs.get("segmentation") if isinstance(raw_outputs, dict) else None,
                "detections": decoded.get("detections", []),
                "seg_mask_raw": decoded.get("seg_mask_raw"),
                "seg_mask_w": decoded.get("seg_mask_w", 0),
                "seg_mask_h": decoded.get("seg_mask_h", 0),
                "inference_params": self._inference_params,
                "person_class_id": int(self._inference_params.get("person_class_id", 0)),
                "class_whitelist": list(self._inference_params.get("class_whitelist", [0])),
                "tile_count": len(tiles),
                "inference_ms": float(infer_ms),
                "prepare_batch_ms": float(raw_outputs.get("prepare_batch_ms", 0.0)),
                "enqueue_ms": float(raw_outputs.get("enqueue_ms", 0.0)),
                "stream_sync_ms": float(raw_outputs.get("stream_sync_ms", 0.0)),
                "decode_ms": float(decode_ms),
            }
        finally:
            self._context.release_stream(stream_key)

    def _infer_chunked(self, frame_id: int, tiles: list[Any], max_batch: int, *, preprocess_events: Sequence[Any] | None = None, tile_plan: Any | None = None) -> dict[str, Any]:
        """Split tiles into sub-batches of max_batch and merge detections."""
        import math
        all_detections: list[Any] = []
        total_infer_ms = total_prepare_ms = total_enqueue_ms = total_sync_ms = total_decode_ms = 0.0
        n_chunks = math.ceil(len(tiles) / max_batch)
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * max_batch
            chunk_tiles = tiles[chunk_start: chunk_start + max_batch]
            # Build a sub-plan covering only this chunk's tasks.
            chunk_plan: Any = None
            if tile_plan is not None:
                try:
                    from app_v2.core.preprocessor_types.preprocess_plan import PreprocessPlan as _PP  # noqa: PLC0415
                    sub_tasks = tile_plan.tasks[chunk_start: chunk_start + len(chunk_tiles)]
                    chunk_plan = _PP(
                        model_name=tile_plan.model_name,
                        frame_width=tile_plan.frame_width,
                        frame_height=tile_plan.frame_height,
                        tasks=sub_tasks,
                        metadata=tile_plan.metadata,
                    )
                except Exception:
                    pass
            result = self._infer_batch(frame_id, chunk_tiles,
                                       preprocess_events=preprocess_events,
                                       tile_plan=chunk_plan)
            all_detections.extend(result.get("detections", []))
            total_infer_ms += result.get("inference_ms", 0.0)
            total_prepare_ms += result.get("prepare_batch_ms", 0.0)
            total_enqueue_ms += result.get("enqueue_ms", 0.0)
            total_sync_ms += result.get("stream_sync_ms", 0.0)
            total_decode_ms += result.get("decode_ms", 0.0)
        return {
            "frame_id": frame_id,
            "model": self._name,
            "prediction": {"status": "ok"},
            "segmentation": None,
            "detections": all_detections,
            "seg_mask_raw": None,
            "seg_mask_w": 0,
            "seg_mask_h": 0,
            "inference_params": self._inference_params,
            "person_class_id": int(self._inference_params.get("person_class_id", 0)),
            "class_whitelist": list(self._inference_params.get("class_whitelist", [0])),
            "tile_count": len(tiles),
            "inference_ms": float(total_infer_ms),
            "prepare_batch_ms": float(total_prepare_ms),
            "enqueue_ms": float(total_enqueue_ms),
            "stream_sync_ms": float(total_sync_ms),
            "decode_ms": float(total_decode_ms),
        }

    def close(self) -> None:
        """Release the CUDA resources held by this tiling model."""
        pass

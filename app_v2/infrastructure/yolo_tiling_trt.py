from __future__ import annotations

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
        self._decoder = YoloDecoder(person_class_id=int(self._inference_params.get("person_class_id", 0)))

    @property
    def name(self) -> str:
        return self._name

    @property
    def stream_id(self) -> int:
        return self._stream_id

    def warm_up(self, batch_size: int) -> None:
        """Bind TensorRT profiles for the expected batch."""
        pass

    def infer(self, frame_id: int, inputs: Sequence[Any], *, preprocess_events: Sequence[Any] | None = None) -> dict[str, Any]:
        """Execute tiled inference and return decoded placeholders + metrics."""
        stream_key = f"model:{self._name}"
        start_ns = time.perf_counter_ns()
        self._context.bind_stream(stream_key)
        try:
            raw_outputs = self._context.execute(
                {
                    "frame_id": frame_id,
                    "inputs": list(inputs),
                    "model": self._name,
                    "params": dict(self._inference_params),
                    "preprocess_events": list(preprocess_events or []),
                }
            )
            decode_start_ns = time.perf_counter_ns()
            decoded = self._decoder.process(frame_id, raw_outputs)
            decode_ms = (time.perf_counter_ns() - decode_start_ns) / 1_000_000.0
            infer_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return {
                "frame_id": frame_id,
                "model": self._name,
                "prediction": raw_outputs,
                "segmentation": raw_outputs.get("segmentation") if isinstance(raw_outputs, dict) else None,
                "detections": decoded.get("detections", []),
                "inference_params": self._inference_params,
                "person_class_id": int(self._inference_params.get("person_class_id", 0)),
                "class_whitelist": list(self._inference_params.get("class_whitelist", [0])),
                "tile_count": len(inputs),
                "inference_ms": float(infer_ms),
                "prepare_batch_ms": float(raw_outputs.get("prepare_batch_ms", 0.0)),
                "enqueue_ms": float(raw_outputs.get("enqueue_ms", 0.0)),
                "stream_sync_ms": float(raw_outputs.get("stream_sync_ms", 0.0)),
                "decode_ms": float(decode_ms),
            }
        finally:
            self._context.release_stream(stream_key)

    def close(self) -> None:
        """Release the CUDA resources held by this tiling model."""
        pass

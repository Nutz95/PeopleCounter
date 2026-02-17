from __future__ import annotations

from contextlib import nullcontext
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class PreprocessStreamManager:
    """Owns preprocess stream mapping and CUDA stream lifecycle."""

    def __init__(self) -> None:
        self._stream_by_model: dict[str, int] = {}
        self._cuda_streams: dict[int, Any] = {}

    def configure(self, metadata: dict[str, Any]) -> None:
        self._stream_by_model = self.build_stream_map(metadata)
        self._cuda_streams = {}

    def stream_for_model(self, model_name: str) -> int:
        return int(self._stream_by_model.get(model_name, 0))

    def stream_context(self, stream_id: int) -> Any:
        stream = self._get_or_create_cuda_stream(stream_id)
        if stream is None:
            return nullcontext()
        assert torch is not None
        return torch.cuda.stream(stream)

    def synchronize_streams(self, stream_ids: set[int]) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        for stream_id in sorted(stream_ids):
            stream = self._get_or_create_cuda_stream(stream_id)
            if stream is not None:
                stream.synchronize()

    def stream_handle(self, stream_id: int) -> int | None:
        stream = self._get_or_create_cuda_stream(stream_id)
        if stream is None:
            return None
        handle = getattr(stream, "cuda_stream", None)
        if handle is None:
            return None
        return int(handle)

    def _get_or_create_cuda_stream(self, stream_id: int) -> Any | None:
        if torch is None or not torch.cuda.is_available():
            return None
        normalized_stream_id = int(stream_id)
        if normalized_stream_id == 0:
            return torch.cuda.default_stream()
        existing = self._cuda_streams.get(normalized_stream_id)
        if existing is not None:
            return existing
        created = torch.cuda.Stream()
        self._cuda_streams[normalized_stream_id] = created
        return created

    @staticmethod
    def build_stream_map(metadata: dict[str, Any]) -> dict[str, int]:
        streams = metadata.get("streams", {})
        if not isinstance(streams, dict):
            return {}
        mapping: dict[str, int] = {}
        yolo_stream = int(streams.get("yolo", 0))
        yolo_global_preprocess_stream = int(streams.get("yolo_global_preprocess", yolo_stream))
        yolo_tiles_preprocess_stream = int(streams.get("yolo_tiles_preprocess", yolo_stream))
        density_stream = int(streams.get("density", 0))
        density_preprocess_stream = int(streams.get("density_preprocess", density_stream))
        for model_name in metadata.get("preprocess", {}).keys():
            normalized_name = str(model_name)
            if normalized_name == "yolo_global":
                mapping[normalized_name] = yolo_global_preprocess_stream
            elif normalized_name.startswith("yolo_tiles"):
                mapping[normalized_name] = yolo_tiles_preprocess_stream
            elif normalized_name.startswith("density") or normalized_name.startswith("lwcc"):
                mapping[normalized_name] = density_preprocess_stream
            else:
                mapping[normalized_name] = 0
        return mapping

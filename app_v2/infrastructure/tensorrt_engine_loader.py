from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

try:
    import tensorrt as trt
except Exception:  # pragma: no cover
    trt = None  # type: ignore[assignment]


class TensorRTEngineLoader:
    """Loads TensorRT engine blobs and exposes metadata for other infrastructure classes."""

    def __init__(self, engine_path: str, profiles: Dict[str, Any]) -> None:
        self.engine_path = engine_path
        self.profiles = profiles
        self._metadata: dict[str, Any] = {}
        self._runtime: Any | None = None
        self._engine: Any | None = None

    @property
    def engine(self) -> Any | None:
        return self._engine

    def load(self) -> dict[str, Any]:
        """Deserialize the engine and cache tensor names/shapes."""
        if not self._metadata:
            resolved = self._resolve_engine_path(self.engine_path)
            metadata: dict[str, Any] = {
                "path": str(resolved) if resolved else self.engine_path,
                "profiles": self.profiles,
                "available": False,
                "reason": "engine unavailable",
                "io_tensors": [],
            }
            if resolved is None or not resolved.exists():
                metadata["reason"] = "engine file not found"
                self._metadata = metadata
                return self._metadata
            if trt is None:
                metadata["reason"] = "tensorrt python package unavailable"
                self._metadata = metadata
                return self._metadata

            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(resolved.read_bytes())
            if engine is None:
                metadata["reason"] = "deserialize_cuda_engine returned None"
                self._metadata = metadata
                return self._metadata

            io_tensors: list[dict[str, Any]] = []
            for index in range(int(engine.num_io_tensors)):
                name = engine.get_tensor_name(index)
                mode = engine.get_tensor_mode(name)
                io_tensors.append(
                    {
                        "name": name,
                        "mode": "input" if mode == trt.TensorIOMode.INPUT else "output",
                        "dtype": str(engine.get_tensor_dtype(name)),
                        "shape": tuple(int(x) for x in engine.get_tensor_shape(name)),
                    }
                )

            self._runtime = runtime
            self._engine = engine
            metadata.update({"available": True, "reason": "ok", "io_tensors": io_tensors})
            self._metadata = metadata
        return self._metadata

    @staticmethod
    def _resolve_engine_path(engine_path: str) -> Path | None:
        candidate = Path(engine_path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / candidate
        return candidate

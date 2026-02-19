from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

try:
    import tensorrt as trt
    # TRT uses a process-global logger singleton (nvinfer1::getLogger()).  The
    # *first* trt.Runtime(logger) call registers that logger globally.  If the
    # logger object is ever garbage-collected while TRT is still active, TRT's
    # internal pointer becomes dangling and the next createInferRuntime call
    # segfaults with CUDA error 700.  Keeping one module-level instance alive
    # for the lifetime of the process prevents this.
    _TRT_LOGGER: "trt.Logger | None" = trt.Logger(trt.Logger.WARNING)
except Exception:  # pragma: no cover
    trt = None  # type: ignore[assignment]
    _TRT_LOGGER = None

try:
    from app_v2.infrastructure.engine_refitter import EngineRefitter, get_onnx_path_for_engine
    _REFITTER_AVAILABLE = True
except Exception:  # pragma: no cover
    _REFITTER_AVAILABLE = False


class TensorRTEngineLoader:
    """Loads TensorRT engine blobs and exposes metadata for other infrastructure classes.

    Weight-stripped engines (produced with ``convert_onnx_to_trt.py --strip``)
    are detected automatically via their ``.onnxpath`` sidecar file and refitted
    transparently before use.  Callers do not need to handle this case explicitly.
    """

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

            logger = _TRT_LOGGER or trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            # Auto-detect weight-stripped engine and refit from ONNX sidecar.
            engine_bytes = resolved.read_bytes()
            if _REFITTER_AVAILABLE and get_onnx_path_for_engine(resolved) is not None:
                refitter = EngineRefitter(resolved)
                engine = refitter.load_and_refit()
                if engine is not None:
                    print(f"[TensorRTEngineLoader] Refitted stripped engine from {resolved.name}")
                else:
                    metadata["reason"] = "engine refit failed"
                    self._metadata = metadata
                    return self._metadata
            else:
                engine = runtime.deserialize_cuda_engine(engine_bytes)
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

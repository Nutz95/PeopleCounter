"""Weight-Stripped Engine Refitter (Opt #4).

A TensorRT engine built with ``BuilderFlag.STRIP_PLAN`` contains no weight data
and must be *refitted* before first use.  This module provides a thin wrapper
around ``trt.Refitter`` + ``trt.OnnxParserRefitter`` so that the rest of the
application can treat stripped engines transparently.

Sidecar convention
------------------
When ``convert_onnx_to_trt.build_engine()`` is called with ``strip_plan=True`` it
writes a text file next to the engine:

    <engine_path>.onnxpath   →   /abs/path/to/source.onnx

``EngineRefitter`` reads that sidecar automatically; callers may also pass
``onnx_path`` explicitly.

Usage::

    refitter = EngineRefitter("model-stripped.engine")
    engine   = refitter.load_and_refit()
    ctx      = engine.create_execution_context()
"""

from __future__ import annotations

import logging
from pathlib import Path

import tensorrt as trt  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)
_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Sidecar suffix: "<engine>.onnxpath" stores the absolute ONNX path as plain text
REFIT_SIDECAR_SUFFIX = ".onnxpath"


def get_onnx_path_for_engine(engine_path: str | Path) -> Path | None:
    """Return the ONNX source path from the sidecar file, or *None* if absent."""
    ep = Path(engine_path)
    sidecar = Path(str(ep) + REFIT_SIDECAR_SUFFIX)
    if sidecar.exists():
        candidate = Path(sidecar.read_text().strip())
        if candidate.exists():
            return candidate
    return None


def write_refit_sidecar(engine_path: str | Path, onnx_path: str | Path) -> None:
    """Write a sidecar file so the engine can be refitted without extra args."""
    sidecar = Path(str(engine_path) + REFIT_SIDECAR_SUFFIX)
    sidecar.write_text(str(Path(onnx_path).resolve()))


class EngineRefitter:
    """Loads a weight-stripped TRT engine and refits it from its ONNX source."""

    def __init__(
        self,
        engine_path: str | Path,
        onnx_path: str | Path | None = None,
    ) -> None:
        self._engine_path = Path(engine_path)
        if onnx_path is None:
            onnx_path = get_onnx_path_for_engine(engine_path)
            if onnx_path is None:
                raise FileNotFoundError(
                    f"No ONNX source found for stripped engine '{engine_path}'. "
                    "Pass onnx_path explicitly or create a .onnxpath sidecar."
                )
        self._onnx_path = Path(onnx_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_and_refit(self) -> trt.ICudaEngine:
        """Deserialise + refit. Returns the engine ready for inference."""
        runtime = trt.Runtime(_TRT_LOGGER)
        engine_bytes = self._engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialise engine '{self._engine_path}'"
            )

        refitter = trt.Refitter(engine, _TRT_LOGGER)
        parser_refitter = trt.OnnxParserRefitter(refitter, _TRT_LOGGER)

        if not parser_refitter.refit_from_file(str(self._onnx_path)):
            raise RuntimeError(
                f"OnnxParserRefitter failed refitting from '{self._onnx_path}'"
            )
        if not refitter.refit_cuda_engine():
            raise RuntimeError("refitter.refit_cuda_engine() returned False")

        mb = len(engine_bytes) / (1024 * 1024)
        _log.info(
            "[WeightStrip] Refitted '%s' (%.1f MB) from '%s'",
            self._engine_path.name,
            mb,
            self._onnx_path.name,
        )
        return engine

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def engine_path(self) -> Path:
        return self._engine_path

    @property
    def onnx_path(self) -> Path:
        return self._onnx_path

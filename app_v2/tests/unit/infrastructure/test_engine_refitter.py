"""Unit tests for app_v2.infrastructure.engine_refitter (Opt #4 — Weight Stripping).

All TRT objects are mocked so tests run without a GPU / TRT installation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Mock tensorrt before the module is imported (no GPU / TRT required)
# ---------------------------------------------------------------------------
_fake_trt = MagicMock()
_fake_trt.Logger.WARNING = "WARNING"
_fake_trt.BuilderFlag.STRIP_PLAN = "STRIP_PLAN"
sys.modules.setdefault("tensorrt", _fake_trt)

from app_v2.infrastructure.engine_refitter import (  # noqa: E402
    EngineRefitter,
    REFIT_SIDECAR_SUFFIX,
    get_onnx_path_for_engine,
    write_refit_sidecar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_file(tmp_path: Path, engine_name: str = "model.engine") -> Path:
    engine_file_path = tmp_path / engine_name
    engine_file_path.write_bytes(b"\x00\x01\x02\x03")
    return engine_file_path


def _make_onnx_file(tmp_path: Path, onnx_name: str = "model.onnx") -> Path:
    onnx_file_path = tmp_path / onnx_name
    onnx_file_path.write_bytes(b"\x08\x00")  # minimal fake ONNX bytes
    return onnx_file_path


# ---------------------------------------------------------------------------
# Tests: sidecar helpers
# ---------------------------------------------------------------------------


class TestSidecarHelpers:
    def test_write_and_read_sidecar(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)

        write_refit_sidecar(engine_file_path, onnx_file_path)
        sidecar = Path(str(engine_file_path) + REFIT_SIDECAR_SUFFIX)
        assert sidecar.exists(), "sidecar file should be created"
        result = get_onnx_path_for_engine(engine_file_path)
        assert result == onnx_file_path.resolve()

    def test_get_onnx_path_returns_none_when_no_sidecar(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        assert get_onnx_path_for_engine(engine_file_path) is None

    def test_get_onnx_path_returns_none_when_sidecar_points_to_missing_file(
        self, tmp_path: Path
    ):
        engine_file_path = _make_engine_file(tmp_path)
        sidecar = Path(str(engine_file_path) + REFIT_SIDECAR_SUFFIX)
        sidecar.write_text("/nonexistent/path/model.onnx")
        assert get_onnx_path_for_engine(engine_file_path) is None

    def test_sidecar_stores_absolute_path(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)
        write_refit_sidecar(engine_file_path, onnx_file_path)
        sidecar_text = Path(str(engine_file_path) + REFIT_SIDECAR_SUFFIX).read_text().strip()
        assert Path(sidecar_text).is_absolute()


# ---------------------------------------------------------------------------
# Tests: EngineRefitter constructor
# ---------------------------------------------------------------------------


class TestEngineRefitterInit:
    def test_explicit_onnx_path(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)
        refitter = EngineRefitter(engine_file_path, onnx_file_path)
        assert refitter.engine_path == engine_file_path
        assert refitter.onnx_path == onnx_file_path

    def test_auto_onnx_from_sidecar(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)
        write_refit_sidecar(engine_file_path, onnx_file_path)
        refitter = EngineRefitter(engine_file_path)
        assert refitter.onnx_path == onnx_file_path.resolve()

    def test_raises_when_no_onnx_and_no_sidecar(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        with pytest.raises(FileNotFoundError, match="No ONNX source found"):
            EngineRefitter(engine_file_path)


# ---------------------------------------------------------------------------
# Tests: EngineRefitter.load_and_refit
# ---------------------------------------------------------------------------


class TestEngineRefitterLoadAndRefit:
    def _make_mock_trt(self):
        """Return configured mock objects for the TRT refit pipeline."""
        mock_engine = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        mock_refitter = MagicMock()
        mock_refitter.refit_cuda_engine.return_value = True

        mock_parser_refitter = MagicMock()
        mock_parser_refitter.refit_from_file.return_value = True

        return mock_engine, mock_runtime, mock_refitter, mock_parser_refitter

    def test_refit_succeeds(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)
        mock_engine, mock_runtime, mock_refitter, mock_parser_refitter = self._make_mock_trt()

        with (
            patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=mock_runtime),
            patch("app_v2.infrastructure.engine_refitter.trt.Refitter", return_value=mock_refitter),
            patch(
                "app_v2.infrastructure.engine_refitter.trt.OnnxParserRefitter",
                return_value=mock_parser_refitter,
            ),
        ):
            refitter = EngineRefitter(engine_file_path, onnx_file_path)
            result = refitter.load_and_refit()

        mock_runtime.deserialize_cuda_engine.assert_called_once()
        mock_parser_refitter.refit_from_file.assert_called_once_with(str(onnx_file_path))
        mock_refitter.refit_cuda_engine.assert_called_once()
        assert result is mock_engine

    def test_raises_when_deserialize_fails(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)
        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = None

        with patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=mock_runtime):
            refitter = EngineRefitter(engine_file_path, onnx_file_path)
            with pytest.raises(RuntimeError, match="Failed to deserialise"):
                refitter.load_and_refit()

    def test_raises_when_parser_refitter_fails(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)
        mock_engine, mock_runtime, mock_refitter, mock_parser_refitter = self._make_mock_trt()
        mock_parser_refitter.refit_from_file.return_value = False  # <- failure

        with (
            patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=mock_runtime),
            patch("app_v2.infrastructure.engine_refitter.trt.Refitter", return_value=mock_refitter),
            patch(
                "app_v2.infrastructure.engine_refitter.trt.OnnxParserRefitter",
                return_value=mock_parser_refitter,
            ),
        ):
            refitter = EngineRefitter(engine_file_path, onnx_file_path)
            with pytest.raises(RuntimeError, match="OnnxParserRefitter failed"):
                refitter.load_and_refit()

    def test_raises_when_refit_cuda_engine_fails(self, tmp_path: Path):
        engine_file_path = _make_engine_file(tmp_path)
        onnx_file_path = _make_onnx_file(tmp_path)
        mock_engine, mock_runtime, mock_refitter, mock_parser_refitter = self._make_mock_trt()
        mock_refitter.refit_cuda_engine.return_value = False  # <- failure

        with (
            patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=mock_runtime),
            patch("app_v2.infrastructure.engine_refitter.trt.Refitter", return_value=mock_refitter),
            patch(
                "app_v2.infrastructure.engine_refitter.trt.OnnxParserRefitter",
                return_value=mock_parser_refitter,
            ),
        ):
            refitter = EngineRefitter(engine_file_path, onnx_file_path)
            with pytest.raises(RuntimeError, match="refit_cuda_engine.*returned False"):
                refitter.load_and_refit()

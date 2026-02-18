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
    ep = tmp_path / engine_name
    ep.write_bytes(b"\x00\x01\x02\x03")
    return ep


def _make_onnx_file(tmp_path: Path, onnx_name: str = "model.onnx") -> Path:
    op = tmp_path / onnx_name
    op.write_bytes(b"\x08\x00")  # minimal fake ONNX bytes
    return op


# ---------------------------------------------------------------------------
# Tests: sidecar helpers
# ---------------------------------------------------------------------------


class TestSidecarHelpers:
    def test_write_and_read_sidecar(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)

        write_refit_sidecar(ep, op)
        sidecar = Path(str(ep) + REFIT_SIDECAR_SUFFIX)
        assert sidecar.exists(), "sidecar file should be created"
        result = get_onnx_path_for_engine(ep)
        assert result == op.resolve()

    def test_get_onnx_path_returns_none_when_no_sidecar(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        assert get_onnx_path_for_engine(ep) is None

    def test_get_onnx_path_returns_none_when_sidecar_points_to_missing_file(
        self, tmp_path: Path
    ):
        ep = _make_engine_file(tmp_path)
        sidecar = Path(str(ep) + REFIT_SIDECAR_SUFFIX)
        sidecar.write_text("/nonexistent/path/model.onnx")
        assert get_onnx_path_for_engine(ep) is None

    def test_sidecar_stores_absolute_path(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)
        write_refit_sidecar(ep, op)
        sidecar_text = Path(str(ep) + REFIT_SIDECAR_SUFFIX).read_text().strip()
        assert Path(sidecar_text).is_absolute()


# ---------------------------------------------------------------------------
# Tests: EngineRefitter constructor
# ---------------------------------------------------------------------------


class TestEngineRefitterInit:
    def test_explicit_onnx_path(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)
        r = EngineRefitter(ep, op)
        assert r.engine_path == ep
        assert r.onnx_path == op

    def test_auto_onnx_from_sidecar(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)
        write_refit_sidecar(ep, op)
        r = EngineRefitter(ep)
        assert r.onnx_path == op.resolve()

    def test_raises_when_no_onnx_and_no_sidecar(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        with pytest.raises(FileNotFoundError, match="No ONNX source found"):
            EngineRefitter(ep)


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
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)
        eng, rt, ref, pref = self._make_mock_trt()

        with (
            patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=rt),
            patch("app_v2.infrastructure.engine_refitter.trt.Refitter", return_value=ref),
            patch(
                "app_v2.infrastructure.engine_refitter.trt.OnnxParserRefitter",
                return_value=pref,
            ),
        ):
            r = EngineRefitter(ep, op)
            result = r.load_and_refit()

        rt.deserialize_cuda_engine.assert_called_once()
        pref.refit_from_file.assert_called_once_with(str(op))
        ref.refit_cuda_engine.assert_called_once()
        assert result is eng

    def test_raises_when_deserialize_fails(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)
        rt = MagicMock()
        rt.deserialize_cuda_engine.return_value = None

        with patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=rt):
            r = EngineRefitter(ep, op)
            with pytest.raises(RuntimeError, match="Failed to deserialise"):
                r.load_and_refit()

    def test_raises_when_parser_refitter_fails(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)
        eng, rt, ref, pref = self._make_mock_trt()
        pref.refit_from_file.return_value = False  # <- failure

        with (
            patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=rt),
            patch("app_v2.infrastructure.engine_refitter.trt.Refitter", return_value=ref),
            patch(
                "app_v2.infrastructure.engine_refitter.trt.OnnxParserRefitter",
                return_value=pref,
            ),
        ):
            r = EngineRefitter(ep, op)
            with pytest.raises(RuntimeError, match="OnnxParserRefitter failed"):
                r.load_and_refit()

    def test_raises_when_refit_cuda_engine_fails(self, tmp_path: Path):
        ep = _make_engine_file(tmp_path)
        op = _make_onnx_file(tmp_path)
        eng, rt, ref, pref = self._make_mock_trt()
        ref.refit_cuda_engine.return_value = False  # <- failure

        with (
            patch("app_v2.infrastructure.engine_refitter.trt.Runtime", return_value=rt),
            patch("app_v2.infrastructure.engine_refitter.trt.Refitter", return_value=ref),
            patch(
                "app_v2.infrastructure.engine_refitter.trt.OnnxParserRefitter",
                return_value=pref,
            ),
        ):
            r = EngineRefitter(ep, op)
            with pytest.raises(RuntimeError, match="refit_cuda_engine.*returned False"):
                r.load_and_refit()

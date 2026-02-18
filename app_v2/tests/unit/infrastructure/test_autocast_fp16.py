from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Guard: requires onnx + tensorrt
_onnx_available = True
_trt_available = True
try:
    import onnx
except ImportError:
    _onnx_available = False
try:
    import tensorrt  # noqa: F401
except ImportError:
    _trt_available = False


# ---------------------------------------------------------------------------
# Helper: build a minimal FP32 ONNX model for testing
# ---------------------------------------------------------------------------

def _make_minimal_fp32_onnx(output_path: Path) -> None:
    """Create a tiny valid FP32 ONNX model: input → Relu → output."""
    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 64, 64])
    relu = helper.make_node("Relu", ["images"], ["output"])
    graph = helper.make_graph([relu], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(output_path))


# ---------------------------------------------------------------------------
# Tests for the conversion helpers (onnxconverter-common path)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _onnx_available, reason="onnx not available")
class TestConvertOnnxToFp16:
    def test_output_file_created(self, tmp_path: Path) -> None:
        onnx_in = tmp_path / "model_fp32.onnx"
        onnx_out = tmp_path / "model_fp16.onnx"
        _make_minimal_fp32_onnx(onnx_in)

        # Import the function under test
        import prepare_yolo_autocast_fp16 as m
        m.convert_onnx_to_fp16(onnx_in, onnx_out)

        assert onnx_out.exists(), "FP16 ONNX output file should be created"
        assert onnx_out.stat().st_size > 0

    def test_output_is_valid_onnx(self, tmp_path: Path) -> None:
        import onnx as onnx_module
        onnx_in = tmp_path / "model_fp32.onnx"
        onnx_out = tmp_path / "model_fp16.onnx"
        _make_minimal_fp32_onnx(onnx_in)

        import prepare_yolo_autocast_fp16 as m
        m.convert_onnx_to_fp16(onnx_in, onnx_out)

        loaded = onnx_module.load(str(onnx_out))
        onnx_module.checker.check_model(loaded)  # raises if invalid

    def test_keeps_io_types_as_float32(self, tmp_path: Path) -> None:
        """With keep_io_types=True the graph I/O should remain FLOAT (Cast nodes inserted)."""
        import onnx as onnx_module
        onnx_in = tmp_path / "model_fp32.onnx"
        onnx_out = tmp_path / "model_fp16.onnx"
        _make_minimal_fp32_onnx(onnx_in)

        import prepare_yolo_autocast_fp16 as m
        m.convert_onnx_to_fp16(onnx_in, onnx_out)

        model = onnx_module.load(str(onnx_out))
        FLOAT = onnx_module.TensorProto.FLOAT
        for inp in model.graph.input:
            assert inp.type.tensor_type.elem_type == FLOAT, (
                f"Input {inp.name} should remain FLOAT32, got {inp.type.tensor_type.elem_type}"
            )
        for out in model.graph.output:
            assert out.type.tensor_type.elem_type == FLOAT, (
                f"Output {out.name} should remain FLOAT32, got {out.type.tensor_type.elem_type}"
            )


# ---------------------------------------------------------------------------
# Tests for CLI / main() argument parsing
# ---------------------------------------------------------------------------

class TestAutocastCliArgs:
    def test_missing_onnx_exits(self, tmp_path: Path) -> None:
        """Script should exit(1) if input ONNX file doesn't exist."""
        import prepare_yolo_autocast_fp16 as m
        with patch.object(sys, "argv", ["prog", "--onnx-in", str(tmp_path / "missing.onnx")]):
            with pytest.raises(SystemExit) as exc_info:
                m.main()
        assert exc_info.value.code == 1

    def test_help_exits_zero(self) -> None:
        import prepare_yolo_autocast_fp16  # noqa: F401
        with patch.object(sys, "argv", ["prog", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                import prepare_yolo_autocast_fp16 as m
                m.main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Tests for build_trt_engine (mocked TRT)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _onnx_available, reason="onnx not available")
class TestBuildTrtEngine:
    def _make_mock_trt(self, tmp_path: Path):
        """Return a mock tensorrt module that writes fake engine bytes."""
        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = "WARNING"
        mock_trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED = 1
        mock_trt.MemoryPoolType.WORKSPACE = "WORKSPACE"

        fake_serialized = b"fake_engine_content_" * 10
        mock_builder = MagicMock()
        mock_builder.build_serialized_network.return_value = fake_serialized
        mock_builder.create_optimization_profile.return_value = MagicMock()
        mock_builder.create_builder_config.return_value = MagicMock()
        mock_builder.create_network.return_value = MagicMock()

        # Parser that succeeds
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.Builder.return_value = mock_builder
        mock_trt.OnnxParser.return_value = mock_parser

        return mock_trt, fake_serialized

    def test_engine_file_written(self, tmp_path: Path) -> None:
        onnx_in = tmp_path / "model.onnx"
        _make_minimal_fp32_onnx(onnx_in)
        engine_out = tmp_path / "model.engine"

        mock_trt, expected_bytes = self._make_mock_trt(tmp_path)

        with patch("prepare_yolo_autocast_fp16.trt", mock_trt):
            import prepare_yolo_autocast_fp16 as m
            m.build_trt_engine(
                onnx_path=onnx_in,
                engine_out=engine_out,
                workspace_gb=1,
                timing_cache_path=None,
            )

        assert engine_out.exists()
        assert engine_out.read_bytes() == expected_bytes

    def test_timing_cache_used_when_provided(self, tmp_path: Path) -> None:
        onnx_in = tmp_path / "model.onnx"
        _make_minimal_fp32_onnx(onnx_in)
        engine_out = tmp_path / "model.engine"
        cache_path = tmp_path / "timing_cache.bin"

        mock_trt, _ = self._make_mock_trt(tmp_path)
        mock_cache_mgr = MagicMock()

        with (
            patch("prepare_yolo_autocast_fp16.trt", mock_trt),
            patch("prepare_yolo_autocast_fp16.TimingCacheManager", return_value=mock_cache_mgr),
        ):
            import prepare_yolo_autocast_fp16 as m
            m.build_trt_engine(
                onnx_path=onnx_in,
                engine_out=engine_out,
                workspace_gb=1,
                timing_cache_path=cache_path,
            )

        mock_cache_mgr.load_into_config.assert_called_once()
        mock_cache_mgr.save_from_config.assert_called_once()

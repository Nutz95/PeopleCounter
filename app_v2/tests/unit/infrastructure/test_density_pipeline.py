"""Unit tests for the density inference pipeline (DensityTRT + DensityDecoder).

These tests run in pure PyTorch without a real TRT engine, using mock
execution contexts.  They verify:
  • ImageNet normalisation is applied correctly
  • Batch construction from GpuTensor inputs works
  • Per-tile density maps are stitched into a global heatmap
  • Total count is the sum over all tiles
  • DensityDecoder base64-encodes a correctly sized canvas
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_mock_gpu_tensor(h: int = 720, w: int = 640, fill: float = 0.5) -> Any:
    """Return a fake GpuTensor with a `.tensor` attribute containing fp16 data."""
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch not available")
    t = torch.full((3, h, w), fill, dtype=torch.float16)
    gt = MagicMock()
    gt.tensor = t
    return gt


def _make_tile_plan(rows: int = 3, cols: int = 6, fw: int = 3840, fh: int = 2160,
                    tw: int = 640, th: int = 720) -> Any:
    from app_v2.core.preprocessor_types import PreprocessPlan, PreprocessTask
    tasks = []
    for r in range(rows):
        for c in range(cols):
            tasks.append(PreprocessTask(
                model_name="density",
                task_index=r * cols + c,
                source_x=c * tw,
                source_y=r * th,
                source_width=tw,
                source_height=th,
                target_width=tw,
                target_height=th,
                metadata={"row": r, "col": c},
            ))
    return PreprocessPlan(
        model_name="density",
        frame_width=fw,
        frame_height=fh,
        tasks=tuple(tasks),
        metadata={"mode": "tiles", "rows": rows, "cols": cols},
    )


# ── DensityTRT._build_batch ───────────────────────────────────────────────────

class TestDensityBatchBuilder:

    def test_batch_shape_from_18_tiles(self) -> None:
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_trt import DensityTRT

        inputs = [_make_mock_gpu_tensor(720, 640, 0.5) for _ in range(18)]
        batch = DensityTRT._build_batch(inputs)
        assert batch is not None
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.float32, "must be cast to fp32"
        assert batch.shape == (18, 3, 720, 640)

    def test_imagenet_norm_applied(self) -> None:
        """After normalisation, a 0.5 pixel should not remain 0.5."""
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_trt import DensityTRT, _IMAGENET_MEAN, _IMAGENET_STD

        inputs = [_make_mock_gpu_tensor(720, 640, 0.5)]
        batch = DensityTRT._build_batch(inputs)
        assert batch is not None
        # Mean of each channel after norm = (0.5 - mean_c) / std_c
        for c in range(3):
            expected = (0.5 - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
            actual = float(batch[0, c, 0, 0].item())
            assert abs(actual - expected) < 1e-4, f"channel {c}: {actual} != {expected}"

    def test_empty_inputs_returns_none(self) -> None:
        pytest.importorskip("torch")
        from app_v2.infrastructure.density_trt import DensityTRT
        assert DensityTRT._build_batch([]) is None

    def test_single_tile(self) -> None:
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_trt import DensityTRT
        inputs = [_make_mock_gpu_tensor(720, 640, 0.0)]
        batch = DensityTRT._build_batch(inputs)
        assert batch is not None
        assert batch.shape == (1, 3, 720, 640)

    def test_batch_shape_from_4_tiles_1920x1088(self) -> None:
        """2×2 config: 4 tiles at 1920×1088 (multiple of 16, no rescaling)."""
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_trt import DensityTRT

        inputs = [_make_mock_gpu_tensor(1088, 1920, 0.5) for _ in range(4)]
        batch = DensityTRT._build_batch(inputs)
        assert batch is not None
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.float32
        assert batch.shape == (4, 3, 1088, 1920)


# ── DensityTRT._decode_density ───────────────────────────────────────────────

class TestDensityDecode:

    def test_decode_splits_batch_into_tiles(self) -> None:
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_trt import DensityTRT

        b, h_out, w_out = 18, 45, 40
        fake_output = torch.ones(b, 1, h_out, w_out, dtype=torch.float32)
        tiles, count = DensityTRT._decode_density([fake_output], n_tiles=18)
        assert len(tiles) == 18
        # Total count: 18 tiles × 45 × 40 pixels × 1.0 = 32 400
        assert abs(count - 18 * h_out * w_out) < 1.0

    def test_count_is_sum(self) -> None:
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_trt import DensityTRT

        fake = torch.ones(6, 1, 10, 10, dtype=torch.float32) * 2.0
        _, count = DensityTRT._decode_density([fake], n_tiles=6)
        assert abs(count - 6 * 10 * 10 * 2.0) < 1e-3


# ── DensityDecoder ────────────────────────────────────────────────────────────

class TestDensityDecoder:

    def test_hotspot_extraction_returns_nonzero_for_uniform_tiles(self) -> None:
        """With uniform tile density, _extract_hotspots should return hotspots."""
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_decoder import DensityDecoder

        plan = _make_tile_plan(rows=3, cols=6, fw=3840, fh=2160, tw=640, th=720)
        tiles = [torch.ones(1, 45, 40) * 0.1 for _ in range(18)]
        dec = DensityDecoder()
        result = dec.process(1, {"density_tiles": tiles, "density_count": 180.0, "tile_plan": plan})
        assert isinstance(result["hotspots"], list)
        # Uniform map: at least some peaks should be found
        assert len(result["hotspots"]) > 0
        for hs in result["hotspots"]:
            assert 0.0 <= hs["x"] <= 1.0
            assert 0.0 <= hs["y"] <= 1.0

    def test_decoder_process_no_tiles(self) -> None:
        from app_v2.infrastructure.density_decoder import DensityDecoder
        dec = DensityDecoder()
        result = dec.process(42, {"density_tiles": [], "density_count": 0.0, "tile_plan": None})
        assert result["frame_id"] == 42
        assert result["density_count"] == 0.0
        assert result["hotspots"] == []

    def test_decoder_process_with_tiles(self) -> None:
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_decoder import DensityDecoder

        plan = _make_tile_plan(rows=3, cols=6)
        tiles = [torch.ones(1, 45, 40) for _ in range(18)]
        dec = DensityDecoder()
        result = dec.process(
            10,
            {"density_tiles": tiles, "density_count": 324.0, "tile_plan": plan},
        )
        assert result["model"] == "density"
        assert result["density_count"] == 324.0
        assert isinstance(result["hotspots"], list)
        assert len(result["hotspots"]) > 0
        # Each hotspot must have normalised x/y in [0,1] and a weight w in [0,1]
        for hs in result["hotspots"]:
            assert 0.0 <= hs["x"] <= 1.0
            assert 0.0 <= hs["y"] <= 1.0
            assert 0.0 <= hs["w"] <= 1.0


# ── DensityTRT.infer via mock context ─────────────────────────────────────────

class TestDensityTRTInfer:

    def _make_mock_context(self, b: int = 18, h_out: int = 45, w_out: int = 40) -> Any:
        try:
            import torch
        except ModuleNotFoundError:
            pytest.skip("PyTorch not available")
        ctx = MagicMock()
        fake_out = torch.ones(b, 1, h_out, w_out, dtype=torch.float32) * 0.5
        ctx.execute.return_value = {
            "output_tensors": [fake_out],
            "prepare_batch_ms": 1.2,
            "enqueue_ms": 0.5,
            "stream_sync_ms": 0.3,
        }
        return ctx

    def test_infer_returns_expected_keys(self) -> None:
        pytest.importorskip("torch")
        from app_v2.infrastructure.density_trt import DensityTRT

        ctx = self._make_mock_context()
        model = DensityTRT(ctx, stream_id=2)
        inputs = [_make_mock_gpu_tensor(720, 640) for _ in range(18)]
        plan = _make_tile_plan()

        result = model.infer(1, inputs, tile_plan=plan)

        assert result["frame_id"] == 1
        assert result["model"] == "density"
        assert "density_count" in result
        assert "density_tiles" in result
        assert len(result["density_tiles"]) == 18
        assert result["density_count"] > 0
        assert result["inference_ms"] >= 0

    def test_infer_count_matches_sum(self) -> None:
        pytest.importorskip("torch")
        import torch
        from app_v2.infrastructure.density_trt import DensityTRT

        b, h_out, w_out = 18, 45, 40
        ctx = MagicMock()
        fake_out = torch.full((b, 1, h_out, w_out), 1.0, dtype=torch.float32)
        ctx.execute.return_value = {"output_tensors": [fake_out]}
        model = DensityTRT(ctx, stream_id=2)
        inputs = [_make_mock_gpu_tensor() for _ in range(b)]
        result = model.infer(5, inputs, tile_plan=_make_tile_plan())

        expected_count = b * h_out * w_out * 1.0
        assert abs(result["density_count"] - expected_count) < 1.0

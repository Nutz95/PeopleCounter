from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

# Guard: tests that need torch+trt run inside Docker; unit tests mock everything
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="torch not available")


def _make_trt_mocks(batch_size: int = 1, height: int = 64, width: int = 64):
    """Return (mock_trt, mock_context, mock_engine) for tests."""
    import torch

    mock_trt = MagicMock()
    mock_trt.DataType.FLOAT = "FLOAT"
    mock_trt.DataType.HALF = "HALF"
    mock_trt.DataType.INT8 = "INT8"
    mock_trt.DataType.INT32 = "INT32"

    mock_engine = MagicMock()
    mock_engine.get_tensor_dtype.return_value = mock_trt.DataType.FLOAT

    mock_context = MagicMock()
    # Simulated output shape: [batch, 84, 8400] (YOLO-style)
    mock_context.get_tensor_shape.return_value = (batch_size, 84, 8400)
    mock_context.execute_async_v3.return_value = True

    return mock_trt, mock_context, mock_engine


def _make_stream() -> "torch.cuda.Stream":
    """Return a real CUDA stream for capture tests."""
    import torch
    return torch.cuda.Stream()


def _make_input(batch_size: int = 1) -> "torch.Tensor":
    import torch
    return torch.zeros(batch_size, 3, 64, 64, device="cuda", dtype=torch.float16)


# ---------------------------------------------------------------------------
# Unit tests (mocked torch.cuda.CUDAGraph)
# ---------------------------------------------------------------------------

class TestCudaGraphCacheCapture:
    def test_first_call_triggers_capture(self) -> None:
        import torch
        from app_v2.infrastructure.cuda_graph_cache import CudaGraphCache

        mock_trt, mock_context, mock_engine = _make_trt_mocks(batch_size=1)
        stream = _make_stream()
        input_t = _make_input(batch_size=1)

        mock_graph = MagicMock()
        mock_graph.replay = MagicMock()

        with (
            patch("app_v2.infrastructure.cuda_graph_cache.trt", mock_trt),
            patch("torch.cuda.CUDAGraph", return_value=mock_graph),
        ):
            cache = CudaGraphCache(mock_context, stream, mock_engine, warmup_iters=1)
            result = cache.execute(input_t, ["output0"], "images")

        assert result, "Should return non-empty outputs"
        assert cache.is_captured(1), "Batch-1 graph should be captured"

    def test_second_call_uses_replay_not_capture(self) -> None:
        """After first capture, graph.replay() is called on every execute() call.
        First call: capture → replay (1 replay).
        Second call: replay only (2 replays total).
        torch.cuda.CUDAGraph constructor is called only once (no re-capture).
        """
        import torch
        from app_v2.infrastructure.cuda_graph_cache import CudaGraphCache

        mock_trt, mock_context, mock_engine = _make_trt_mocks(batch_size=1)
        stream = _make_stream()
        input_t = _make_input(batch_size=1)

        graph_ctor_count = {"n": 0}
        replay_count = {"n": 0}

        real_graph_cls = torch.cuda.CUDAGraph

        class SpyGraph(real_graph_cls):
            def __init__(self):
                super().__init__()
                graph_ctor_count["n"] += 1
            def replay(self):
                super().replay()
                replay_count["n"] += 1

        with (
            patch("app_v2.infrastructure.cuda_graph_cache.trt", mock_trt),
            patch("torch.cuda.CUDAGraph", side_effect=SpyGraph),
        ):
            cache = CudaGraphCache(mock_context, stream, mock_engine, warmup_iters=1)
            cache.execute(input_t, ["output0"], "images")  # capture + replay
            cache.execute(input_t, ["output0"], "images")  # replay only

        assert graph_ctor_count["n"] == 1, "Graph should be captured only once"
        assert replay_count["n"] == 2, "replay() called once per execute() call"

    def test_separate_graphs_per_batch_size(self) -> None:
        import torch
        from app_v2.infrastructure.cuda_graph_cache import CudaGraphCache

        mock_trt, _, mock_engine = _make_trt_mocks()
        stream = _make_stream()

        graphs_created = {"n": 0}
        original_graph_cls = torch.cuda.CUDAGraph

        class CountingGraph:
            def __init__(self):
                graphs_created["n"] += 1
                self._g = original_graph_cls()
            def __enter__(self):
                return self._g.__enter__()
            def __exit__(self, *a):
                return self._g.__exit__(*a)
            def replay(self):
                self._g.replay()

        mock_context = MagicMock()
        mock_context.execute_async_v3.return_value = True
        mock_context.get_tensor_shape.side_effect = lambda n: (1, 84, 8400)

        with patch("app_v2.infrastructure.cuda_graph_cache.trt", mock_trt):
            cache = CudaGraphCache(mock_context, stream, mock_engine, warmup_iters=1)
            for bs in [1, 2, 4]:
                t = _make_input(batch_size=bs)
                mock_context.get_tensor_shape.return_value = (bs, 84, 8400)
                cache.execute(t, ["output0"], "images")

        assert len(cache._entries) == 3, "Should have one graph per batch size"

    def test_capture_failure_falls_back_to_direct(self) -> None:
        import torch
        from app_v2.infrastructure.cuda_graph_cache import CudaGraphCache

        mock_trt, mock_context, mock_engine = _make_trt_mocks(batch_size=1)
        stream = _make_stream()
        input_t = _make_input(batch_size=1)

        with (
            patch("app_v2.infrastructure.cuda_graph_cache.trt", mock_trt),
            patch("torch.cuda.CUDAGraph", side_effect=RuntimeError("graph capture unavailable")),
        ):
            cache = CudaGraphCache(mock_context, stream, mock_engine, warmup_iters=1)
            result = cache.execute(input_t, ["output0"], "images")

        assert 1 in cache._capture_failed
        # Direct path should still have attempted set_tensor_address + execute_async_v3
        mock_context.execute_async_v3.assert_called()


class TestCudaGraphCacheRelease:
    def test_release_clears_entries(self) -> None:
        import torch
        from app_v2.infrastructure.cuda_graph_cache import CudaGraphCache

        mock_trt, mock_context, mock_engine = _make_trt_mocks(batch_size=1)
        stream = _make_stream()
        input_t = _make_input(batch_size=1)

        with (
            patch("app_v2.infrastructure.cuda_graph_cache.trt", mock_trt),
        ):
            cache = CudaGraphCache(mock_context, stream, mock_engine, warmup_iters=1)
            cache.execute(input_t, ["output0"], "images")
            assert cache.is_captured(1)
            cache.release()
            assert not cache.is_captured(1)
            assert len(cache._capture_failed) == 0


# ---------------------------------------------------------------------------
# Integration with TensorRTExecutionContext (mocked engine, real CUDA graph)
# ---------------------------------------------------------------------------

class TestTensorRTExecutionContextCudaGraphs:
    def test_cuda_graphs_flag_activates_cache(self) -> None:
        import torch
        from app_v2.infrastructure.tensorrt_execution_context import TensorRTExecutionContext

        mock_loader = MagicMock()
        mock_loader.load.return_value = {"path": "fake.engine"}
        mock_loader.engine = None  # Engine disabled → trt_enabled=False
        mock_pool = MagicMock()

        ctx = TensorRTExecutionContext(
            engine_loader=mock_loader,
            stream_pool=mock_pool,
            options={"enabled": False, "cuda_graphs": True},
        )
        assert ctx._cuda_graphs_enabled

    def test_cuda_graphs_disabled_by_default(self) -> None:
        from app_v2.infrastructure.tensorrt_execution_context import TensorRTExecutionContext

        mock_loader = MagicMock()
        mock_loader.load.return_value = {"path": "fake.engine"}
        mock_loader.engine = None
        mock_pool = MagicMock()

        ctx = TensorRTExecutionContext(
            engine_loader=mock_loader,
            stream_pool=mock_pool,
            options={"enabled": False},
        )
        assert not ctx._cuda_graphs_enabled

    def test_release_graphs_clears_cache(self) -> None:
        from app_v2.infrastructure.tensorrt_execution_context import TensorRTExecutionContext

        mock_loader = MagicMock()
        mock_loader.load.return_value = {"path": "fake.engine"}
        mock_loader.engine = None
        mock_pool = MagicMock()

        ctx = TensorRTExecutionContext(
            engine_loader=mock_loader,
            stream_pool=mock_pool,
            options={"enabled": False, "cuda_graphs": True},
        )
        # Inject a fake cache
        fake_cache = MagicMock()
        ctx._cuda_graph_cache = fake_cache

        ctx.release_graphs()

        fake_cache.release.assert_called_once()
        assert ctx._cuda_graph_cache is None

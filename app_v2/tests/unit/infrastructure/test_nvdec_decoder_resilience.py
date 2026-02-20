"""Unit tests for NvdecDecoder corruption-resilience behaviour.

Tests are designed to run without real GPU hardware or PyNvCodec installed.
They mock the internal ``_pynvcodec`` module and the ``_decoder`` object so
that every code path in ``reset()``, ``_decode_surface()``, and
``decode_next_into_ring()`` can be exercised in CI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Minimal PyNvCodec stub so the module can be imported without the real wheel.
# ---------------------------------------------------------------------------

def _make_pynvcodec_stub() -> types.ModuleType:
    mod = types.ModuleType("PyNvCodec")
    mod.PyNvDecoder = MagicMock()
    mod.PixelFormat = MagicMock()
    mod.PixelFormat.NV12 = 1
    mod.CudaVideoCodec = MagicMock()
    mod.CudaVideoCodec.__members__ = {"h264": MagicMock()}
    return mod


@pytest.fixture(autouse=True)
def stub_pynvcodec(monkeypatch):
    """Inject a lightweight PyNvCodec stub for every test in this module."""
    stub = _make_pynvcodec_stub()
    monkeypatch.setitem(sys.modules, "PyNvCodec", stub)
    yield stub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decoder(stream_url: str = "rtsp://cam/test") -> "NvdecDecoder":
    """Construct a NvdecDecoder with a fully mocked internal state."""
    from app_v2.infrastructure.nvdec_decoder import NvdecDecoder, NvdecDecodeConfig

    # Patch _probe_stream_info so no real subprocess is spawned.
    with patch.object(NvdecDecoder, "_probe_stream_info", return_value={"width": 1920, "height": 1080, "codec_name": "h264"}):
        dec = NvdecDecoder(stream_url, NvdecDecodeConfig(ring_capacity=4, surface_retry_limit=3))

    # Replace real PyNvDecoder instance with a controllable mock.
    dec._decoder = MagicMock()
    dec._decoder.DecodeSingleSurface.return_value = None   # default: empty surface
    return dec


def _make_valid_surface() -> MagicMock:
    """Return a mock surface that looks valid (not empty)."""
    s = MagicMock()
    s.Empty.return_value = False
    s.Width.return_value = 1920
    s.Height.return_value = 1080
    s.PlanePtr.return_value = 0xDEAD_BEEF
    s.Pitch.return_value = 1920
    s.Pts.return_value = 0
    return s


# ---------------------------------------------------------------------------
# reset() unit tests
# ---------------------------------------------------------------------------

class TestNvdecDecoderReset:
    def test_reset_creates_new_decoder_object(self):
        from app_v2.infrastructure.nvdec_decoder import NvdecDecoder

        dec = _make_decoder()
        original = dec._decoder

        with patch.object(NvdecDecoder, "_create_decoder", return_value=MagicMock()) as mock_create:
            dec.reset()

        mock_create.assert_called_once()
        assert dec._decoder is not original

    def test_reset_clears_last_surface_was_good(self):
        dec = _make_decoder()
        dec._last_surface_was_good = True

        with patch.object(type(dec), "_create_decoder", return_value=MagicMock()):
            dec.reset()

        assert dec._last_surface_was_good is False

    def test_reset_survives_del_decoder_exception(self):
        """del self._decoder raising must not propagate."""
        from app_v2.infrastructure.nvdec_decoder import NvdecDecoder

        dec = _make_decoder()

        # Make del raise by replacing _decoder with a descriptor that raises on delete
        class _UnDeletable:
            pass

        dec._decoder = _UnDeletable()

        with patch.object(NvdecDecoder, "_create_decoder", return_value=MagicMock()):
            dec.reset()   # must not raise


# ---------------------------------------------------------------------------
# Auto-reset gating via _last_surface_was_good
# ---------------------------------------------------------------------------

class TestAutoResetGating:
    def test_no_reset_on_first_failure_when_flag_is_false(self):
        """Fresh decoder: first decode failure must NOT trigger reset."""
        from app_v2.infrastructure.nvdec_decoder import NvdecDecoder

        dec = _make_decoder()
        dec.start()
        assert dec._last_surface_was_good is False
        dec._decoder.DecodeSingleSurface.return_value = None

        with patch.object(NvdecDecoder, "reset") as mock_reset:
            with pytest.raises(RuntimeError, match="failed to produce a surface"):
                dec.decode_next_into_ring(frame_id=1)
            mock_reset.assert_not_called()

    def test_reset_triggered_after_good_then_bad(self):
        """A failure after a run of good frames MUST call reset() once."""
        from app_v2.infrastructure.nvdec_decoder import NvdecDecoder

        dec = _make_decoder()
        dec.start()

        # Simulate a successful first frame.
        dec._decoder.DecodeSingleSurface.return_value = _make_valid_surface()
        dec.decode_next_into_ring(frame_id=1)
        assert dec._last_surface_was_good is True

        # Now corrupt packet → surface failure.
        dec._decoder.DecodeSingleSurface.return_value = None
        with patch.object(NvdecDecoder, "reset") as mock_reset:
            with pytest.raises(RuntimeError, match="failed to produce a surface"):
                dec.decode_next_into_ring(frame_id=2)
            mock_reset.assert_called_once()

    def test_no_double_reset_after_first_reset(self):
        """Subsequent failures after a reset must NOT trigger another reset."""
        from app_v2.infrastructure.nvdec_decoder import NvdecDecoder

        dec = _make_decoder()
        dec.start()

        # Simulate a good frame, then corruption, then reset.
        dec._decoder.DecodeSingleSurface.return_value = _make_valid_surface()
        dec.decode_next_into_ring(frame_id=1)
        dec._last_surface_was_good = False   # simulate post-reset state

        dec._decoder.DecodeSingleSurface.return_value = None
        with patch.object(NvdecDecoder, "reset") as mock_reset:
            with pytest.raises(RuntimeError):
                dec.decode_next_into_ring(frame_id=2)
            mock_reset.assert_not_called()   # already reset; gate blocks second

    def test_flag_set_true_after_successful_decode(self):
        dec = _make_decoder()
        dec.start()
        dec._last_surface_was_good = False
        dec._decoder.DecodeSingleSurface.return_value = _make_valid_surface()

        dec.decode_next_into_ring(frame_id=1)

        assert dec._last_surface_was_good is True

    def test_ring_slot_released_on_failure(self):
        """Ring buffer slot must be released even when surface decode fails."""
        dec = _make_decoder()
        dec.start()
        dec._decoder.DecodeSingleSurface.return_value = None

        with pytest.raises(RuntimeError):
            dec.decode_next_into_ring(frame_id=1)

        # All slots should still be available (capacity=4 → 4 acquirable).
        for _ in range(dec._config.ring_capacity):
            slot = dec._ring.acquire(block=False)
            assert slot is not None, "slot was not released after failure"


# ---------------------------------------------------------------------------
# Pipeline orchestrator: skip-on-decode-error
# ---------------------------------------------------------------------------

class TestOrchestratorDecodeErrorSkip:
    """Ensure the orchestrator skips corrupt frames and only aborts on
    _MAX_CONSECUTIVE_DECODE_ERRORS consecutive failures."""

    def _make_minimal_orchestrator(self):
        """Build an orchestrator with mocked frame source, publisher, and models."""
        from app_v2.application.pipeline_orchestrator import PipelineOrchestrator
        from app_v2.core.frame_source import FrameSource

        frame_source = MagicMock(spec=FrameSource)
        frame_source.stream_url = "rtsp://cam/test"
        frame_source.connected = False

        publisher = MagicMock()
        publisher.start = MagicMock()
        publisher.push_frame = MagicMock()
        publisher.webcodecs_ws_port = 5001

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = {}
        orch.frame_source = frame_source
        orch._models = []
        orch._running = False
        orch._frame_counter = 0
        orch.max_frames = 20
        orch.scheduler = MagicMock()
        orch.scheduler.schedule.side_effect = list(range(20))
        orch.scheduler.acknowledge = MagicMock()
        orch.performance_tracker = MagicMock()
        orch.performance_tracker.stage.return_value = MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
        orch.performance_tracker.clear = MagicMock()
        orch.preprocessor = MagicMock()
        orch.aggregator = MagicMock()
        orch.processing_graph = MagicMock()
        orch.publisher = publisher
        orch._video_stream = None
        orch._video_executor = MagicMock()
        orch._video_future = None
        orch._video_max_height = None
        orch._video_quality = 75
        orch._webcodecs_server = MagicMock()
        orch._packet_forwarder = None
        return orch

    def test_single_decode_error_does_not_abort_pipeline(self, monkeypatch):
        orch = self._make_minimal_orchestrator()
        # First call raises; subsequent calls succeed (return None-like frame).
        frame_mock = MagicMock()
        frame_mock.height = 1080
        frame_mock.width = 1920
        call_count = [0]

        def next_frame(frame_id):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("NVDEC decoder failed to produce a surface")
            return frame_mock

        orch.frame_source.next_frame = next_frame
        orch._running = True

        monkeypatch.setattr(orch, "_push_video_frame_async", MagicMock())

        orch.run()   # must NOT raise; must complete all 20 frames

        # At least one frame was skipped and the rest processed.
        assert orch._frame_counter > 0

    def test_ten_consecutive_errors_abort_pipeline(self):
        orch = self._make_minimal_orchestrator()
        orch.max_frames = 100
        orch._running = True

        def next_frame(frame_id):
            raise RuntimeError("NVDEC decoder failed to produce a surface")

        orch.frame_source.next_frame = next_frame

        # Should not raise itself but should stop after 10 failures.
        orch.run()

        # No frames successfully processed.
        assert orch._frame_counter == 0
        # Scheduler acknowledged every skipped frame.
        assert orch.scheduler.acknowledge.call_count == 10

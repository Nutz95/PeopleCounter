from __future__ import annotations

import concurrent.futures
import os
import sys
import threading
import time
from typing import Any

_PERF_LOG: bool = os.environ.get("PERF_LOG", "0").strip() not in ("", "0", "false", "no")

import torch

from app_v2.application.frame_scheduler import FrameScheduler
from app_v2.application.inference_stream_controller import InferenceStreamController
from app_v2.application.model_builder import ModelBuilder
from app_v2.application.processing_graph import ProcessingGraph
from app_v2.application.performance_tracker import PerformanceTracker
from app_v2.application.result_aggregator import ResultAggregator
from app_v2.config import load_pipeline_config
from app_v2.core.strategies import FusionStrategy, RawStreamFusionStrategy, SimpleFusionStrategy
from app_v2.core.frame_source import FrameSource
from app_v2.core.result_publisher import ResultPublisher
from app_v2.enums import FusionStrategyType
from app_v2.infrastructure.cuda_preprocessor import CudaPreprocessor
from app_v2.infrastructure.density_decoder import DensityDecoder
from app_v2.infrastructure.flask_server.server import FlaskStreamServer
from app_v2.infrastructure.nvdec_decoder import build_stream_open_opts
from app_v2.infrastructure.nvdec_packet_forwarder import NvdecPacketForwarder
from app_v2.infrastructure.stream_pool import SimpleStreamPool
from app_v2.infrastructure.webcodecs_server import WebCodecsServer
from app_v2.kernels.nv12_cuda_bridge import nv12_to_rgb_hwc_resized_cuda
from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning

# Buffer-cache slot ID for the video encoder's NV12 plane buffers inside
# nv12_to_rgb_hwc_resized_cuda.  Must not overlap with any preprocess stream
# ID defined in pipeline.yaml (currently 0–5).
_VIDEO_BUFFER_SLOT = 99


class PipelineOrchestrator:
    """Drives the NVDEC loop, preprocessors, inference contexts, and result fusion."""

    def __init__(
        self,
        frame_source: FrameSource,
        max_frames: int | None = None,
        publisher: ResultPublisher | None = None,
        fusion_strategy: FusionStrategy | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.config = config if config is not None else load_pipeline_config()
        self.scheduler = FrameScheduler()
        self.processing_graph = ProcessingGraph()
        self.preprocessor = CudaPreprocessor()
        self.stream_pool = SimpleStreamPool()
        self.inference_controller = InferenceStreamController(self.config)
        self.model_builder = ModelBuilder(self.config, self.inference_controller, self.stream_pool)
        if fusion_strategy is not None:
            self.fusion_strategy = fusion_strategy
        else:
            configured_strategy_name = self.config.get("fusion_strategy", "ASYNC_OVERLAY")
            try:
                configured_strategy_type = FusionStrategyType(configured_strategy_name)
            except ValueError:
                configured_strategy_type = FusionStrategyType.ASYNC_OVERLAY
            if configured_strategy_type == FusionStrategyType.RAW_STREAM_WITH_METADATA:
                self.fusion_strategy = RawStreamFusionStrategy()
            else:
                self.fusion_strategy = SimpleFusionStrategy(strategy_type=configured_strategy_type)
        self.publisher = publisher or FlaskStreamServer(initial_config=self.config)
        self.aggregator = ResultAggregator(self.fusion_strategy, self.publisher)
        self.performance_tracker = PerformanceTracker()
        self.max_frames = max_frames
        self._models = self.model_builder.build_models()
        self._active_sync_mode = "async" if isinstance(self.fusion_strategy, RawStreamFusionStrategy) else "sync"
        self._apply_decoder_runtime_policy(self._active_sync_mode)
        self._density_decoder = DensityDecoder(
            min_peak_weight=float(self.config.get("density", {}).get("min_peak_weight", 0.05)),
            nms_kernel=int(self.config.get("density", {}).get("nms_kernel", 3)),
            export_hotspots_for_ui=bool(self.config.get("density", {}).get("export_hotspots_for_ui", True)),
        )
        self._frame_counter = 0
        self._running = False
        # Dedicated CUDA stream for NV12→RGB conversion + resize (separate from
        # inference streams so it doesn't stall preprocess/inference pipelines).
        self._video_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        # Dedicated CUDA stream for NVJPEG encode in the background thread.
        # Keeps NVJPEG kernels on their own stream so they never contend with
        # _video_stream (NV12→RGB) or the inference streams.
        self._nvjpeg_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        # Single-worker executor: NVJPEG encode + push to SSE clients.
        # If previous encode is still running when next frame arrives → drop silently.
        self._video_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nvjpeg"
        )
        self._video_future: concurrent.futures.Future[None] | None = None
        # Video stream config: read from pipeline.yaml [video_stream] section.
        video_stream_config = self.config.get("video_stream") or {}
        self._video_max_height: int | None = video_stream_config.get("max_height") or None
        self._video_quality: int = int(video_stream_config.get("quality", 75))
        self._video_encode_backend: str = str(video_stream_config.get("backend", "auto")).strip().lower()
        # Zero-drop stash: when encoder is busy the latest RGB tensor is kept
        # here; _on_video_encode_done auto-submits it when the slot is free.
        self._pending_chw: torch.Tensor | None = None
        self._pending_enc_event: torch.cuda.Event | None = None
        self._pending_frame_id: int | None = None
        # Guard to ensure a stashed frame is never encoded before inference
        # has published hotspots for that same frame.
        self._pending_ready_to_encode: bool = False
        self._encode_running: bool = False
        # When sync mode is active the browser displays MJPEG instead of WebCodecs,
        # so NVJPEG must run even when a WebCodecs WS client is still connected.
        self._force_mjpeg: bool = False
        # When a WebCodecs client is connected the browser renders video via the
        # zero-encode WebSocket path, so MJPEG/NVJPEG encoding is unnecessary.
        # We track nothing here — the check is done live in _push_video_frame_async.
        self._pending_frame_lock: threading.Lock = threading.Lock()
        self._video_metrics_lock: threading.Lock = threading.Lock()
        self._video_last_encode_ms: float = 0.0
        self._video_last_wait_event_ms: float = 0.0
        self._video_last_backend_code: float = 0.0  # 0 none, 1 cpu, 2 nvjpeg
        self._video_last_cpu_copy_ms: float = 0.0
        self._video_last_push_ms: float = 0.0
        self._video_last_jpeg_kb: float = 0.0
        self._video_last_hotspot_lookup_ms: float = 0.0
        self._video_last_hotspot_lookup_mode_code: float = 0.0  # 0 none, 1 exact, 2 fallback, 3 miss
        self._video_hotspot_lookup_exact_count: int = 0
        self._video_hotspot_lookup_fallback_count: int = 0
        self._video_hotspot_lookup_miss_count: int = 0
        self._video_encode_jobs_count: int = 0
        self._video_encode_errors_count: int = 0

        # ── WebCodecs zero-encode path ──────────────────────────────────
        # PyFFmpegDemuxer opens a second connection to the same stream URL and
        # forwards raw H.264/H.265 compressed packets (Annex B) directly to the
        # browser via a pure-Python WebSocket server (port 4999).
        # The browser uses the WebCodecs VideoDecoder API to render the stream
        # without any server-side re-encoding.
        # Falls back to MJPEG transparently if PyNvCodec is unavailable or the
        # source codec is not H.264/H.265.
        webcodecs_ws_port = int(video_stream_config.get("webcodecs_ws_port", WebCodecsServer.DEFAULT_PORT))
        self._webcodecs_server = WebCodecsServer(port=webcodecs_ws_port)
        self._packet_forwarder: NvdecPacketForwarder | None = self._build_packet_forwarder(webcodecs_ws_port)
        # NOTE: publisher.webcodecs_ws_port is updated in run() AFTER start() so it
        # always reflects the actually-bound port (may differ from _ws_port when the
        # preferred port was already in use and the server fell back to a free port).

        log_info(LogChannel.GLOBAL, "PipelineOrchestrator components initialized")

    def run(self) -> None:
        log_info(LogChannel.GLOBAL, f"Starting app_v2 pipeline with config {self.config}")
        log_info(LogChannel.GLOBAL, "Frame scheduling will track frame IDs until fusion completes.")
        # Start WebCodecs server BEFORE Flask so port 4999 is already listening
        # when the browser first loads the page.  Without this, the browser can
        # attempt WebSocket connections during the frame_source.connect() window
        # (1-3 s) and accumulate onerror events → spurious error banner.
        self._webcodecs_server.start()
        if isinstance(self.publisher, FlaskStreamServer):
            self.publisher.webcodecs_ws_port = self._webcodecs_server.port
        self._start_publisher()
        self.frame_source.connect()
        if self._packet_forwarder is not None:
            self._packet_forwarder.start()
        self.preprocessor.configure(self.config)
        # Ensure the fusion strategy waits for ALL models before publishing
        # a frame — otherwise we get one SSE event per model (3×) and the
        # ring-buffer release hooks fire before downstream models finish.
        # Not applicable for RAW_STREAM_WITH_METADATA which publishes per-model.
        if isinstance(self.fusion_strategy, SimpleFusionStrategy) and self._models:
            self.fusion_strategy.expected_count = len(self._models)
        self._running = True
        # Consecutive decode-error counter: skip bad frames up to this limit
        # before giving up (handles NVDEC "HW decoder faced error" after a corrupt
        # packet — the decoder self-heals after a few buffering-phase misses).
        max_consecutive_decode_errors = 10
        consecutive_decode_errors = 0
        previous_loop_done_ns: int = 0  # tracks end of previous iteration for gap measurement
        try:
            while self._should_continue():
                frame_id = self.scheduler.schedule(None)
                perf_loop_start_ns = time.perf_counter_ns() if _PERF_LOG else 0

                # ── NVDEC decode ────────────────────────────────────────────
                try:
                    with self.performance_tracker.stage(frame_id, "nvdec"):
                        frame = self.frame_source.next_frame(frame_id)
                except RuntimeError as decode_exc:
                    consecutive_decode_errors += 1
                    log_warning(
                        LogChannel.GLOBAL,
                        f"Frame {frame_id} skipped — decode error "
                        f"({consecutive_decode_errors}/{max_consecutive_decode_errors}): {decode_exc}",
                    )
                    self.scheduler.acknowledge(frame_id)
                    self.performance_tracker.clear(frame_id)
                    if consecutive_decode_errors >= max_consecutive_decode_errors:
                        raise RuntimeError(
                            f"Aborting after {max_consecutive_decode_errors} "
                            "consecutive decode failures"
                        ) from decode_exc
                    continue
                consecutive_decode_errors = 0
                perf_after_nvdec_ns = time.perf_counter_ns() if _PERF_LOG else 0
                # ────────────────────────────────────────────────────────────

                # ── Video: NV12→RGB conversion (before inference, async GPU) ──────
                # NV12→RGB runs on _video_stream NOW so it overlaps with inference.
                # The JPEG encode+push is dispatched AFTER inference so that
                # hotspots are available and passed directly — no cache lookup race.
                self._prepare_video_frame(frame, frame_id)
                perf_after_video_dispatch_ns = time.perf_counter_ns() if _PERF_LOG else 0
                # ──────────────────────────────────────────────────────────────────

                output = self.preprocessor.build_output(frame_id, frame)
                self.aggregator.attach_telemetry(frame_id, output.telemetry)
                if output.telemetry is not None:
                    output.telemetry.add_metrics(self._snapshot_video_encode_metrics())
                perf_src_wait_ms = 0.0
                perf_src_age_ms = 0.0
                perf_src_copy_sync_ms = 0.0
                if output.telemetry is not None:
                    tele_snapshot = output.telemetry.snapshot()
                    perf_src_wait_ms = float(tele_snapshot.get("frame_source_wait_latest_ms", 0.0) or 0.0)
                    perf_src_age_ms = float(tele_snapshot.get("frame_source_age_at_consume_ms", 0.0) or 0.0)
                    perf_src_copy_sync_ms = float(tele_snapshot.get("frame_source_copy_sync_ms", 0.0) or 0.0)
                perf_after_preprocess_ns = time.perf_counter_ns() if _PERF_LOG else 0

                if isinstance(self.fusion_strategy, RawStreamFusionStrategy):
                    # RAW_STREAM_WITH_METADATA: release the NVDEC ring-buffer slot
                    # and preprocessed tensor pool *immediately* — before inference.
                    # The video frame has already been handed off to _video_stream
                    # (GPU async), so we only need to wait for that work to finish
                    # before releasing.  Inference uses CPU-side tensor slices that
                    # remain valid after the GPU release.
                    if self._video_stream is not None:
                        self._video_stream.synchronize()
                    output.release_all()
                else:
                    self.aggregator.attach_release_hook(frame_id, output.release_all)
                perf_after_video_sync_ns = time.perf_counter_ns() if _PERF_LOG else 0
                perf_before_infer_ns = perf_after_video_sync_ns
                perf_after_infer_ns = perf_after_video_sync_ns
                perf_before_collect_ns = perf_after_video_sync_ns
                perf_after_collect_ns = perf_after_video_sync_ns
                perf_trt_sync_ms = 0.0
                perf_trt_prepare_ms = 0.0
                perf_decode_ms = 0.0
                perf_flatten_ms_total = 0.0
                perf_register_ms_total = 0.0
                perf_collect_ms_total = 0.0
                for model in self._models:
                    perf_flatten_start_ns = time.perf_counter_ns()
                    processed = output.flatten_inputs(model.name)
                    perf_flatten_ms_total += (time.perf_counter_ns() - perf_flatten_start_ns) / 1_000_000.0
                    tile_plan = output.plans.get(model.name)
                    with self.performance_tracker.stage(frame_id, model.name):
                        if _PERF_LOG:
                            perf_before_infer_ns = time.perf_counter_ns()
                        prediction = model.infer(
                            frame_id,
                            processed,
                            preprocess_events=list(output.cuda_events.values()),
                            tile_plan=tile_plan,
                        )
                        if isinstance(prediction, dict):
                            perf_trt_sync_ms = float(prediction.get("stream_sync_ms", perf_trt_sync_ms) or perf_trt_sync_ms)
                            perf_trt_prepare_ms = float(prediction.get("prepare_batch_ms", perf_trt_prepare_ms) or perf_trt_prepare_ms)
                            perf_decode_ms = float(prediction.get("decode_ms", perf_decode_ms) or perf_decode_ms)
                        perf_after_infer_ns = time.perf_counter_ns() if _PERF_LOG else 0
                        if isinstance(prediction, dict):
                            prediction["_inference_done_ns"] = int(time.time_ns())
                            # DM-Count density: convert raw GPU tiles → base64 heatmap
                            if model.name == "density":
                                prediction = self._density_decoder.process(frame_id, prediction)
                        perf_register_start_ns = time.perf_counter_ns()
                        self.processing_graph.register(model.name, {"frame_id": frame_id})
                        perf_register_ms_total += (time.perf_counter_ns() - perf_register_start_ns) / 1_000_000.0
                        if _PERF_LOG:
                            perf_before_collect_ns = time.perf_counter_ns()
                        perf_collect_start_ns = time.perf_counter_ns()
                        self.aggregator.collect(frame_id, prediction)
                        perf_collect_ms_total += (time.perf_counter_ns() - perf_collect_start_ns) / 1_000_000.0
                        perf_after_collect_ns = time.perf_counter_ns() if _PERF_LOG else 0

                # ── Video encode dispatch (after inference, with fresh hotspots) ──
                # hotspots come directly from publisher cache that was just populated
                # by aggregator.collect() → publisher.publish() above.
                self._flush_video_encode(frame_id)
                # ──────────────────────────────────────────────────────────────────

                if output.telemetry is not None:
                    output.telemetry.add_metrics(
                        {
                            "orchestrator_flatten_ms": perf_flatten_ms_total,
                            "orchestrator_register_ms": perf_register_ms_total,
                            "orchestrator_collect_ms": perf_collect_ms_total,
                        }
                    )

                if _PERF_LOG:
                    perf_loop_done_ns = time.perf_counter_ns()
                    format_duration_ms = lambda start_ns, end_ns: f"{(end_ns - start_ns) / 1e6:.1f}"  # noqa: E731
                    modes_str = ",".join(m.name for m in self._models) or "pass"
                    frame_gap_ms = (
                        (perf_loop_start_ns - previous_loop_done_ns) / 1e6
                        if previous_loop_done_ns else 0.0
                    )
                    print(
                        f"[PERF] f={frame_id} mode={modes_str}"
                        f" gap={frame_gap_ms:.1f}"
                        f" nvdec={format_duration_ms(perf_loop_start_ns, perf_after_nvdec_ns)}"
                        f" vid_dispatch={format_duration_ms(perf_after_nvdec_ns, perf_after_video_dispatch_ns)}"
                        f" preproc={format_duration_ms(perf_after_video_dispatch_ns, perf_after_preprocess_ns)}"
                        f" vid_sync={format_duration_ms(perf_after_preprocess_ns, perf_after_video_sync_ns)}"
                        f" flat={format_duration_ms(perf_after_video_sync_ns, perf_before_infer_ns)}"
                        f" infer={format_duration_ms(perf_before_infer_ns, perf_after_infer_ns)}"
                        f" pre_collect={format_duration_ms(perf_after_infer_ns, perf_before_collect_ns)}"
                        f" collect={format_duration_ms(perf_before_collect_ns, perf_after_collect_ns)}"
                        f" src_wait={perf_src_wait_ms:.1f}"
                        f" src_age={perf_src_age_ms:.1f}"
                        f" src_copy_sync={perf_src_copy_sync_ms:.1f}"
                        f" trt_prepare={perf_trt_prepare_ms:.1f}"
                        f" trt_sync={perf_trt_sync_ms:.1f}"
                        f" decode={perf_decode_ms:.1f}"
                        f" infer+collect={format_duration_ms(perf_after_video_sync_ns, perf_loop_done_ns)}"
                        f" total={format_duration_ms(perf_loop_start_ns, perf_loop_done_ns)}ms",
                        file=sys.stderr, flush=True,
                    )
                    previous_loop_done_ns = perf_loop_done_ns

                # Passthrough mode: no models active — ring slot never released via
                # aggregator.collect() so we release it here immediately.
                # Wait for the video GPU work first (NV12→RGB kernel) to avoid a
                # race where the decoder overwrites the ring slot mid-conversion.
                if not self._models:
                    if not isinstance(self.fusion_strategy, RawStreamFusionStrategy):
                        if self._video_stream is not None:
                            self._video_stream.synchronize()
                        self.aggregator.discard_frame(frame_id)
                    # Flush the JPEG encode for frames with no inference (no hotspots).
                    self._flush_video_encode(frame_id)
                    # Emit a lightweight SSE heartbeat so the browser can count FPS
                    if isinstance(self.publisher, FlaskStreamServer):
                        self.publisher.publish_passthrough_frame(frame_id)

                self.scheduler.acknowledge(frame_id)
                self.performance_tracker.clear(frame_id)
                self._frame_counter += 1

                # ── Runtime mode change (applied between frames) ────────────
                if isinstance(self.publisher, FlaskStreamServer):
                    pending_mode = self.publisher.get_and_clear_pending_mode()
                    if pending_mode is not None:
                        self._apply_mode_change(pending_mode)
                    pending_sync_mode = self.publisher.get_and_clear_pending_sync_mode()
                    if pending_sync_mode is not None:
                        self._apply_sync_mode_change(pending_sync_mode)
                    pending_threshold = self.publisher.get_and_clear_pending_density_threshold()
                    if pending_threshold is not None:
                        self._density_decoder.min_peak_weight = pending_threshold
                    pending_crowd_conf = self.publisher.get_and_clear_pending_crowd_confidence()
                    if pending_crowd_conf is not None:
                        for model in self._models:
                            if model.name in ("crowd_global", "crowd_tiles") and hasattr(model, "_decoder"):
                                model._decoder.confidence_threshold = pending_crowd_conf
                    pending_video_backend = self.publisher.get_and_clear_pending_video_backend()
                    if pending_video_backend is not None:
                        self._video_encode_backend = pending_video_backend
                        self.publisher.set_active_video_backend(pending_video_backend)
                        log_info(LogChannel.GLOBAL, f"MJPEG encode backend switched to '{pending_video_backend}'")
        except StopIteration:
            log_info(LogChannel.GLOBAL, "Frame source signaled completion")
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"Pipeline aborted during iteration: {exc}")
        finally:
            self._running = False
            self.frame_source.disconnect()
            self._shutdown()

    def _should_continue(self) -> bool:
        if not self._running:
            return False
        if self.max_frames is None:
            return True
        return self._frame_counter < self.max_frames

    def _start_publisher(self) -> None:
        if not hasattr(self.publisher, "start"):
            return
        try:
            self.publisher.start()
            log_info(LogChannel.GLOBAL, f"FlaskStreamServer started on {self.publisher.host}:{self.publisher.port}")
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"FlaskStreamServer failed to start: {exc}")

    def _apply_mode_change(self, new_mode: str) -> None:
        """Hot-swap inference models between frames."""
        from app_v2.infrastructure.flask_server.mode_registry import _INFERENCE_MODES, _PREPROCESS_BRANCH_MAP

        mode_state = _INFERENCE_MODES.get(new_mode)
        if mode_state is None:
            log_warning(LogChannel.GLOBAL, f"Unknown mode '{new_mode}' — ignoring")
            return

        # Close existing models cleanly
        for model in self._models:
            try:
                model.close()
            except Exception as exc:
                log_warning(LogChannel.GLOBAL, f"Model close error during mode switch: {exc}")
        self._models = []

        # Update config in-place
        models_config = self.config.setdefault("models", {})
        preprocess_branches_config = self.config.setdefault("preprocess_branches", {})
        for model_name, enabled in mode_state.items():
            if model_name in models_config:
                models_config[model_name]["enabled"] = enabled
            branch_key = _PREPROCESS_BRANCH_MAP.get(model_name)
            if branch_key:
                preprocess_branches_config[branch_key] = enabled

        # Rebuild inference components
        self.inference_controller = InferenceStreamController(self.config)
        self.model_builder = ModelBuilder(self.config, self.inference_controller, self.stream_pool)
        self._models = self.model_builder.build_models()

        # Reconfigure preprocessor for the new branches
        self.preprocessor.configure(self.config)
        self._apply_decoder_runtime_policy(self._active_sync_mode)

        # Update fusion strategy expected model count
        if isinstance(self.fusion_strategy, SimpleFusionStrategy):
            self.fusion_strategy.expected_count = len(self._models)

        # Notify publisher
        if isinstance(self.publisher, FlaskStreamServer):
            self.publisher.set_active_mode(new_mode)
            self.publisher.update_available_modes(self.config)

        log_info(
            LogChannel.GLOBAL,
            f"Mode switched to '{new_mode}' — {len(self._models)} model(s) active",
        )

    def _apply_sync_mode_change(self, sync_mode: str) -> None:
        """Hot-swap fusion strategy between async and sync publication modes."""
        mode = (sync_mode or "").strip().lower()
        if mode not in ("async", "sync"):
            log_warning(LogChannel.GLOBAL, f"Unknown sync mode '{sync_mode}' — ignoring")
            return

        if mode == "async":
            self.fusion_strategy = RawStreamFusionStrategy()
            self.config["fusion_strategy"] = FusionStrategyType.RAW_STREAM_WITH_METADATA.value
            self._force_mjpeg = False
        else:
            strict = SimpleFusionStrategy(strategy_type=FusionStrategyType.STRICT_SYNC)
            strict.expected_count = len(self._models) if self._models else 1
            self.fusion_strategy = strict
            self.config["fusion_strategy"] = FusionStrategyType.STRICT_SYNC.value
            self._force_mjpeg = True

        # Keep aggregator aligned with the newly active strategy.
        self.aggregator.fusion_strategy = self.fusion_strategy
        self._active_sync_mode = mode
        self._apply_decoder_runtime_policy(mode)

        if isinstance(self.publisher, FlaskStreamServer):
            self.publisher.set_active_sync_mode(mode)

        log_info(
            LogChannel.GLOBAL,
            f"Synchronization mode switched to '{mode}' (fusion={self.fusion_strategy.strategy_type.value})",
        )

    def _apply_decoder_runtime_policy(self, sync_mode: str) -> None:
        """Apply low-latency decode policy according to active sync mode.

        Runtime sync mode must not silently change detection semantics.
        Decoder toggles are driven by model inference config only so that
        bbox overlays and counts stay consistent across async/sync transport.
        """
        for model in self._models:
            decoder = getattr(model, "_decoder", None)
            if decoder is None:
                continue

            model_name = str(getattr(model, "name", "")).lower()
            is_yolo_family = model_name.startswith("yolo") or model_name.startswith("crowd")
            if not is_yolo_family:
                continue

            inference_params = getattr(model, "_inference_params", {})
            if not isinstance(inference_params, dict):
                inference_params = {}

            if hasattr(decoder, "count_only_mode"):
                decoder.count_only_mode = bool(inference_params.get("count_only_mode", False))
            if hasattr(decoder, "person_summary_enabled"):
                decoder.person_summary_enabled = bool(inference_params.get("person_summary_enabled", False))
            if hasattr(decoder, "seg_mask_enabled"):
                decoder.seg_mask_enabled = bool(inference_params.get("seg_mask_enabled", False))

    def _shutdown(self) -> None:
        self._video_executor.shutdown(wait=False)
        if self._packet_forwarder is not None:
            self._packet_forwarder.stop()
        self._webcodecs_server.stop()
        for model in self._models:
            try:
                model.close()
            except Exception as exc:
                log_warning(LogChannel.GLOBAL, f"Model {model.name} closed with error: {exc}")

    def _prepare_video_frame(self, frame: Any, frame_id: int | None = None) -> None:
        """Step 1 of 2: NV12→RGB conversion on _video_stream (async, before inference).

        Records the CUDA event and stashes the CHW tensor.  Does NOT submit the
        JPEG encode job — that happens in _flush_video_encode() AFTER inference
        so that hotspots are available and passed directly without any cache race.
        """
        if self._webcodecs_server.has_clients() and not self._force_mjpeg:
            self._set_video_encode_metrics_idle()
            return
        if self._video_stream is None:
            self._set_video_encode_metrics_idle()
            return
        push_frame = getattr(self.publisher, "push_frame", None)
        if not callable(push_frame):
            return
        try:
            frame_height = int(getattr(frame, "height", 0))
            frame_width = int(getattr(frame, "width", 0))
            if self._video_max_height is not None and frame_height > self._video_max_height:
                resize_scale = self._video_max_height / frame_height
                target_h = self._video_max_height
                target_w = int(frame_width * resize_scale) & ~1
            else:
                target_h, target_w = frame_height, frame_width
            with torch.cuda.stream(self._video_stream):
                rgb_hwc = nv12_to_rgb_hwc_resized_cuda(
                    frame, target_h, target_w, stream_id=_VIDEO_BUFFER_SLOT
                )
                chw_uint8 = rgb_hwc.permute(2, 0, 1).contiguous()
                encode_input_chw = chw_uint8.clone()
            enc_event = torch.cuda.Event()
            enc_event.record(self._video_stream)
            with self._pending_frame_lock:
                self._pending_chw = encode_input_chw
                self._pending_enc_event = enc_event
                self._pending_frame_id = int(frame_id) if isinstance(frame_id, int) else None
                self._pending_ready_to_encode = False
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"Video frame NV12→RGB failed: {exc}")

    def _flush_video_encode(self, frame_id: int | None = None) -> None:
        """Step 2 of 2: submit JPEG encode+push AFTER inference.

        At this point publisher.publish() has already populated the hotspot cache
        for this frame_id, so get_hotspots_for_frame() returns the correct data.
        """
        push_frame = getattr(self.publisher, "push_frame", None)
        if not callable(push_frame):
            return
        with self._pending_frame_lock:
            encode_input_chw = self._pending_chw
            enc_event = self._pending_enc_event
            encode_frame_id = self._pending_frame_id
            if encode_input_chw is None or enc_event is None:
                # NV12→RGB was skipped (WebCodecs path or stream idle)
                return
            if frame_id is not None and encode_frame_id is not None and int(frame_id) != int(encode_frame_id):
                # Stash was replaced by a newer frame; wait for its own flush.
                return
            # Mark current stashed frame as publish-complete (hotspots ready).
            self._pending_ready_to_encode = True
            if self._encode_running:
                # Previous encode still running: keep the stash as-is,
                # _on_video_encode_done will pick it up when free.
                return
            self._encode_running = True
            self._pending_chw = None
            self._pending_enc_event = None
            self._pending_frame_id = None
            self._pending_ready_to_encode = False
        self._video_future = self._submit_video_encode(
            encode_input_chw,
            enc_event,
            push_frame,
            encode_frame_id,
        )
        self._video_future.add_done_callback(self._on_video_encode_done)

    def _on_video_encode_done(self, _future: concurrent.futures.Future) -> None:  # type: ignore[type-arg]
        """Callback: fired by the executor thread when an NVJPEG encode finishes.

        If a newer frame was stashed while the encoder was busy, it is submitted
        immediately — implementing the zero-drop strategy.
        """
        push_frame = getattr(self.publisher, "push_frame", None)
        with self._pending_frame_lock:
            pending_chw = self._pending_chw
            pending_event = self._pending_enc_event
            pending_frame_id = self._pending_frame_id
            pending_ready = self._pending_ready_to_encode
            if pending_chw is None or not callable(push_frame):
                # Nothing pending or publisher gone → encoder goes idle.
                self._encode_running = False
                self._pending_enc_event = None
                self._pending_frame_id = None
                self._pending_ready_to_encode = False
                return
            if not pending_ready:
                # Critical ordering guard: a frame prepared before inference
                # must never be encoded until _flush_video_encode marks it ready.
                self._encode_running = False
                return
            # Pop the stash and keep _encode_running = True.
            self._pending_chw = None
            self._pending_enc_event = None
            self._pending_frame_id = None
            self._pending_ready_to_encode = False

        try:
            new_future = self._submit_video_encode(pending_chw, pending_event, push_frame, pending_frame_id)
            new_future.add_done_callback(self._on_video_encode_done)
            with self._pending_frame_lock:
                self._video_future = new_future
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"Video stash re-submit failed: {exc}")
            with self._pending_frame_lock:
                self._encode_running = False

    def _submit_video_encode(
        self,
        chw_uint8: torch.Tensor,
        enc_event: torch.cuda.Event,
        push_frame: Any,
        frame_id: int | None,
    ) -> concurrent.futures.Future[None]:
        """Submit one JPEG encode job to the dedicated video executor.

        backend=auto:
                    - sync mode  -> NVJPEG when CUDA is available (lower end-to-end latency)
                    - async mode -> NVJPEG
                    - fallback to CPU JPEG only when CUDA is unavailable
        backend=cpu/nvjpeg force the chosen encoder.
        """
        backend = self._video_encode_backend
        if backend == "auto":
                        backend = "nvjpeg" if torch.cuda.is_available() else "cpu"
        if backend not in ("cpu", "nvjpeg"):
            log_warning(LogChannel.GLOBAL, f"Unknown video backend '{backend}', fallback to 'nvjpeg'")
            backend = "nvjpeg"
        if backend == "cpu":
            return self._video_executor.submit(
                PipelineOrchestrator._encode_and_push_cpujpeg,
                chw_uint8,
                enc_event,
                self._video_quality,
                push_frame,
                self._record_video_encode_metrics,
            )
        return self._video_executor.submit(
            PipelineOrchestrator._encode_and_push_nvjpeg,
            chw_uint8,
            enc_event,
            self._video_quality,
            push_frame,
            self._nvjpeg_stream,
            self._record_video_encode_metrics,
            self.publisher,  # Pass publisher for GPU hotspot rendering
            frame_id,
        )

    @staticmethod
    def _encode_and_push_nvjpeg(
        chw_uint8: torch.Tensor,
        enc_event: torch.cuda.Event,
        quality: int,
        push_frame: Any,
        nvjpeg_stream: "torch.cuda.Stream | None",
        metrics_callback: Any | None = None,
        publisher: Any | None = None,
        frame_id: int | None = None,
    ) -> None:
        """Background thread: [3×H×W uint8 CUDA] → (GPU hotspot render) → NVJPEG bytes → MJPEG clients.

        Hotspots are rendered onto the frame on GPU before JPEG encoding for
        ultra-fast dense-scene annotation (no CPU/GPU sync, pure parallel work).
        
        NVJPEG runs on a dedicated stream (``nvjpeg_stream``) so it never blocks
        or is blocked by the NV12→RGB stream or the inference streams.
        """
        try:
            perf_start_ns = time.perf_counter_ns()
            enc_event.synchronize()  # CPU-side wait: data is ready in GPU memory
            perf_after_wait_ns = time.perf_counter_ns()

            # GPU hotspot rendering (if enabled and hotspots available)
            hotspot_lookup_mode_code = 0.0
            hotspot_lookup_start_ns = time.perf_counter_ns()
            if publisher is not None:
                try:
                    renderer = publisher.get_gpu_hotspot_renderer()
                    # Prefer frame-aligned hotspots; fallback to latest if that
                    # specific frame is not cached yet.
                    lookup_frame_id = int(frame_id) if isinstance(frame_id, int) else -1
                    hotspots = publisher.get_hotspots_for_frame(lookup_frame_id)
                    has_exact = False
                    is_empty_exact = False
                    if isinstance(hotspots, torch.Tensor):
                        is_empty_exact = hotspots.numel() == 0
                    elif isinstance(hotspots, list):
                        is_empty_exact = len(hotspots) == 0
                    else:
                        is_empty_exact = True
                    if is_empty_exact:
                        hotspots = publisher.get_hotspots_for_frame(-1)
                        hotspot_lookup_mode_code = 2.0
                    else:
                        has_exact = True
                        hotspot_lookup_mode_code = 1.0
                    has_hotspots = False
                    if isinstance(hotspots, torch.Tensor):
                        has_hotspots = hotspots.numel() > 0
                    elif isinstance(hotspots, list):
                        has_hotspots = len(hotspots) > 0
                    if hotspot_lookup_mode_code == 2.0 and not has_hotspots:
                        hotspot_lookup_mode_code = 3.0
                    if has_exact and not has_hotspots:
                        hotspot_lookup_mode_code = 3.0
                    if has_hotspots:
                        chw_uint8 = renderer.draw_hotspots_on_frame(
                            chw_uint8,
                            hotspots,
                            metrics_callback=metrics_callback,
                        )
                except Exception as exc:
                    import sys as _sys
                    print(f"[GPU Hotspot Render] failed (non-fatal): {exc}", file=_sys.stderr, flush=True)
            hotspot_lookup_done_ns = time.perf_counter_ns()

            perf_after_render_ns = time.perf_counter_ns()
            import torchvision.io as tvio

            # Encode on dedicated NVJPEG stream, then explicitly wait for stream
            # completion before touching the output on CPU to avoid any partial
            # read/copy race across CUDA streams.
            if nvjpeg_stream is not None:
                with torch.cuda.stream(nvjpeg_stream):
                    buf = tvio.encode_jpeg(chw_uint8, quality=quality)  # NVJPEG (CUDA → CUDA)
                nvjpeg_stream.synchronize()
            else:
                buf = tvio.encode_jpeg(chw_uint8, quality=quality)
                torch.cuda.current_stream().synchronize()

            perf_after_encode_ns = time.perf_counter_ns()

            jpeg_np = buf.cpu().numpy()
            jpeg_bytes = jpeg_np.tobytes()
            perf_after_cpu_copy_ns = time.perf_counter_ns()

            # Defensive check: reject obviously corrupted bitstreams.
            # JPEG must start with SOI (FFD8) and end with EOI (FFD9).
            if len(jpeg_bytes) < 4 or jpeg_bytes[0:2] != b"\xff\xd8" or jpeg_bytes[-2:] != b"\xff\xd9":
                raise RuntimeError("NVJPEG produced invalid JPEG markers")

            push_frame(jpeg_bytes)
            perf_after_push_ns = time.perf_counter_ns()
            if callable(metrics_callback):
                metrics_callback(
                    {
                        "video_encode_backend_code": 2.0,
                        "video_encode_last_ms": (perf_after_push_ns - perf_start_ns) / 1_000_000.0,
                        "video_encode_wait_event_ms": (perf_after_wait_ns - perf_start_ns) / 1_000_000.0,
                        "video_hotspot_lookup_ms": (hotspot_lookup_done_ns - hotspot_lookup_start_ns) / 1_000_000.0,
                        "video_hotspot_lookup_mode_code": hotspot_lookup_mode_code,
                        "video_encode_gpu_hotspot_render_ms": (perf_after_render_ns - perf_after_wait_ns) / 1_000_000.0,
                        "video_encode_kernel_ms": (perf_after_encode_ns - perf_after_render_ns) / 1_000_000.0,
                        "video_encode_cpu_copy_ms": (perf_after_cpu_copy_ns - perf_after_encode_ns) / 1_000_000.0,
                        "video_encode_push_ms": (perf_after_push_ns - perf_after_cpu_copy_ns) / 1_000_000.0,
                        "video_jpeg_kb": len(jpeg_bytes) / 1024.0,
                        "video_encode_error": 0.0,
                    }
                )
        except Exception as _nvjpeg_exc:
            import sys as _sys
            print(f"[NVJPEG] encode failed: {type(_nvjpeg_exc).__name__}: {_nvjpeg_exc}",
                  file=_sys.stderr, flush=True)
            if callable(metrics_callback):
                metrics_callback({"video_encode_backend_code": 2.0, "video_encode_error": 1.0})

    @staticmethod
    def _encode_and_push_cpujpeg(
        chw_uint8: torch.Tensor,
        enc_event: torch.cuda.Event,
        quality: int,
        push_frame: Any,
        metrics_callback: Any | None = None,
    ) -> None:
        """Background thread: CUDA RGB tensor → CPU JPEG bytes.

        This path removes NVJPEG kernels from the shared GPU when MJPEG is used
        as the sync display transport. We still pay one GPU→CPU copy, but the
        main pipeline thread stays fully async and the JPEG compression work is
        isolated on the worker thread / CPU.
        """
        try:
            perf_start_ns = time.perf_counter_ns()
            enc_event.synchronize()
            perf_after_wait_ns = time.perf_counter_ns()
            import torchvision.io as tvio

            cpu_tensor = chw_uint8.cpu()
            perf_after_cpu_copy_ns = time.perf_counter_ns()
            buf = tvio.encode_jpeg(cpu_tensor, quality=quality)
            jpeg_bytes = bytes(buf.numpy())
            perf_after_encode_ns = time.perf_counter_ns()
            push_frame(jpeg_bytes)
            perf_after_push_ns = time.perf_counter_ns()
            if callable(metrics_callback):
                metrics_callback(
                    {
                        "video_encode_backend_code": 1.0,
                        "video_encode_last_ms": (perf_after_push_ns - perf_start_ns) / 1_000_000.0,
                        "video_encode_wait_event_ms": (perf_after_wait_ns - perf_start_ns) / 1_000_000.0,
                        "video_encode_kernel_ms": (perf_after_encode_ns - perf_after_cpu_copy_ns) / 1_000_000.0,
                        "video_encode_cpu_copy_ms": (perf_after_cpu_copy_ns - perf_after_wait_ns) / 1_000_000.0,
                        "video_encode_push_ms": (perf_after_push_ns - perf_after_encode_ns) / 1_000_000.0,
                        "video_jpeg_kb": len(jpeg_bytes) / 1024.0,
                        "video_encode_error": 0.0,
                    }
                )
        except Exception as _cpujpeg_exc:
            import sys as _sys
            print(f"[CPUJPEG] encode failed: {type(_cpujpeg_exc).__name__}: {_cpujpeg_exc}",
                  file=_sys.stderr, flush=True)
            if callable(metrics_callback):
                metrics_callback({"video_encode_backend_code": 1.0, "video_encode_error": 1.0})

    def _record_video_encode_metrics(self, metrics: dict[str, float]) -> None:
        with self._video_metrics_lock:
            self._video_encode_jobs_count += 1
            self._video_last_backend_code = float(metrics.get("video_encode_backend_code", self._video_last_backend_code))
            if "video_encode_last_ms" in metrics:
                self._video_last_encode_ms = float(metrics["video_encode_last_ms"])
            if "video_encode_wait_event_ms" in metrics:
                self._video_last_wait_event_ms = float(metrics["video_encode_wait_event_ms"])
            if "video_encode_cpu_copy_ms" in metrics:
                self._video_last_cpu_copy_ms = float(metrics["video_encode_cpu_copy_ms"])
            if "video_encode_push_ms" in metrics:
                self._video_last_push_ms = float(metrics["video_encode_push_ms"])
            if "video_jpeg_kb" in metrics:
                self._video_last_jpeg_kb = float(metrics["video_jpeg_kb"])
            if "video_hotspot_lookup_ms" in metrics:
                self._video_last_hotspot_lookup_ms = float(metrics["video_hotspot_lookup_ms"])
            if "video_hotspot_lookup_mode_code" in metrics:
                self._video_last_hotspot_lookup_mode_code = float(metrics["video_hotspot_lookup_mode_code"])
                if self._video_last_hotspot_lookup_mode_code == 1.0:
                    self._video_hotspot_lookup_exact_count += 1
                elif self._video_last_hotspot_lookup_mode_code == 2.0:
                    self._video_hotspot_lookup_fallback_count += 1
                elif self._video_last_hotspot_lookup_mode_code == 3.0:
                    self._video_hotspot_lookup_miss_count += 1
            if float(metrics.get("video_encode_error", 0.0)) > 0.0:
                self._video_encode_errors_count += 1

    def _set_video_encode_metrics_idle(self) -> None:
        """Mark encode metrics as idle when no JPEG encode path is executed."""
        with self._video_metrics_lock:
            self._video_last_backend_code = 0.0
            self._video_last_encode_ms = 0.0
            self._video_last_wait_event_ms = 0.0
            self._video_last_cpu_copy_ms = 0.0
            self._video_last_push_ms = 0.0
            self._video_last_jpeg_kb = 0.0
            self._video_last_hotspot_lookup_ms = 0.0
            self._video_last_hotspot_lookup_mode_code = 0.0

    def _snapshot_video_encode_metrics(self) -> dict[str, float]:
        with self._video_metrics_lock:
            snapshot = {
                "video_encode_last_ms": self._video_last_encode_ms,
                "video_encode_wait_event_ms": self._video_last_wait_event_ms,
                "video_encode_cpu_copy_ms": self._video_last_cpu_copy_ms,
                "video_encode_push_ms": self._video_last_push_ms,
                "video_jpeg_kb": self._video_last_jpeg_kb,
                "video_hotspot_lookup_ms": self._video_last_hotspot_lookup_ms,
                "video_hotspot_lookup_mode_code": self._video_last_hotspot_lookup_mode_code,
                "video_hotspot_lookup_exact": float(self._video_hotspot_lookup_exact_count),
                "video_hotspot_lookup_fallback": float(self._video_hotspot_lookup_fallback_count),
                "video_hotspot_lookup_miss": float(self._video_hotspot_lookup_miss_count),
                "video_encode_jobs": float(self._video_encode_jobs_count),
                "video_encode_errors": float(self._video_encode_errors_count),
                "video_backend_code": self._video_last_backend_code,
            }
        with self._pending_frame_lock:
            snapshot["video_encode_inflight"] = 1.0 if self._encode_running else 0.0
            snapshot["video_encode_stashed"] = 1.0 if self._pending_chw is not None else 0.0
        return snapshot

    def _build_packet_forwarder(self, ws_port: int) -> NvdecPacketForwarder | None:
        """Construct a NvdecPacketForwarder if the frame source exposes a stream URL."""
        # Traverse common attribute paths to find the raw stream URL.
        stream_url: str | None = None
        for attr_path in (
            ("stream_url",),
            ("_decoder", "stream_url"),
            ("_source", "stream_url"),
        ):
            current_object = self.frame_source
            try:
                for attr in attr_path:
                    current_object = getattr(current_object, attr)
                if isinstance(current_object, str) and current_object:
                    stream_url = current_object
                    break
            except AttributeError:
                continue

        if not stream_url:
            log_warning(LogChannel.GLOBAL, "WebCodecs packet forwarder: no stream URL found on frame_source — skipping")
            return None

        # Reuse the same HTTP/RTSP input opts used by NvdecDecoder so the raw
        # WebCodecs demux path and the inference decode path behave identically
        # on reconnect (notably RTSP-over-TCP for the MediaMTX bridge).
        stream_open_options = build_stream_open_opts(stream_url)

        log_info(LogChannel.GLOBAL, f"WebCodecs packet forwarder configured for {stream_url} → ws port {ws_port}")
        return NvdecPacketForwarder(stream_url, self._webcodecs_server, decoder_opts=stream_open_options)

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import torch

from app_v2.application.frame_scheduler import FrameScheduler
from app_v2.application.inference_stream_controller import InferenceStreamController
from app_v2.application.model_builder import ModelBuilder
from app_v2.application.processing_graph import ProcessingGraph
from app_v2.application.performance_tracker import PerformanceTracker
from app_v2.application.result_aggregator import ResultAggregator
from app_v2.config import load_pipeline_config
from app_v2.core.fusion_strategy import FusionStrategy, SimpleFusionStrategy
from app_v2.core.frame_source import FrameSource
from app_v2.core.result_publisher import ResultPublisher
from app_v2.infrastructure.cuda_preprocessor import CudaPreprocessor
from app_v2.infrastructure.flask_stream_server import FlaskStreamServer
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
    ) -> None:
        self.frame_source = frame_source
        self.config = load_pipeline_config()
        self.scheduler = FrameScheduler()
        self.processing_graph = ProcessingGraph()
        self.preprocessor = CudaPreprocessor()
        self.stream_pool = SimpleStreamPool()
        self.inference_controller = InferenceStreamController(self.config)
        self.model_builder = ModelBuilder(self.config, self.inference_controller, self.stream_pool)
        self.fusion_strategy = fusion_strategy or SimpleFusionStrategy()
        self.publisher = publisher or FlaskStreamServer()
        self.aggregator = ResultAggregator(self.fusion_strategy, self.publisher)
        self.performance_tracker = PerformanceTracker()
        self.max_frames = max_frames
        self._models = self.model_builder.build_models()
        self._frame_counter = 0
        self._running = False
        # Dedicated CUDA stream for NV12→RGB conversion + resize (separate from
        # inference streams so it doesn't stall preprocess/inference pipelines).
        self._video_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        # Single-worker executor: NVJPEG encode + push to SSE clients.
        # If previous encode is still running when next frame arrives → drop silently.
        self._video_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nvjpeg"
        )
        self._video_future: concurrent.futures.Future[None] | None = None
        # Video stream config: read from pipeline.yaml [video_stream] section.
        _vcfg = self.config.get("video_stream") or {}
        self._video_max_height: int | None = _vcfg.get("max_height") or None
        self._video_quality: int = int(_vcfg.get("quality", 75))

        # ── WebCodecs zero-encode path ──────────────────────────────────
        # PyFFmpegDemuxer opens a second connection to the same stream URL and
        # forwards raw H.264/H.265 compressed packets (Annex B) directly to the
        # browser via a pure-Python WebSocket server (port 5001).
        # The browser uses the WebCodecs VideoDecoder API to render the stream
        # without any server-side re-encoding.
        # Falls back to MJPEG transparently if PyNvCodec is unavailable or the
        # source codec is not H.264/H.265.
        _ws_port = int(_vcfg.get("webcodecs_ws_port", WebCodecsServer.DEFAULT_PORT))
        self._webcodecs_server = WebCodecsServer(port=_ws_port)
        self._packet_forwarder: NvdecPacketForwarder | None = self._build_packet_forwarder(_ws_port)
        # Publish the WebSocket port to the Flask template.
        if isinstance(self.publisher, FlaskStreamServer):
            self.publisher.webcodecs_ws_port = _ws_port

        log_info(LogChannel.GLOBAL, "PipelineOrchestrator components initialized")

    def run(self) -> None:
        log_info(LogChannel.GLOBAL, f"Starting app_v2 pipeline with config {self.config}")
        log_info(LogChannel.GLOBAL, "Frame scheduling will track frame IDs until fusion completes.")
        self._start_publisher()
        self.frame_source.connect()
        # Start WebCodecs server + packet forwarder after the frame source is
        # connected so the RTSP URL is resolved and the camera is reachable.
        self._webcodecs_server.start()
        if self._packet_forwarder is not None:
            self._packet_forwarder.start()
        self.preprocessor.configure(self.config)
        self._running = True
        try:
            while self._should_continue():
                frame_id = self.scheduler.schedule(None)
                with self.performance_tracker.stage(frame_id, "nvdec"):
                    frame = self.frame_source.next_frame(frame_id)

                # ── Video encode dispatch ───────────────────────────────────
                # Submitted immediately after NVDEC decode, BEFORE inference.
                # NV12→RGB→resize runs on _video_stream (GPU, async from CPU).
                # NVJPEG encode runs in background thread, parallel with inference.
                self._push_video_frame_async(frame)
                # ───────────────────────────────────────────────────────────

                output = self.preprocessor.build_output(frame_id, frame)
                self.aggregator.attach_telemetry(frame_id, output.telemetry)
                self.aggregator.attach_release_hook(frame_id, output.release_all)
                for model in self._models:
                    processed = output.flatten_inputs(model.name)
                    with self.performance_tracker.stage(frame_id, model.name):
                        prediction = model.infer(frame_id, processed, preprocess_events=list(output.cuda_events.values()))
                        if isinstance(prediction, dict):
                            prediction["_inference_done_ns"] = int(time.time_ns())
                        self.processing_graph.register(model.name, {"frame_id": frame_id})
                        self.aggregator.collect(frame_id, prediction)
                self.scheduler.acknowledge(frame_id)
                self.performance_tracker.clear(frame_id)
                self._frame_counter += 1
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

    def _push_video_frame_async(self, frame: Any) -> None:
        """Encode the raw NV12 GpuFrame as JPEG and push it to the MJPEG feed.

        Called immediately after NVDEC decode — BEFORE inference.  All GPU work
        runs on ``_video_stream``, fully parallel with the YOLO inference pipeline.

        Conversion order: NV12 → bilinear resize in YUV space → RGB HWC uint8.
        This avoids allocating the full-resolution RGB intermediate (~25 MB for
        4K) — see :func:`nv12_to_rgb_hwc_resized_cuda` for details.

        Drop strategy: if the background thread is still busy with the previous
        frame, drop the incoming frame silently so the main loop is never blocked.

        Output resolution:
          - ``video_stream.max_height: null`` in pipeline.yaml → native camera
            resolution (e.g. 4K for a 4K source).
          - ``video_stream.max_height: N`` → height capped at N pixels, width
            scaled proportionally and rounded to an even number.

        Encoding: NVJPEG via ``torchvision.io.encode_jpeg`` (CUDA tensor input).
        Quality controlled by ``video_stream.quality`` in pipeline.yaml.
        """
        push_frame = getattr(self.publisher, "push_frame", None)
        if not callable(push_frame):
            return
        if self._video_stream is None:
            return

        # Drop silently if previous encode is still in progress.
        if self._video_future is not None and not self._video_future.done():
            return

        try:
            h = int(getattr(frame, "height", 0))
            w = int(getattr(frame, "width", 0))
            if self._video_max_height is not None and h > self._video_max_height:
                scale = self._video_max_height / h
                target_h = self._video_max_height
                target_w = int(w * scale) & ~1  # keep even for JPEG chroma sub-sampling
            else:
                target_h, target_w = h, w

            with torch.cuda.stream(self._video_stream):
                # NV12 → resize in YUV space → RGB HWC uint8 at target resolution.
                # nv12_to_rgb_hwc_resized_cuda uses stream_id=_VIDEO_BUFFER_SLOT to
                # keep its plane-buffer cache separate from all preprocess streams.
                rgb_hwc = nv12_to_rgb_hwc_resized_cuda(
                    frame, target_h, target_w, stream_id=_VIDEO_BUFFER_SLOT
                )  # [target_h, target_w, 3] uint8 CUDA
                chw_uint8 = rgb_hwc.permute(2, 0, 1).contiguous()  # [3, th, tw] uint8

            # Record event so background thread waits for GPU ops before NVJPEG.
            enc_event = torch.cuda.Event()
            enc_event.record(self._video_stream)

            self._video_future = self._video_executor.submit(
                PipelineOrchestrator._encode_and_push_nvjpeg,
                chw_uint8,
                enc_event,
                self._video_quality,
                push_frame,
            )
        except Exception as exc:
            log_warning(LogChannel.GLOBAL, f"Video frame submit failed: {exc}")

    @staticmethod
    def _encode_and_push_nvjpeg(
        chw_uint8: torch.Tensor,
        enc_event: torch.cuda.Event,
        quality: int,
        push_frame: Any,
    ) -> None:
        """Background thread: [3×H×W uint8 CUDA] → NVJPEG bytes → MJPEG clients."""
        try:
            enc_event.synchronize()
            import torchvision.io as tvio
            buf = tvio.encode_jpeg(chw_uint8, quality=quality)  # NVJPEG (CUDA → CUDA)
            push_frame(bytes(buf.cpu().numpy()))
        except Exception:
            pass

    def _build_packet_forwarder(self, ws_port: int) -> NvdecPacketForwarder | None:
        """Construct a NvdecPacketForwarder if the frame source exposes a stream URL."""
        # Traverse common attribute paths to find the raw stream URL.
        url: str | None = None
        for attr_path in (
            ("stream_url",),
            ("_decoder", "stream_url"),
            ("_source", "stream_url"),
        ):
            obj = self.frame_source
            try:
                for attr in attr_path:
                    obj = getattr(obj, attr)
                if isinstance(obj, str) and obj:
                    url = obj
                    break
            except AttributeError:
                continue

        if not url:
            log_warning(LogChannel.GLOBAL, "WebCodecs packet forwarder: no stream URL found on frame_source — skipping")
            return None

        # Reuse the same HTTP/RTSP decoder opts used by NvdecDecoder.
        opts: dict = {}
        url_lower = url.lower()
        if url_lower.startswith("http://") or url_lower.startswith("https://"):
            opts = {
                "probesize": "5000000",
                "analyzeduration": "10000000",
                "reconnect": "1",
                "reconnect_streamed": "1",
                "reconnect_delay_max": "2",
            }

        log_info(LogChannel.GLOBAL, f"WebCodecs packet forwarder configured for {url} → ws port {ws_port}")
        return NvdecPacketForwarder(url, self._webcodecs_server, decoder_opts=opts)

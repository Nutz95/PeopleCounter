"""FlaskStreamServer — SSE + MJPEG publisher backed by a Flask application.

Serves the browser UI and exposes:
  GET  /             → IHM HTML page
  GET  /health       → {"status": "ok"}
  GET  /api/config   → current mode + UI options
  POST /api/mode     → request inference mode change
  POST /api/sync_mode → request sync mode change
  GET  /api/last     → last published payload (debug)
  GET  /api/stream   → SSE stream of inference results
  GET  /api/video    → MJPEG stream of latest encoded frames
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from app_v2.core.result_publisher import ResultPublisher
from app_v2.infrastructure.flask_server.gpu_monitor import GpuMonitor
from app_v2.infrastructure.flask_server.mode_registry import _MODE_LABELS, _MODE_OVERLAYS
from app_v2.infrastructure.flask_server.runtime_state import RuntimeState
from app_v2.infrastructure.flask_server.streaming import MjpegFrameStore, SseHub
from app_v2.infrastructure.metadata_ws_server import MetadataWsServer
from app_v2.kernels.gpu_hotspot_renderer import GpuHotspotRenderer

try:
    from flask import Flask, Response, jsonify, render_template, stream_with_context
except Exception:  # pragma: no cover
    Flask = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    render_template = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    stream_with_context = None  # type: ignore[assignment]


class FlaskStreamServer(ResultPublisher):
    """Publishes fused payloads via SSE and MJPEG to connected browser clients."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        initial_config: dict[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.webcodecs_ws_port: int = 4999
        self.metadata_ws_port: int = MetadataWsServer.DEFAULT_PORT
        self._metadata_transport_mode: str = os.environ.get("METADATA_WS_MODE", "bbox").strip().lower()
        if self._metadata_transport_mode not in ("bbox", "centers"):
            self._metadata_transport_mode = "bbox"
        self._thread: threading.Thread | None = None

        # Snapshot of last inference payload (/api/last)
        self._last_lock = threading.Lock()
        self._last: dict[str, Any] = {"frame_id": None, "payload": None}

        # Overlay visibility flags synchronized with UI state
        # Updated by client via POST /api/overlay_options
        self._heatmap_requested: bool = True  # default: assume heatmap active
        self._mask_requested: bool = True
        self._server_side_heatmap_enabled: bool = True
        self._server_side_points_enabled: bool = True
        self._count_only_ui_enabled: bool = True

        video_stream_cfg = (initial_config or {}).get("video_stream", {}) if isinstance(initial_config, dict) else {}
        hotspot_render_cfg = video_stream_cfg.get("hotspot_render", {}) if isinstance(video_stream_cfg, dict) else {}
        radius_px = int(hotspot_render_cfg.get("radius_px", 3)) if isinstance(hotspot_render_cfg, dict) else 3
        color_rgb = hotspot_render_cfg.get("color_rgb", [220, 30, 30]) if isinstance(hotspot_render_cfg, dict) else [220, 30, 30]
        if not isinstance(color_rgb, (list, tuple)) or len(color_rgb) < 3:
            color_rgb = [220, 30, 30]
        hold_empty_frames = int(hotspot_render_cfg.get("hold_empty_frames", 2)) if isinstance(hotspot_render_cfg, dict) else 2
        max_render_points = int(hotspot_render_cfg.get("max_points", 0)) if isinstance(hotspot_render_cfg, dict) else 0
        self._server_side_points_for_yolo = bool(hotspot_render_cfg.get("enable_yolo_points", False)) if isinstance(hotspot_render_cfg, dict) else False
        self._server_side_points_yolo_min_detections = int(hotspot_render_cfg.get("yolo_min_detections", 800)) if isinstance(hotspot_render_cfg, dict) else 800
        self._server_side_max_render_points = max(0, max_render_points)
        self._hotspots_hold_empty_frames = max(0, hold_empty_frames)
        self._hotspots_empty_streak = 0

        # GPU hotspot rendering for dense scenes (P2PNet / Density)
        # Thread-safe cache of latest hotspots for NVJPEG annotation
        self._gpu_hotspot_renderer = GpuHotspotRenderer(
            circle_radius_px=max(1, radius_px),
            color_red=int(color_rgb[0]),
            color_green=int(color_rgb[1]),
            color_blue=int(color_rgb[2]),
        )
        self._hotspots_cache_lock = threading.Lock()
        self._hotspots_cache: list[tuple[float, float, float]] = []
        self._hotspots_cache_gpu_tensor: Any | None = None
        self._hotspots_frame_id = -1
        self._hotspots_history_limit = 8
        self._hotspots_history: dict[int, tuple[list[tuple[float, float, float]], Any | None]] = {}

        # Background thread for JSON encoding (off hot path)
        self._json_queue: Any = None  # Will be initialized in start()
        self._json_executor: Any = None  # Will be initialized in start()
        # Dedicated state/services (SOLID split)
        self._runtime = RuntimeState(initial_config)
        self._active_sync_mode = str(self._runtime.config_snapshot().get("sync_mode", "async"))
        self._sse = SseHub()
        self._mjpeg = MjpegFrameStore()
        self._gpu = GpuMonitor()
        self._metadata_ws = MetadataWsServer(port=self.metadata_ws_port)
        self._publish_metrics_prev: dict[str, float] = {
            "server_json_encode_ms": 0.0,
            "server_sse_publish_ms": 0.0,
            "server_publish_total_ms": 0.0,
        }

        assets_root = Path(__file__).resolve().parent
        self._template_dir = assets_root / "templates"
        self._static_dir = assets_root / "static"

        if Flask is None:
            self._app = None
            return

        self._app = Flask(
            __name__,
            template_folder=str(self._template_dir),
            static_folder=str(self._static_dir),
            static_url_path="/static",
        )
        self._app.config["TEMPLATES_AUTO_RELOAD"] = True
        self._register_routes()

    # ------------------------------------------------------------------
    # Flask routes
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        assert self._app is not None

        @self._app.get("/")
        def index() -> Any:
            return render_template(
                "index.html",
                port=self.port,
                ws_port=self.webcodecs_ws_port,
                meta_ws_port=self.metadata_ws_port,
            )

        @self._app.get("/health")
        def health() -> Any:
            return jsonify({"status": "ok"})

        @self._app.get("/api/ws_port")
        def api_ws_port() -> Any:
            return jsonify({"ws_port": self.webcodecs_ws_port})

        @self._app.get("/api/meta_ws_port")
        def api_meta_ws_port() -> Any:
            return jsonify({"ws_port": self.metadata_ws_port})

        @self._app.get("/api/config")
        def api_config() -> Any:
            snap = self._runtime.config_snapshot()
            snap["mode_labels"] = _MODE_LABELS
            snap["mode_overlays"] = _MODE_OVERLAYS
            return jsonify(snap)

        @self._app.post("/api/mode")
        def api_set_mode() -> Any:
            from flask import request

            body = request.get_json(silent=True) or {}
            _, payload, status = self._runtime.request_mode(str(body.get("mode", "")))
            return jsonify(payload), status

        @self._app.post("/api/sync_mode")
        def api_set_sync_mode() -> Any:
            from flask import request

            body = request.get_json(silent=True) or {}
            _, payload, status = self._runtime.request_sync_mode(str(body.get("mode", "")))
            return jsonify(payload), status

        @self._app.post("/api/video_backend")
        def api_set_video_backend() -> Any:
            from flask import request

            body = request.get_json(silent=True) or {}
            _, payload, status = self._runtime.request_video_backend(str(body.get("backend", "")))
            return jsonify(payload), status

        @self._app.post("/api/crowd/confidence")
        def api_set_crowd_confidence() -> Any:
            from flask import request

            body = request.get_json(silent=True) or {}
            try:
                value = float(body.get("confidence", 0.25))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "confidence must be a number"}), 400
            clamped = self._runtime.set_crowd_confidence(value)
            return jsonify({"ok": True, "confidence": clamped})

        @self._app.post("/api/density/threshold")
        def api_set_density_threshold() -> Any:
            from flask import request

            body = request.get_json(silent=True) or {}
            try:
                value = float(body.get("threshold", 0.05))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "threshold must be a number"}), 400
            clamped = self._runtime.set_density_threshold(value)
            return jsonify({"ok": True, "threshold": clamped})

        @self._app.get("/api/last")
        def api_last() -> Any:
            with self._last_lock:
                snapshot = dict(self._last)
            return jsonify(snapshot)

        @self._app.get("/api/stream")
        def api_stream() -> Any:
            """Server-Sent Events endpoint."""

            def generate() -> Any:
                q = self._sse.add_client()
                try:
                    while True:
                        try:
                            data = q.get(timeout=25.0)
                            yield f"data: {data}\n\n"
                        except Exception:
                            yield ": heartbeat\n\n"
                finally:
                    self._sse.remove_client(q)

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @self._app.get("/api/video")
        def api_video() -> Any:
            return Response(
                self._mjpeg.stream_iter(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache"},
            )

        @self._app.get("/api/gpu/stats")
        def api_gpu_stats() -> Any:
            return jsonify(self._gpu.snapshot())

        @self._app.post("/api/overlay_options")
        def api_overlay_options() -> Any:
            from flask import request

            body = request.get_json(silent=True) or {}
            heatmap_req = body.get("heatmap_enabled")
            if isinstance(heatmap_req, bool):
                self._heatmap_requested = heatmap_req
            mask_req = body.get("mask_enabled")
            if isinstance(mask_req, bool):
                self._mask_requested = mask_req
            server_side_req = body.get("server_side_heatmap_enabled")
            if isinstance(server_side_req, bool):
                self._server_side_heatmap_enabled = server_side_req
            server_side_points_req = body.get("server_side_points_enabled")
            if isinstance(server_side_points_req, bool):
                self._server_side_points_enabled = server_side_points_req
            count_only_req = body.get("count_only_ui_enabled")
            if isinstance(count_only_req, bool):
                self._count_only_ui_enabled = count_only_req
            return jsonify({
                "ok": True,
                "heatmap_enabled": self._heatmap_requested,
                "mask_enabled": self._mask_requested,
                "server_side_heatmap_enabled": self._server_side_heatmap_enabled,
                "server_side_points_enabled": self._server_side_points_enabled,
                "count_only_ui_enabled": self._count_only_ui_enabled,
            })

    # ------------------------------------------------------------------
    # Public API (ResultPublisher + control hooks)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start Flask web server in a background thread."""
        if self._app is None:
            raise RuntimeError("Flask is not available in this environment")
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._gpu.start()
        self._metadata_ws.start()
        self.metadata_ws_port = self._metadata_ws.port

    def _run(self) -> None:
        assert self._app is not None
        self._app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

    def publish(self, frame_id: int, payload: Sequence[dict[str, object]]) -> None:
        publish_start_ns = time.perf_counter_ns()
        payload_list = list(payload)
        async_passthrough_mode = self._active_sync_mode != "sync"

        # Push compact binary detections to metadata WS when clients are connected.
        # In server-side heatmap mode we disable metadata transport entirely to
        # avoid sending dense geometry to the browser.
        server_side_heatmap_active = (not async_passthrough_mode) and self._server_side_heatmap_enabled and self._heatmap_requested

        # Only enable server-side points for YOLO when explicitly enabled in config
        # and when detection count is high enough to justify replacing client overlay.
        yolo_detection_count = 0
        has_hotspot_model_payload = False
        for entry in payload_list:
            if not isinstance(entry, dict):
                continue
            model_name = str(entry.get("model", "")).lower()
            detections = entry.get("detections")
            if isinstance(detections, list):
                yolo_detection_count += len(detections)
            det_count = entry.get("detection_count")
            if isinstance(det_count, (int, float)):
                yolo_detection_count += int(det_count)
            if model_name in ("density", "p2pnet"):
                has_hotspot_model_payload = True

        server_side_points_active = False
        if (not async_passthrough_mode) and self._server_side_points_enabled:
            # Hotspot-native models (density/p2pnet): in sync mode, server-side
            # points must remain active independently from the bbox/mask toggle.
            # This avoids accidental on/off flicker when UI overlay controls
            # manipulate `mask_enabled` for non-bbox modes.
            if has_hotspot_model_payload:
                server_side_points_active = True
            elif self._mask_requested and yolo_detection_count > 0:
                # Sync mode policy: YOLO/CROWD points should render even at low
                # counts by default. The legacy config gate remains optional.
                if self._server_side_points_for_yolo:
                    server_side_points_active = yolo_detection_count >= self._server_side_points_yolo_min_detections
                else:
                    server_side_points_active = True
        server_side_overlay_active = server_side_heatmap_active or server_side_points_active
        has_metadata_clients = self._metadata_ws.has_clients()
        metadata_transport_active = (not async_passthrough_mode) and has_metadata_clients and not server_side_overlay_active
        meta_push_start_ns = time.perf_counter_ns()
        if metadata_transport_active:
            packed_rows, row_width, flags = self._pack_detection_rows(payload_list)
            self._metadata_ws.push_rows(frame_id, packed_rows, row_width=row_width, flags=flags)
        meta_push_done_ns = time.perf_counter_ns()

        # When metadata WS is active, strip heavy geometry arrays from SSE payload
        # (detections/hotspots) and keep only counters to minimize JSON overhead.
        # Additionally, gate hotspots from SSE if heatmap overlay is disabled by client.
        sse_payload = self._compact_payload_for_sse(
            payload_list,
            drop_detections=async_passthrough_mode or metadata_transport_active or self._count_only_ui_enabled,
            drop_hotspots=async_passthrough_mode or metadata_transport_active or (not self._heatmap_requested) or server_side_heatmap_active,
        )

        # Cache hotspots for GPU rendering in video encoder
        # Extract from payload before compaction so we always have the full data
        if server_side_overlay_active:
            self._update_hotspots_cache(payload_list, frame_id)
        else:
            self._clear_hotspots_cache(frame_id)
        compact_done_ns = time.perf_counter_ns()

        telemetry_dict: dict[str, object] | None = None
        for item in sse_payload:
            if isinstance(item, dict) and isinstance(item.get("telemetry"), dict):
                telemetry_dict = item.get("telemetry")  # type: ignore[assignment]
                break
        telemetry_update_start_ns = time.perf_counter_ns()
        if telemetry_dict is not None:
            telemetry_dict["server_meta_ws_push_ms"] = (meta_push_done_ns - meta_push_start_ns) / 1_000_000.0
            telemetry_dict["server_compact_payload_ms"] = (compact_done_ns - meta_push_done_ns) / 1_000_000.0
            telemetry_dict["server_meta_ws_clients"] = 1.0 if metadata_transport_active else 0.0
            telemetry_dict["server_side_heatmap_active"] = 1.0 if server_side_heatmap_active else 0.0
            telemetry_dict["server_side_points_active"] = 1.0 if server_side_points_active else 0.0
            telemetry_dict["server_side_overlay_active"] = 1.0 if server_side_overlay_active else 0.0
            telemetry_dict["server_side_points_yolo_count"] = float(yolo_detection_count)
            telemetry_dict["server_count_only_ui_active"] = 1.0 if self._count_only_ui_enabled else 0.0
            telemetry_dict["server_async_passthrough_mode"] = 1.0 if async_passthrough_mode else 0.0
            telemetry_dict.update(self._publish_metrics_prev)
        telemetry_update_done_ns = time.perf_counter_ns()

        data = {"frame_id": frame_id, "payload": sse_payload}
        lock_start_ns = time.perf_counter_ns()
        # Defer _last update to background (avoid blocking hot path on lock contention).
        # Use non-blocking check to see if we should even bother.
        lock_acquired = self._last_lock.acquire(blocking=False)
        if lock_acquired:
            try:
                self._last = data
            finally:
                self._last_lock.release()
        lock_done_ns = time.perf_counter_ns()
        json_start_ns = time.perf_counter_ns()
        encoded = json.dumps(data, default=str)
        json_done_ns = time.perf_counter_ns()
        self._sse.publish(encoded)
        publish_done_ns = time.perf_counter_ns()
        self._publish_metrics_prev = {
            "server_json_encode_ms": (json_done_ns - json_start_ns) / 1_000_000.0,
            "server_sse_publish_ms": (publish_done_ns - json_done_ns) / 1_000_000.0,
            "server_publish_total_ms": (publish_done_ns - publish_start_ns) / 1_000_000.0,
            "server_telemetry_update_ms": (telemetry_update_done_ns - telemetry_update_start_ns) / 1_000_000.0,
            "server_lock_acquired": 1.0 if lock_acquired else 0.0,
            "server_lock_hold_ms": (lock_done_ns - lock_start_ns) / 1_000_000.0,
        }

    def publish_passthrough_frame(self, frame_id: int) -> None:
        data = {"frame_id": frame_id, "payload": [], "passthrough": True}
        self._sse.publish(json.dumps(data))

    def push_frame(self, jpeg_bytes: bytes) -> None:
        self._mjpeg.push(jpeg_bytes)

    def _update_hotspots_cache(self, payload: Sequence[dict[str, object]], frame_id: int) -> None:
        """Extract hotspots from payload and cache for GPU rendering in video encoder."""
        hotspots: list[tuple[float, float, float]] = []
        hotspots_gpu_tensor: Any | None = None
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            candidate_gpu = entry.get("_hotspots_gpu_tensor")
            if hotspots_gpu_tensor is None and candidate_gpu is not None:
                try:
                    import torch  # local optional import

                    if isinstance(candidate_gpu, torch.Tensor):
                        safe_gpu = candidate_gpu.detach()
                        if (
                            self._server_side_max_render_points > 0
                            and safe_gpu.dim() == 2
                            and int(safe_gpu.shape[0]) > self._server_side_max_render_points
                        ):
                            total = int(safe_gpu.shape[0])
                            target = int(self._server_side_max_render_points)
                            sample_pos = torch.linspace(0, total - 1, steps=target, device=safe_gpu.device)
                            sample_idx = sample_pos.round().long().clamp(0, total - 1)
                            safe_gpu = safe_gpu[sample_idx]
                        if safe_gpu.numel() > 0:
                            # Own a stable tensor independent from decoder buffers.
                            hotspots_gpu_tensor = safe_gpu.clone()
                except Exception:
                    hotspots_gpu_tensor = None

            detections = entry.get("detections")
            if isinstance(detections, list):
                for det in detections:
                    if not isinstance(det, dict):
                        continue
                    bbox = det.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                        continue
                    try:
                        x1 = float(bbox[0]); y1 = float(bbox[1]); x2 = float(bbox[2]); y2 = float(bbox[3])
                        conf = float(det.get("conf", 1.0))
                    except (TypeError, ValueError):
                        continue
                    hotspots.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5, max(0.0, min(1.0, conf))))

            hs_list = entry.get("hotspots")
            if isinstance(hs_list, list):
                for hs in hs_list:
                    if isinstance(hs, dict):
                        try:
                            x = float(hs.get("x", 0.0))
                            y = float(hs.get("y", 0.0))
                            w = float(hs.get("w", 0.0))
                            hotspots.append((x, y, w))
                        except (TypeError, ValueError):
                            pass

        has_hotspots = len(hotspots) > 0
        if self._server_side_max_render_points > 0 and len(hotspots) > self._server_side_max_render_points:
            total = len(hotspots)
            target = self._server_side_max_render_points
            if target > 0:
                step = max(1, total // target)
                hotspots = hotspots[::step][:target]
                has_hotspots = len(hotspots) > 0
        if not has_hotspots and hotspots_gpu_tensor is not None:
            try:
                has_hotspots = int(hotspots_gpu_tensor.shape[0]) > 0
            except Exception:
                has_hotspots = False

        with self._hotspots_cache_lock:
            if has_hotspots:
                self._hotspots_cache = hotspots
                self._hotspots_cache_gpu_tensor = hotspots_gpu_tensor
                self._hotspots_frame_id = frame_id
                self._hotspots_history[int(frame_id)] = (list(hotspots), hotspots_gpu_tensor)
                if len(self._hotspots_history) > self._hotspots_history_limit:
                    keep_keys = sorted(self._hotspots_history.keys())[-self._hotspots_history_limit:]
                    keep_set = set(keep_keys)
                    self._hotspots_history = {
                        key: value for key, value in self._hotspots_history.items() if key in keep_set
                    }
                self._hotspots_empty_streak = 0
                return

            self._hotspots_empty_streak += 1
            if self._hotspots_empty_streak > self._hotspots_hold_empty_frames:
                self._hotspots_cache = []
                self._hotspots_cache_gpu_tensor = None
                self._hotspots_frame_id = frame_id
                self._hotspots_empty_streak = 0

    def _clear_hotspots_cache(self, frame_id: int) -> None:
        with self._hotspots_cache_lock:
            self._hotspots_cache = []
            self._hotspots_cache_gpu_tensor = None
            self._hotspots_frame_id = frame_id
            self._hotspots_history = {}
            self._hotspots_empty_streak = 0

    def get_hotspots_for_frame(self, frame_id: int) -> Any:
        """Retrieve cached hotspots for video encoder (safe for any thread)."""
        with self._hotspots_cache_lock:
            if frame_id < 0:
                if self._hotspots_cache_gpu_tensor is not None:
                    return self._hotspots_cache_gpu_tensor
                return list(self._hotspots_cache)
            if self._hotspots_frame_id == frame_id:
                if self._hotspots_cache_gpu_tensor is not None:
                    return self._hotspots_cache_gpu_tensor
                return list(self._hotspots_cache)
            history_entry = self._hotspots_history.get(int(frame_id))
            if history_entry is not None:
                history_hotspots, history_gpu_tensor = history_entry
                if history_gpu_tensor is not None:
                    return history_gpu_tensor
                return list(history_hotspots)
        return []

    def get_gpu_hotspot_renderer(self) -> GpuHotspotRenderer:
        """Return GPU hotspot renderer instance (for video encoder integration)."""
        return self._gpu_hotspot_renderer

    def _pack_detection_rows(self, payload: Sequence[dict[str, object]]) -> tuple[list[float], int, int]:
        """Flatten detections for binary transport.

        Returns (rows, row_width, flags).
          - bbox mode   -> row_width=5, flags bit0=0, row=[x1,y1,x2,y2,conf]
          - centers mode-> row_width=3, flags bit0=1, row=[cx,cy,conf]
        """
        preferred_centers_mode = self._metadata_transport_mode == "centers"
        row_width = 3 if preferred_centers_mode else 5
        flags = 0x0001 if preferred_centers_mode else 0x0000
        packed: list[float] = []
        for entry in payload:
            detections = entry.get("detections") if isinstance(entry, dict) else None
            if not isinstance(detections, list):
                continue
            for det in detections:
                if not isinstance(det, dict):
                    continue
                bbox = det.get("bbox")
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                try:
                    x1 = float(bbox[0]); y1 = float(bbox[1]); x2 = float(bbox[2]); y2 = float(bbox[3])
                    conf = float(det.get("conf", 0.0))
                except (TypeError, ValueError):
                    continue
                if preferred_centers_mode:
                    packed.extend(((x1 + x2) * 0.5, (y1 + y2) * 0.5, conf))
                else:
                    packed.extend((x1, y1, x2, y2, conf))

        # Fallback path for point-only models (e.g. P2PNet) where payload exposes
        # `hotspots` instead of `detections`.
        if packed:
            return packed, row_width, flags

        hotspot_rows: list[float] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            hotspots = entry.get("hotspots")
            if not isinstance(hotspots, list):
                continue
            for hs in hotspots:
                if not isinstance(hs, dict):
                    continue
                try:
                    x = float(hs.get("x", 0.0))
                    y = float(hs.get("y", 0.0))
                    w = float(hs.get("w", 1.0))
                except (TypeError, ValueError):
                    continue
                hotspot_rows.extend((x, y, w))

        # Hotspots are always center points.
        return hotspot_rows, 3, 0x0001

    @staticmethod
    def _compact_payload_for_sse(
        payload: Sequence[dict[str, object]],
        *,
        drop_detections: bool,
        drop_hotspots: bool,
    ) -> list[dict[str, object]]:
        if not drop_detections and not drop_hotspots:
            return list(payload)
        compacted: list[dict[str, object]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                compacted.append(entry)  # type: ignore[arg-type]
                continue
            item = dict(entry)
            for key in list(item.keys()):
                if isinstance(key, str) and key.startswith("_"):
                    item.pop(key, None)
            detections = item.get("detections")
            if isinstance(detections, list):
                existing_detection_count = item.get("detection_count")
                if isinstance(existing_detection_count, (int, float)):
                    item["detection_count"] = max(int(existing_detection_count), len(detections))
                else:
                    item["detection_count"] = len(detections)
                if drop_detections:
                    item.pop("detections", None)

            hotspots = item.get("hotspots")
            if isinstance(hotspots, list):
                existing_hotspot_count = item.get("hotspot_count")
                if isinstance(existing_hotspot_count, (int, float)):
                    item["hotspot_count"] = max(int(existing_hotspot_count), len(hotspots))
                else:
                    item["hotspot_count"] = len(hotspots)
                if drop_hotspots:
                    item.pop("hotspots", None)
            compacted.append(item)
        return compacted

    # ------------------------------------------------------------------
    # Runtime mode/threshold management (called from pipeline loop)
    # ------------------------------------------------------------------

    def get_and_clear_pending_mode(self) -> str | None:
        return self._runtime.get_and_clear_pending_mode()

    def get_and_clear_pending_sync_mode(self) -> str | None:
        return self._runtime.get_and_clear_pending_sync_mode()

    def get_and_clear_pending_density_threshold(self) -> float | None:
        return self._runtime.get_and_clear_pending_density_threshold()

    def get_and_clear_pending_crowd_confidence(self) -> float | None:
        return self._runtime.get_and_clear_pending_crowd_confidence()

    def get_and_clear_pending_video_backend(self) -> str | None:
        return self._runtime.get_and_clear_pending_video_backend()

    def set_active_video_backend(self, backend: str) -> None:
        self._runtime.set_active_video_backend(backend)

    def set_active_mode(self, mode: str) -> None:
        self._runtime.set_active_mode(mode)

    def set_active_sync_mode(self, mode: str) -> None:
        self._runtime.set_active_sync_mode(mode)
        self._active_sync_mode = mode

    def update_available_modes(self, config: dict[str, Any]) -> None:
        self._runtime.update_available_modes(config)

    @staticmethod
    def _compute_available_modes(config: dict[str, Any]) -> list[str]:
        # Backward-compatible helper used by older callers/tests.
        return RuntimeState.compute_available_modes(config)

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

        # Background thread for JSON encoding (off hot path)
        self._json_queue: Any = None  # Will be initialized in start()
        self._json_executor: Any = None  # Will be initialized in start()
        # Dedicated state/services (SOLID split)
        self._runtime = RuntimeState(initial_config)
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

        # Push compact binary detections to metadata WS when clients are connected.
        has_metadata_clients = self._metadata_ws.has_clients()
        meta_push_start_ns = time.perf_counter_ns()
        if has_metadata_clients:
            packed_rows, row_width, flags = self._pack_detection_rows(payload_list)
            self._metadata_ws.push_rows(frame_id, packed_rows, row_width=row_width, flags=flags)
        meta_push_done_ns = time.perf_counter_ns()

        # When metadata WS is active, strip heavy detections from SSE payload and
        # keep only detection_count to minimize JSON serialization cost.
        sse_payload = self._compact_payload_for_sse(payload_list, drop_detections=has_metadata_clients)
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
            telemetry_dict["server_meta_ws_clients"] = 1.0 if has_metadata_clients else 0.0
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

    def _pack_detection_rows(self, payload: Sequence[dict[str, object]]) -> tuple[list[float], int, int]:
        """Flatten detections for binary transport.

        Returns (rows, row_width, flags).
          - bbox mode   -> row_width=5, flags bit0=0, row=[x1,y1,x2,y2,conf]
          - centers mode-> row_width=3, flags bit0=1, row=[cx,cy,conf]
        """
        centers_mode = self._metadata_transport_mode == "centers"
        row_width = 3 if centers_mode else 5
        flags = 0x0001 if centers_mode else 0x0000
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
                if centers_mode:
                    packed.extend(((x1 + x2) * 0.5, (y1 + y2) * 0.5, conf))
                else:
                    packed.extend((x1, y1, x2, y2, conf))
        return packed, row_width, flags

    @staticmethod
    def _compact_payload_for_sse(
        payload: Sequence[dict[str, object]],
        *,
        drop_detections: bool,
    ) -> list[dict[str, object]]:
        if not drop_detections:
            return list(payload)
        compacted: list[dict[str, object]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                compacted.append(entry)  # type: ignore[arg-type]
                continue
            item = dict(entry)
            detections = item.get("detections")
            if isinstance(detections, list):
                item["detection_count"] = len(detections)
                item.pop("detections", None)
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

    def update_available_modes(self, config: dict[str, Any]) -> None:
        self._runtime.update_available_modes(config)

    @staticmethod
    def _compute_available_modes(config: dict[str, Any]) -> list[str]:
        # Backward-compatible helper used by older callers/tests.
        return RuntimeState.compute_available_modes(config)

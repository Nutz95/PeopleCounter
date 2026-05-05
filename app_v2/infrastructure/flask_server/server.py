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
import threading
from pathlib import Path
from typing import Any, Sequence

from app_v2.core.result_publisher import ResultPublisher
from app_v2.infrastructure.flask_server.gpu_monitor import GpuMonitor
from app_v2.infrastructure.flask_server.mode_registry import _MODE_LABELS, _MODE_OVERLAYS
from app_v2.infrastructure.flask_server.runtime_state import RuntimeState
from app_v2.infrastructure.flask_server.streaming import MjpegFrameStore, SseHub

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
        self._thread: threading.Thread | None = None

        # Snapshot of last inference payload (/api/last)
        self._last_lock = threading.Lock()
        self._last: dict[str, Any] = {"frame_id": None, "payload": None}

        # Dedicated state/services (SOLID split)
        self._runtime = RuntimeState(initial_config)
        self._sse = SseHub()
        self._mjpeg = MjpegFrameStore()
        self._gpu = GpuMonitor()

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
            return render_template("index.html", port=self.port, ws_port=self.webcodecs_ws_port)

        @self._app.get("/health")
        def health() -> Any:
            return jsonify({"status": "ok"})

        @self._app.get("/api/ws_port")
        def api_ws_port() -> Any:
            return jsonify({"ws_port": self.webcodecs_ws_port})

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

    def _run(self) -> None:
        assert self._app is not None
        self._app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

    def publish(self, frame_id: int, payload: Sequence[dict[str, object]]) -> None:
        data = {"frame_id": frame_id, "payload": list(payload)}
        with self._last_lock:
            self._last = data
        self._sse.publish(json.dumps(data, default=str))

    def publish_passthrough_frame(self, frame_id: int) -> None:
        data = {"frame_id": frame_id, "payload": [], "passthrough": True}
        self._sse.publish(json.dumps(data))

    def push_frame(self, jpeg_bytes: bytes) -> None:
        self._mjpeg.push(jpeg_bytes)

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

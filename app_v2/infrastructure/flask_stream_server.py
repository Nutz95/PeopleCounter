from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from app_v2.core.result_publisher import ResultPublisher


try:
    from flask import Flask, Response, jsonify, render_template, stream_with_context
except Exception:  # pragma: no cover
    Flask = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    render_template = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]
    stream_with_context = None  # type: ignore[assignment]

# Minimal 1x1 black JPEG served when no frame has been pushed yet.
_PLACEHOLDER_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"C  C\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
    b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4"
    b"\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00"
    b"\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142"
    b"\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18"
    b"\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85"
    b"\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3"
    b"\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba"
    b"\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8"
    b"\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4"
    b"\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb"
    b"\xd4P\x00\x00\x00\x1f\xff\xd9"
)


class FlaskStreamServer(ResultPublisher):
    """Publishes fused payloads via SSE and MJPEG to connected browser clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port
        # Set by PipelineOrchestrator when WebCodecsServer is started.
        # Passed to index.html so JS can connect to the correct WS port.
        self.webcodecs_ws_port: int = 5001
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last: dict[str, Any] = {"frame_id": None, "payload": None}
        # SSE: one SimpleQueue per connected browser tab
        self._sse_clients: list[queue.SimpleQueue[str]] = []
        self._sse_lock = threading.Lock()
        # MJPEG: latest JPEG frame bytes
        self._last_jpeg: bytes = _PLACEHOLDER_JPEG
        self._video_lock = threading.Lock()

        assets_root = Path(__file__).resolve().parent / "flask_server"
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

        @self._app.get("/")
        def index() -> Any:
            return render_template("index.html", port=self.port, ws_port=self.webcodecs_ws_port)

        @self._app.get("/health")
        def health() -> Any:
            return jsonify({"status": "ok"})

        @self._app.get("/api/last")
        def api_last() -> Any:
            with self._lock:
                snapshot = dict(self._last)
            return jsonify(snapshot)

        @self._app.get("/api/stream")
        def api_stream() -> Any:
            """Server-Sent Events endpoint — pushes each published frame to the browser."""

            def generate() -> Any:
                q: queue.SimpleQueue[str] = queue.SimpleQueue()
                with self._sse_lock:
                    self._sse_clients.append(q)
                try:
                    while True:
                        try:
                            data = q.get(timeout=25.0)
                            yield f"data: {data}\n\n"
                        except queue.Empty:
                            # heartbeat to keep the connection alive
                            yield ": heartbeat\n\n"
                finally:
                    with self._sse_lock:
                        try:
                            self._sse_clients.remove(q)
                        except ValueError:
                            pass

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @self._app.get("/api/video")
        def api_video() -> Any:
            """MJPEG stream — each published frame is served as a multipart boundary."""

            def generate() -> Any:
                while True:
                    with self._video_lock:
                        frame = self._last_jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    time.sleep(1 / 25)

            return Response(
                generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache"},
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Flask web server in a background thread."""
        if self._app is None:
            raise RuntimeError("Flask is not available in this environment")
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._app is not None
        self._app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)

    def publish(self, frame_id: int, payload: Sequence[dict[str, object]]) -> None:
        """Store last payload and push to all SSE subscribers."""
        data = {"frame_id": frame_id, "payload": list(payload)}
        with self._lock:
            self._last = data
        # Serialise once; send to all connected SSE clients
        serialised = json.dumps(data, default=str)
        with self._sse_lock:
            clients = list(self._sse_clients)
        for q in clients:
            try:
                q.put_nowait(serialised)
            except Exception:
                pass

    def push_frame(self, jpeg_bytes: bytes) -> None:
        """Update the MJPEG frame buffer with a freshly encoded JPEG."""
        with self._video_lock:
            self._last_jpeg = jpeg_bytes

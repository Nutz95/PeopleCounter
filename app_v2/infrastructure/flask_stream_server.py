from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence

from app_v2.core.result_publisher import ResultPublisher


try:
    from flask import Flask, jsonify, render_template
except Exception:  # pragma: no cover
    Flask = None  # type: ignore[assignment]
    jsonify = None  # type: ignore[assignment]
    render_template = None  # type: ignore[assignment]


class FlaskStreamServer(ResultPublisher):
    """Publishes fused payloads to the Flask backend without inline inference."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last: dict[str, Any] = {"frame_id": None, "payload": None}

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
            return render_template("index.html", port=self.port)

        @self._app.get("/health")
        def health() -> Any:
            return jsonify({"status": "ok"})

        @self._app.get("/api/last")
        def api_last() -> Any:
            with self._lock:
                snapshot = dict(self._last)
            return jsonify(snapshot)

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
        """Push metadata downstream (mask timing, detection counts, etc.)."""
        with self._lock:
            self._last = {"frame_id": frame_id, "payload": list(payload)}

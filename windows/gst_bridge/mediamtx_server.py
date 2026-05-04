from __future__ import annotations

import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path


class MediaMtxServer:
    def __init__(self, executable: Path, rtsp_port: int, rtmp_port: int, on_log: Callable[[str], None] | None = None) -> None:
        self._executable = executable
        self._rtsp_port = rtsp_port
        self._rtmp_port = rtmp_port
        self._on_log = on_log
        self._process: subprocess.Popen[str] | None = None
        self._reader_threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._is_port_open("127.0.0.1", self._rtsp_port):
            raise RuntimeError(
                f"RTSP port {self._rtsp_port} is already in use before MediaMTX startup. "
                "Another MediaMTX instance or RTSP server is probably still running."
            )
        if self._is_port_open("127.0.0.1", self._rtmp_port):
            raise RuntimeError(
                f"RTMP port {self._rtmp_port} is already in use before MediaMTX startup. "
                "Another MediaMTX instance or RTMP server is probably still running."
            )
        config_path = self._write_config(self._executable.parent)
        self._process = subprocess.Popen(
            [str(self._executable), str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(self._executable.parent),
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._reader_threads = [
            threading.Thread(target=self._reader_loop, args=(self._process.stdout,), daemon=True),
            threading.Thread(target=self._reader_loop, args=(self._process.stderr,), daemon=True),
        ]
        for thread in self._reader_threads:
            thread.start()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("MediaMTX exited during startup")
            if self._is_port_open("127.0.0.1", self._rtsp_port) and self._is_port_open("127.0.0.1", self._rtmp_port):
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"MediaMTX did not open RTSP port {self._rtsp_port} and RTMP port {self._rtmp_port} in time"
        )

    def stop(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()
        self._process = None
        self._reader_threads = []

    def _write_config(self, working_dir: Path) -> Path:
        config_path = working_dir / "mediamtx-generated.yml"
        config_path.write_text(
            "\n".join(
                [
                    f"rtspAddress: :{self._rtsp_port}",
                    "rtspTransports: [tcp]",
                    f"rtmpAddress: :{self._rtmp_port}",
                    "hls: no",
                    "webrtc: no",
                    "srt: no",
                    "paths:",
                    "  live:",
                    "    source: publisher",
                    "  all_others:",
                    "    source: publisher",
                ]
            ),
            encoding="utf-8",
        )
        return config_path

    def _reader_loop(self, stream) -> None:
        if stream is None:
            return
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            self._emit_log(line)

    def _emit_log(self, line: str) -> None:
        if self._on_log is not None:
            self._on_log(f"[mediamtx] {line}")

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

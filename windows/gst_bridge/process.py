from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

FPS_LINE = re.compile(r"current:\s*(?P<current>[0-9.]+).*average:\s*(?P<average>[0-9.]+).*dropped:\s*(?P<dropped>[0-9.]+)", re.IGNORECASE)
PIPELINE_STATE_LINE = re.compile(
    r'element\s+"pipeline0".*state-changed.*new-state=\(GstState\)(?P<state>\w+)',
    re.IGNORECASE,
)
WARNING_LINE = re.compile(r"warning", re.IGNORECASE)
ERROR_LINE = re.compile(r"error", re.IGNORECASE)


class GStreamerProcess:
    def __init__(
        self,
        gst_launch_path: Path,
        env: dict[str, str],
        on_log: Callable[[str], None],
        on_metrics: Callable[[float, float, float], None],
        on_state: Callable[[str], None],
        on_warning: Callable[[], None],
        on_error: Callable[[], None],
        on_terminated: Callable[[int], None],
    ) -> None:
        self._gst_launch_path = gst_launch_path
        self._env = env
        self._on_log = on_log
        self._on_metrics = on_metrics
        self._on_state = on_state
        self._on_warning = on_warning
        self._on_error = on_error
        self._on_terminated = on_terminated
        self._process: subprocess.Popen[str] | None = None
        self._reader_threads: list[threading.Thread] = []
        self._process_lock = threading.Lock()
        self._process_token = 0
        self._handled_tokens: set[int] = set()

    def start(self, args: list[str]) -> None:
        self.stop()
        process = subprocess.Popen(
            [str(self._gst_launch_path), *args],
            env=self._env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with self._process_lock:
            self._process = process
            self._process_token += 1
            token = self._process_token
        self._reader_threads = [
            threading.Thread(target=self._reader_loop, args=(process.stdout,), daemon=True),
            threading.Thread(target=self._reader_loop, args=(process.stderr,), daemon=True),
            threading.Thread(target=self._wait_loop, args=(process, token), daemon=True),
        ]
        for thread in self._reader_threads:
            thread.start()
        self._on_state("starting")

    def stop(self) -> None:
        with self._process_lock:
            process = self._process
            token = self._process_token
        if process is None:
            return
        try:
            process.terminate()
            process.wait(timeout=1)
        except Exception:
            process.kill()
            process.wait(timeout=2)
        self._on_state("stopped")
        self._finalize_exit(process, token, process.returncode or 0)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _reader_loop(self, stream) -> None:
        if stream is None:
            return
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            self._on_log(line)
            self._parse_line(line)

    def _wait_loop(self, process: subprocess.Popen[str], token: int) -> None:
        exit_code = process.wait()
        self._finalize_exit(process, token, exit_code)

    def _finalize_exit(self, process: subprocess.Popen[str], token: int, exit_code: int) -> None:
        with self._process_lock:
            if token in self._handled_tokens:
                return
            self._handled_tokens.add(token)
            if self._process is process:
                self._process = None
        self._on_terminated(exit_code)

    def _parse_line(self, line: str) -> None:
        match = FPS_LINE.search(line)
        if match:
            self._on_metrics(
                float(match.group("current")),
                float(match.group("average")),
                float(match.group("dropped")),
            )
        state_match = PIPELINE_STATE_LINE.search(line)
        if state_match:
            self._on_state(state_match.group("state").lower())
        if WARNING_LINE.search(line):
            self._on_warning()
        if ERROR_LINE.search(line):
            self._on_error()

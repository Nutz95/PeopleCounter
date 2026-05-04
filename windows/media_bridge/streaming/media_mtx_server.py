from __future__ import annotations

import subprocess
import socket
import time
import urllib.request
import zipfile
from pathlib import Path


MEDIAMTX_DOWNLOAD_URL = "https://github.com/bluenviron/mediamtx/releases/download/v1.11.3/mediamtx_v1.11.3_windows_amd64.zip"


class MediaMtxServer:
    def __init__(self, bin_dir: Path, port: int, always_available_file: Path | None = None) -> None:
        self.bin_dir = bin_dir
        self.port = port
        self.always_available_file = always_available_file
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        executable = self._ensure_binary()
        config_path = self._write_config(executable.parent)
        self.process = subprocess.Popen(
            [str(executable), str(config_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=str(executable.parent),
        )
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("MediaMTX exited during startup")
            if self._is_port_open("127.0.0.1", self.port):
                return
            time.sleep(0.1)
        raise RuntimeError(f"MediaMTX did not open RTSP port {self.port} in time")

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        self.process = None

    def _ensure_binary(self) -> Path:
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        executable = next(self.bin_dir.rglob("mediamtx.exe"), None)
        if executable is not None and executable.exists():
            return executable

        archive_path = self.bin_dir / "mediamtx.zip"
        urllib.request.urlretrieve(MEDIAMTX_DOWNLOAD_URL, archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(self.bin_dir)
        archive_path.unlink(missing_ok=True)

        executable = next(self.bin_dir.rglob("mediamtx.exe"), None)
        if executable is None:
            raise RuntimeError("mediamtx.exe not found after extraction")
        return executable

    def _write_config(self, working_dir: Path) -> Path:
        config_path = working_dir / "mediamtx-generated.yml"
        lines = [
            f"rtspAddress: :{self.port}",
            "paths:",
            "  live:",
            "    source: publisher",
        ]
        lines.extend(
            [
                "  all_others:",
                "    source: publisher",
            ]
        )
        config_path.write_text("\n".join(lines), encoding="utf-8")
        return config_path

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False
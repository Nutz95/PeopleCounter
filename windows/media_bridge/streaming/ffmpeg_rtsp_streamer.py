from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import zmq

from ..models.media_item import MediaItem
from .zmq_command_client import ZmqCommandClient

logger = logging.getLogger("media_bridge.rtsp_streamer")


class FfmpegRtspStreamer:
    def __init__(
        self,
        ffmpeg_path: Path,
        items: list[MediaItem],
        width: int,
        height: int,
        fps: int,
        bitrate_kbps: int,
        encoder_name: str,
        rtsp_port: int,
        rtsp_path: str,
        zmq_port: int,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.items = items
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.encoder_name = encoder_name
        self.rtsp_port = rtsp_port
        self.rtsp_path = rtsp_path
        self.zmq_port = zmq_port
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._zmq_client: ZmqCommandClient | None = None
        self._recent_stderr: deque[str] = deque(maxlen=40)

    @property
    def stream_url(self) -> str:
        return f"rtsp://127.0.0.1:{self.rtsp_port}/{self.rtsp_path}"

    def start(self, initial_source_id: str) -> None:
        initial_index = self._find_item_index(initial_source_id)
        command = self._build_command(initial_index)
        logger.info("Starting FFmpeg RTSP streamer: %s", " ".join(command))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()
        if self._supports_live_switching():
            self._zmq_client = ZmqCommandClient(f"tcp://127.0.0.1:{self.zmq_port}")

    def switch_to(self, source_id: str, timeout_seconds: float = 30.0) -> None:
        if not self._supports_live_switching():
            raise RuntimeError("Live switching is not available when the streamer runs with a single active source")
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("FFmpeg RTSP streamer is not running")
        if self._zmq_client is None:
            raise RuntimeError("ZMQ control channel is not ready")
        target_index = self._find_item_index(source_id)
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("FFmpeg RTSP streamer exited during switch")
            try:
                reply = self._zmq_client.send(
                    "streamselect@video_selector",
                    "map",
                    str(target_index),
                    timeout_ms=2000,
                )
                logger.info("Switched stream to index %s (%s): %s", target_index, source_id, reply.strip())
                return
            except zmq.Again as exc:
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError(f"Timed out while switching FFmpeg stream to {source_id}: {last_error}")

    def wait_until_ready(self, source_id: str, timeout_seconds: float = 45.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        ready_since: float | None = None
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(self._build_startup_error("FFmpeg RTSP streamer exited during startup"))
            control_ready = self._is_control_port_open()
            rtsp_ready = self._is_rtsp_endpoint_responding()
            if control_ready and rtsp_ready:
                if ready_since is None:
                    ready_since = time.monotonic()
                # Ensure the process remains alive a bit after becoming ready.
                if time.monotonic() - ready_since >= 1.5:
                    return
            else:
                ready_since = None
            time.sleep(0.25)
        raise RuntimeError(
            self._build_startup_error(
                f"FFmpeg did not become fully ready (ZMQ+RTSP) in time for {source_id}"
            )
        )

    def stop(self) -> None:
        if self._zmq_client is not None:
            self._zmq_client.close()
            self._zmq_client = None
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
            self._stderr_thread = None

    def _find_item_index(self, source_id: str) -> int:
        for index, item in enumerate(self.items):
            if item.source_id == source_id:
                return index
        raise KeyError(f"Unknown source id: {source_id}")

    def _pump_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            line = self.process.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            self._recent_stderr.append(text)
            logger.warning("ffmpeg: %s", text)

    def _is_control_port_open(self) -> bool:
        if not self._supports_live_switching():
            return True
        try:
            with socket.create_connection(("127.0.0.1", self.zmq_port), timeout=0.5):
                return True
        except OSError:
            return False

    def _is_rtsp_endpoint_responding(self) -> bool:
        request = (
            f"OPTIONS rtsp://127.0.0.1:{self.rtsp_port}/{self.rtsp_path} RTSP/1.0\r\n"
            "CSeq: 1\r\n"
            "User-Agent: peoplecounter-media-bridge\r\n"
            "\r\n"
        ).encode("ascii")
        try:
            with socket.create_connection(("127.0.0.1", self.rtsp_port), timeout=0.75) as sock:
                sock.settimeout(0.75)
                sock.sendall(request)
                response = sock.recv(256)
            return response.startswith(b"RTSP/1.0")
        except OSError:
            return False

    def _build_startup_error(self, base_message: str) -> str:
        if not self._recent_stderr:
            return base_message
        tail = "\n".join(self._recent_stderr)
        return f"{base_message}. Last FFmpeg lines:\n{tail}"

    def _build_command(self, initial_index: int) -> list[str]:
        command = [str(self.ffmpeg_path), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self._uses_intel_qsv():
            command.extend(["-init_hw_device", "qsv=hw"])
        for item in self.items:
            command.extend(item.input_args)

        if not self._supports_live_switching():
            return self._build_single_source_command(command, initial_index)

        target_pix_fmt = self._target_pixel_format()
        filter_parts: list[str] = []
        mapped_inputs: list[str] = []
        for index, _item in enumerate(self.items):
            label = f"v{index}"
            filter_parts.append(
                f"[{index}:v]scale={self.width}:{self.height}:force_original_aspect_ratio=decrease:out_range=tv,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"fps={self.fps},setsar=1,format={target_pix_fmt},setpts=N/({self.fps}*TB)[{label}]"
            )
            mapped_inputs.append(f"[{label}]")

        zmq_filter = f"zmq=bind_address='tcp\\://127.0.0.1\\:{self.zmq_port}'"
        selected_stream = (
            "".join(mapped_inputs)
            + f"streamselect@video_selector=inputs={len(self.items)}:map={initial_index},"
            + zmq_filter
        )
        selected_stream += "[outv]"
        filter_parts.append(selected_stream)

        command.extend(["-filter_complex", ";".join(filter_parts), "-map", "[outv]", "-an", "-c:v", self.encoder_name])
        command.extend(self._build_encoder_args())
        command.extend(
            [
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                f"rtsp://127.0.0.1:{self.rtsp_port}/{self.rtsp_path}",
            ]
        )
        return command

    def _build_single_source_command(self, command: list[str], initial_index: int) -> list[str]:
        target_pix_fmt = self._target_pixel_format()
        command.extend(
            [
                "-vf",
                ",".join(
                    [
                        f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease:out_range=tv",
                        f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black",
                        f"fps={self.fps}",
                        "setsar=1",
                        f"format={target_pix_fmt}",
                    ]
                ),
                "-map",
                f"{initial_index}:v:0",
                "-an",
                "-c:v",
                self.encoder_name,
            ]
        )
        command.extend(self._build_encoder_args())
        command.extend(
            [
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                f"rtsp://127.0.0.1:{self.rtsp_port}/{self.rtsp_path}",
            ]
        )
        return command

    def _target_pixel_format(self) -> str:
        if self.encoder_name in {"h264_qsv", "h264_nvenc"}:
            return "nv12"
        return "yuv420p"

    def _supports_live_switching(self) -> bool:
        return len(self.items) > 1

    def _uses_intel_qsv(self) -> bool:
        return self.encoder_name == "h264_qsv"

    def _build_encoder_args(self) -> list[str]:
        maxrate = int(self.bitrate_kbps * 1.5)
        bufsize = int(self.bitrate_kbps * 2)
        args = []
        if self.encoder_name == "h264_qsv":
            args.extend([
                "-bf",
                "0",
                "-async_depth",
                "4",
                "-look_ahead",
                "0",
                "-forced_idr",
                "1",
                "-idr_interval",
                "1",
                "-repeat_pps",
                "1",
                "-aud",
                "1",
            ])
        elif self.encoder_name == "h264_nvenc":
            args.extend(["-preset", "p4", "-tune", "ll", "-bf", "0"])
        else:
            args.extend(["-preset", "veryfast", "-tune", "zerolatency", "-x264-params", "repeat-headers=1"])
        args.extend(
            [
                "-b:v",
                f"{self.bitrate_kbps}k",
                "-maxrate",
                f"{maxrate}k",
                "-bufsize",
                f"{bufsize}k",
                "-g",
                "1",
                "-keyint_min",
                "1",
            ]
        )
        return args
"""Pure-Python WebSocket broadcast server for H.264/H.265 compressed video packets.

No external dependencies — stdlib only: socket, select, hashlib, base64, struct, threading.

Architecture
------------
NvdecPacketForwarder  ──►  push_init() / push_packet()  ──►  WebCodecsServer
                                                                    │
                           WebSocket binary stream  ◄──────────────┘
                                    │
                           Browser VideoDecoder  ──►  <canvas>

Wire format
-----------
Text frame (JSON) — sent once per client on connect:
    {"type":"init","codec":"avc1.640028","width":W,"height":H,"description":"BASE64"}
    ``description`` is the base64-encoded AVCDecoderConfigurationRecord (avcC box).

Binary frame — one per compressed video packet:
    Byte  0     : flags   (bit 0 = keyframe)
    Bytes 1–8   : pts_us  uint64 little-endian  (microseconds)
    Bytes 9+    : Annex-B compressed video data (H.264 or H.265)
"""
from __future__ import annotations

import base64
import hashlib
import json
import select
import socket
import struct
import threading

from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning

# RFC 6455 magic GUID for the WebSocket handshake.
_WS_GUID = "258EAFA5-E934-7E53-36CE-014B5F9B0A11"


# ---------------------------------------------------------------------------
# Frame helpers (public for unit-testing)
# ---------------------------------------------------------------------------

def make_accept_key(client_key: str) -> str:
    """Return the ``Sec-WebSocket-Accept`` value for *client_key*."""
    digest = hashlib.sha1((client_key + _WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def make_ws_frame(payload: bytes, opcode: int) -> bytes:
    """Build a single WebSocket frame (server→client, no masking required).

    Parameters
    ----------
    payload:
        Raw payload bytes.
    opcode:
        0x01 = text, 0x02 = binary, 0x08 = close, 0x09 = ping, 0x0A = pong.
    """
    n = len(payload)
    if n < 126:
        header = bytes([0x80 | opcode, n])
    elif n < 65_536:
        header = bytes([0x80 | opcode, 126]) + struct.pack(">H", n)
    else:
        header = bytes([0x80 | opcode, 127]) + struct.pack(">Q", n)
    return header + payload


def build_video_packet_frame(packet: bytes, pts_us: int, is_keyframe: bool) -> bytes:
    """Return WebSocket binary frame for a single compressed video packet."""
    flags = 0x01 if is_keyframe else 0x00
    header = bytes([flags]) + struct.pack("<Q", pts_us)
    return make_ws_frame(header + packet, opcode=0x02)


# ---------------------------------------------------------------------------
# WebCodecsServer
# ---------------------------------------------------------------------------

class WebCodecsServer:
    """Minimal WebSocket server that broadcasts compressed video to browser clients.

    Listens on ``(host, port)``.  Each connecting browser:
      1. Completes the WebSocket upgrade handshake.
      2. Receives a **text** JSON ``init`` message with codec / dimension info.
      3. Receives **binary** video packets until it disconnects.

    The server does NOT require ``flask-sock`` or any third-party library.
    """

    DEFAULT_PORT = 5001

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._accept_thread: threading.Thread | None = None
        self._running = False
        # Most recent stream-init JSON; re-sent to each new client on connect.
        self._init_json: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        """Start the accept loop in a daemon background thread."""
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="wc-accept"
        )
        self._accept_thread.start()
        log_info(LogChannel.GLOBAL, f"WebCodecsServer listening on ws://{self._host}:{self._port}")

    def stop(self) -> None:
        """Signal the accept loop to stop and close all client sockets."""
        self._running = False
        with self._lock:
            for conn in self._clients:
                try:
                    conn.close()
                except OSError:
                    pass
            self._clients.clear()

    # ------------------------------------------------------------------
    # Push API (called by NvdecPacketForwarder)
    # ------------------------------------------------------------------

    def push_init(
        self,
        codec: str,
        width: int,
        height: int,
        avcc: bytes | None = None,
    ) -> None:
        """Store and broadcast a stream-init message.

        Parameters
        ----------
        codec:
            WebCodecs codec string, e.g. ``"avc1.640028"``.
        width, height:
            Native stream dimensions.
        avcc:
            Optional AVCDecoderConfigurationRecord (avcC box) bytes.
            Required by Chrome's ``VideoDecoder`` for H.264.
        """
        payload: dict = {"type": "init", "codec": codec, "width": width, "height": height}
        if avcc:
            payload["description"] = base64.b64encode(avcc).decode()
        self._init_json = json.dumps(payload)
        self._broadcast_text(self._init_json)

    def push_packet(self, packet: bytes, pts_us: int, is_keyframe: bool) -> None:
        """Broadcast a single compressed video packet to all connected clients."""
        if not self._clients:
            return
        msg = build_video_packet_frame(packet, pts_us, is_keyframe)
        self._broadcast_raw(msg)

    # ------------------------------------------------------------------
    # Private: accept loop
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((self._host, self._port))
            srv.listen(8)
            srv.setblocking(False)
        except OSError as exc:
            log_warning(LogChannel.GLOBAL, f"WebCodecsServer bind on port {self._port} failed: {exc}")
            return

        try:
            while self._running:
                readable, _, _ = select.select([srv], [], [], 1.0)
                if not readable:
                    continue
                try:
                    conn, addr = srv.accept()
                except OSError:
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                    name="wc-client",
                ).start()
        finally:
            srv.close()

    # ------------------------------------------------------------------
    # Private: per-client handler
    # ------------------------------------------------------------------

    def _handle_client(self, conn: socket.socket, addr: tuple) -> None:
        """Perform WebSocket handshake, send init, then keep connection alive."""
        ws_key = self._read_upgrade_key(conn)
        if ws_key is None:
            conn.close()
            return

        # Send 101 Switching Protocols.
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {make_accept_key(ws_key)}\r\n"
            "\r\n"
        )
        try:
            conn.sendall(response.encode())
        except OSError:
            conn.close()
            return

        # Send stream-init JSON if available.
        if self._init_json:
            try:
                conn.sendall(make_ws_frame(self._init_json.encode(), opcode=0x01))
            except OSError:
                conn.close()
                return

        log_info(LogChannel.GLOBAL, f"WebCodecs client connected: {addr}")
        with self._lock:
            self._clients.append(conn)

        # Read incoming frames to detect disconnect / handle pings.
        conn.settimeout(30.0)
        while self._running:
            try:
                data = conn.recv(128)
            except (TimeoutError, OSError):
                break
            if not data:
                break
            # Handle ping → pong.
            if (data[0] & 0x0F) == 0x09:
                try:
                    conn.sendall(make_ws_frame(b"", opcode=0x0A))
                except OSError:
                    break

        log_info(LogChannel.GLOBAL, f"WebCodecs client disconnected: {addr}")
        with self._lock:
            try:
                self._clients.remove(conn)
            except ValueError:
                pass
        try:
            conn.close()
        except OSError:
            pass

    @staticmethod
    def _read_upgrade_key(conn: socket.socket) -> str | None:
        """Read the HTTP upgrade request and return ``Sec-WebSocket-Key``, or None."""
        raw = b""
        conn.settimeout(5.0)
        try:
            while b"\r\n\r\n" not in raw:
                chunk = conn.recv(4096)
                if not chunk:
                    return None
                raw += chunk
                if len(raw) > 8192:
                    return None
        except OSError:
            return None

        for line in raw.split(b"\r\n"):
            if line.lower().startswith(b"sec-websocket-key:"):
                return line.split(b":", 1)[1].strip().decode()
        return None

    # ------------------------------------------------------------------
    # Private: broadcast helpers
    # ------------------------------------------------------------------

    def _broadcast_raw(self, msg: bytes) -> None:
        """Send *msg* to all connected clients; remove dead sockets."""
        with self._lock:
            dead: list[socket.socket] = []
            for conn in self._clients:
                try:
                    conn.sendall(msg)
                except OSError:
                    dead.append(conn)
            for c in dead:
                self._clients.remove(c)

    def _broadcast_text(self, text: str) -> None:
        if not self._clients:
            return
        self._broadcast_raw(make_ws_frame(text.encode(), opcode=0x01))

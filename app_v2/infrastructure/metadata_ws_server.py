"""Pure-Python WebSocket server for binary detection metadata.

Wire format (little-endian):
  bytes 0..3   : magic ASCII "PCMB"
  byte  4      : version (1)
  byte  5      : reserved (0)
    bytes 6..7   : flags (uint16)
                                    bit0=1 => centers mode rows float32[3]=[cx,cy,conf]
                                    bit0=0 => bbox mode rows float32[5]=[x1,y1,x2,y2,conf]
  bytes 8..11  : frame_id (uint32)
  bytes 12..15 : detection_count (uint32)
    bytes 16..   : detection rows float32[row_width] * N
"""
from __future__ import annotations

import hashlib
import select
import socket
import struct
import threading
import base64

from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAGIC = b"PCMB"
_VERSION = 1


def _make_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key + _WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _make_ws_frame(payload: bytes, opcode: int) -> bytes:
    payload_size = len(payload)
    if payload_size < 126:
        frame_header = bytes([0x80 | opcode, payload_size])
    elif payload_size < 65_536:
        frame_header = bytes([0x80 | opcode, 126]) + struct.pack(">H", payload_size)
    else:
        frame_header = bytes([0x80 | opcode, 127]) + struct.pack(">Q", payload_size)
    return frame_header + payload


class MetadataWsServer:
    DEFAULT_PORT = 5003

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
        self._host = host
        self._port = port
        self._clients: list[socket.socket] = []
        self._send_failures: dict[socket.socket, int] = {}
        self._lock = threading.Lock()
        self._accept_thread: threading.Thread | None = None
        self._running = False

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        self._running = True
        self._server_bound_event = threading.Event()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="meta-ws-accept"
        )
        self._accept_thread.start()
        self._server_bound_event.wait(timeout=3.0)

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for conn in self._clients:
                try:
                    conn.close()
                except OSError as exc:
                    log_warning(LogChannel.GLOBAL, f"MetadataWsServer client close failed: {exc}")
            self._clients.clear()
            self._send_failures.clear()

    def has_clients(self) -> bool:
        with self._lock:
            return bool(self._clients)

    def push_detections(self, frame_id: int, packed_rows: list[float]) -> None:
        self.push_rows(frame_id, packed_rows, row_width=5, flags=0)

    def push_rows(self, frame_id: int, packed_rows: list[float], *, row_width: int, flags: int = 0) -> None:
        if not self._clients:
            return
        if row_width <= 0:
            log_warning(LogChannel.GLOBAL, f"Metadata WS invalid row_width={row_width}, skipping frame {frame_id}")
            return
        if len(packed_rows) % row_width != 0:
            log_warning(
                LogChannel.GLOBAL,
                f"Metadata WS malformed packed_rows len={len(packed_rows)} row_width={row_width}, frame={frame_id}",
            )
            return
        detection_count = len(packed_rows) // row_width
        payload = bytearray(16 + detection_count * row_width * 4)
        payload[0:4] = _MAGIC
        payload[4] = _VERSION
        payload[5] = 0
        struct.pack_into("<H", payload, 6, int(flags) & 0xFFFF)
        struct.pack_into("<I", payload, 8, max(0, int(frame_id)))
        struct.pack_into("<I", payload, 12, detection_count)
        if detection_count:
            struct.pack_into(f"<{len(packed_rows)}f", payload, 16, *packed_rows)
        self._broadcast_raw(_make_ws_frame(bytes(payload), opcode=0x02))

    def _accept_loop(self) -> None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_socket.bind((self._host, self._port))
        except OSError as exc:
            log_warning(
                LogChannel.GLOBAL,
                f"MetadataWsServer bind on port {self._port} failed ({exc}) — fallback to free port",
            )
            try:
                server_socket.bind((self._host, 0))
            except OSError as exc2:
                log_warning(LogChannel.GLOBAL, f"MetadataWsServer fallback bind failed: {exc2}")
                if hasattr(self, "_server_bound_event"):
                    self._server_bound_event.set()
                return

        _, self._port = server_socket.getsockname()
        try:
            server_socket.listen(8)
            server_socket.setblocking(False)
        except OSError as exc:
            log_warning(LogChannel.GLOBAL, f"MetadataWsServer listen failed: {exc}")
            if hasattr(self, "_server_bound_event"):
                self._server_bound_event.set()
            return

        log_info(LogChannel.GLOBAL, f"MetadataWsServer listening on ws://{self._host}:{self._port}")
        if hasattr(self, "_server_bound_event"):
            self._server_bound_event.set()

        try:
            while self._running:
                ready_to_accept, _, _ = select.select([server_socket], [], [], 1.0)
                if not ready_to_accept:
                    continue
                try:
                    client_socket, client_address = server_socket.accept()
                except OSError as exc:
                    log_warning(LogChannel.GLOBAL, f"MetadataWsServer accept failed: {exc}")
                    break
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True,
                    name="meta-ws-client",
                ).start()
        finally:
            server_socket.close()

    def _handle_client(self, client_socket: socket.socket, client_address: tuple) -> None:
        websocket_key = self._read_upgrade_key(client_socket)
        if websocket_key is None:
            log_warning(
                LogChannel.GLOBAL,
                f"Metadata WS handshake failed: missing upgrade key from {client_address}",
            )
            client_socket.close()
            return

        upgrade_response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {_make_accept_key(websocket_key)}\r\n"
            "\r\n"
        )
        try:
            client_socket.sendall(upgrade_response.encode())
        except OSError as exc:
            log_warning(LogChannel.GLOBAL, f"Metadata WS handshake response failed for {client_address}: {exc}")
            client_socket.close()
            return

        log_info(LogChannel.GLOBAL, f"Metadata WS client connected: {client_address}")
        with self._lock:
            self._clients.append(client_socket)
            self._send_failures[client_socket] = 0

        client_socket.setblocking(False)
        while self._running:
            ready_to_read, _, _ = select.select([client_socket], [], [], 1.0)
            if not ready_to_read:
                continue
            try:
                inbound_frame = client_socket.recv(128)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                log_warning(LogChannel.GLOBAL, f"Metadata WS recv failed from {client_address}: {exc}")
                break
            if not inbound_frame:
                break
            if (inbound_frame[0] & 0x0F) == 0x09:  # ping -> pong
                try:
                    client_socket.sendall(_make_ws_frame(b"", opcode=0x0A))
                except OSError as exc:
                    log_warning(LogChannel.GLOBAL, f"Metadata WS pong send failed to {client_address}: {exc}")
                    break

        log_info(LogChannel.GLOBAL, f"Metadata WS client disconnected: {client_address}")
        with self._lock:
            try:
                self._clients.remove(client_socket)
            except ValueError:
                log_warning(
                    LogChannel.GLOBAL,
                    f"Metadata WS disconnect: socket not found in client list for {client_address}",
                )
            self._send_failures.pop(client_socket, None)
        try:
            client_socket.close()
        except OSError as exc:
            log_warning(LogChannel.GLOBAL, f"Metadata WS close failed for {client_address}: {exc}")

    @staticmethod
    def _read_upgrade_key(conn: socket.socket) -> str | None:
        raw_request = b""
        conn.settimeout(5.0)
        try:
            while b"\r\n\r\n" not in raw_request:
                request_chunk = conn.recv(4096)
                if not request_chunk:
                    return None
                raw_request += request_chunk
                if len(raw_request) > 8192:
                    log_warning(LogChannel.GLOBAL, "Metadata WS upgrade request exceeded 8192 bytes")
                    return None
        except OSError as exc:
            log_warning(LogChannel.GLOBAL, f"Metadata WS upgrade read failed: {exc}")
            return None

        for header_line in raw_request.split(b"\r\n"):
            if header_line.lower().startswith(b"sec-websocket-key:"):
                return header_line.split(b":", 1)[1].strip().decode()
        return None

    def _broadcast_raw(self, framed_message: bytes) -> None:
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        disconnected_clients: list[socket.socket] = []
        for client_socket in clients:
            try:
                client_socket.sendall(framed_message)
                with self._lock:
                    self._send_failures[client_socket] = 0
            except (BlockingIOError, InterruptedError):
                # Transient backpressure: skip this frame for this client.
                # Keep socket alive to avoid unnecessary reconnect churn.
                with self._lock:
                    consecutive_send_failures = self._send_failures.get(client_socket, 0) + 1
                    self._send_failures[client_socket] = consecutive_send_failures
                if consecutive_send_failures >= 60:
                    log_warning(LogChannel.GLOBAL, "Metadata WS client stuck (>=60 send retries), dropping socket")
                    disconnected_clients.append(client_socket)
            except OSError as exc:
                log_warning(LogChannel.GLOBAL, f"Metadata WS broadcast send failed: {exc}")
                disconnected_clients.append(client_socket)

        if disconnected_clients:
            with self._lock:
                for client_socket in disconnected_clients:
                    try:
                        self._clients.remove(client_socket)
                    except ValueError:
                        log_warning(LogChannel.GLOBAL, "Metadata WS dead socket already removed")
                    self._send_failures.pop(client_socket, None)
                    try:
                        client_socket.close()
                    except OSError as exc:
                        log_warning(LogChannel.GLOBAL, f"Metadata WS dead socket close failed: {exc}")

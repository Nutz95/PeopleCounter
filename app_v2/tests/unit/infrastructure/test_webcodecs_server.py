"""Unit tests for app_v2.infrastructure.webcodecs_server."""
from __future__ import annotations

import base64
import json
import socket
import struct
import time

import pytest

from app_v2.infrastructure.webcodecs_server import (
    WebCodecsServer,
    build_video_packet_frame,
    make_accept_key,
    make_ws_frame,
)

# ---------------------------------------------------------------------------
# Frame / handshake helpers
# ---------------------------------------------------------------------------

class TestMakeAcceptKey:
    def test_output_is_valid_base64(self):
        result = make_accept_key("anykey123==")
        import base64
        # Must be valid base64 of a 20-byte SHA-1 digest → always 28 chars with padding
        decoded = base64.b64decode(result)
        assert len(decoded) == 20     # SHA-1 produces 20 bytes

    def test_deterministic(self):
        key = "AAAAAAAAAAAAAAAAAAAAAA=="
        assert make_accept_key(key) == make_accept_key(key)

    def test_different_keys_give_different_accepts(self):
        assert make_accept_key("key1") != make_accept_key("key2")


class TestMakeWsFrame:
    def test_small_payload_uses_2_byte_header(self):
        payload = b"hello"
        frame = make_ws_frame(payload, opcode=0x02)
        assert len(frame) == 2 + len(payload)
        assert frame[0] == 0x80 | 0x02
        assert frame[1] == len(payload)

    def test_medium_payload_uses_4_byte_header(self):
        payload = bytes(200)
        frame = make_ws_frame(payload, opcode=0x02)
        assert frame[0] == 0x82
        assert frame[1] == 126
        assert struct.unpack(">H", frame[2:4])[0] == 200
        assert len(frame) == 4 + 200

    def test_large_payload_uses_10_byte_header(self):
        payload = bytes(70_000)
        frame = make_ws_frame(payload, opcode=0x01)
        assert frame[1] == 127
        assert struct.unpack(">Q", frame[2:10])[0] == 70_000
        assert len(frame) == 10 + 70_000

    def test_empty_payload(self):
        assert make_ws_frame(b"", opcode=0x0A) == bytes([0x8A, 0x00])

    def test_exactly_125_bytes_is_small(self):
        frame = make_ws_frame(bytes(125), opcode=0x02)
        assert frame[1] == 125 and len(frame) == 127

    def test_exactly_126_bytes_is_medium(self):
        frame = make_ws_frame(bytes(126), opcode=0x02)
        assert frame[1] == 126 and len(frame) == 4 + 126


class TestBuildVideoPacketFrame:
    def test_keyframe_flag(self):
        frame = build_video_packet_frame(b"\x00\x00\x00\x01\x67", pts_us=0, is_keyframe=True)
        assert frame[2] == 0x01   # flags byte (after 2-byte WS header)

    def test_delta_frame_flag(self):
        frame = build_video_packet_frame(b"\x00\x00\x00\x01\x41", pts_us=0, is_keyframe=False)
        assert frame[2] == 0x00

    def test_pts_encoding(self):
        pts = 12345678
        frame = build_video_packet_frame(b"data", pts_us=pts, is_keyframe=False)
        assert struct.unpack("<Q", frame[3:11])[0] == pts   # 2 WS header + 1 flags = offset 3

    def test_packet_data_appended(self):
        pkt = b"\xAB\xCD\xEF"
        frame = build_video_packet_frame(pkt, pts_us=0, is_keyframe=False)
        assert frame[2 + 1 + 8:] == pkt   # 2 header + 1 flags + 8 pts


# ---------------------------------------------------------------------------
# WebCodecsServer lifecycle (no networking)
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestWebCodecsServerLifecycle:
    def test_start_and_stop_no_exception(self):
        srv = WebCodecsServer(host="127.0.0.1", port=_free_port())
        srv.start(); time.sleep(0.05); srv.stop()

    def test_port_property(self):
        assert WebCodecsServer(port=9876).port == 9876

    def test_push_to_no_clients_is_noop(self):
        srv = WebCodecsServer(host="127.0.0.1", port=_free_port())
        srv.start()
        srv.push_packet(b"\x00", pts_us=0, is_keyframe=True)
        srv.push_init("avc1.640028", 1920, 1080)
        srv.stop()

    def test_start_twice_is_idempotent(self):
        srv = WebCodecsServer(host="127.0.0.1", port=_free_port())
        srv.start(); time.sleep(0.02); srv.start(); srv.stop()


# ---------------------------------------------------------------------------
# Buffered WebSocket client helper
# ---------------------------------------------------------------------------

class _WsClient:
    """Wraps a WebSocket socket with a read buffer to handle TCP coalescing."""

    def __init__(self, conn: socket.socket, initial_buf: bytes = b"") -> None:
        self._conn = conn
        self._buf = initial_buf

    def _read_exactly(self, n: int) -> bytes:
        self._conn.settimeout(2.0)
        while len(self._buf) < n:
            chunk = self._conn.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            self._buf += chunk
        result, self._buf = self._buf[:n], self._buf[n:]
        return result

    def _frame_length(self, header: bytes) -> int:
        n = header[1] & 0x7F
        if n == 126:
            return struct.unpack(">H", self._read_exactly(2))[0]
        if n == 127:
            return struct.unpack(">Q", self._read_exactly(8))[0]
        return n

    def read_text(self) -> str:
        h = self._read_exactly(2)
        assert (h[0] & 0x0F) == 0x01, f"expected text, got opcode 0x{h[0] & 0x0F:02X}"
        return self._read_exactly(self._frame_length(h)).decode()

    def read_binary(self) -> bytes:
        h = self._read_exactly(2)
        assert (h[0] & 0x0F) == 0x02, f"expected binary, got opcode 0x{h[0] & 0x0F:02X}"
        return self._read_exactly(self._frame_length(h))

    def close(self) -> None:
        self._conn.close()


def _ws_connect(host: str, port: int) -> _WsClient:
    """Open TCP socket, complete WebSocket upgrade, return buffered client."""
    key_b64 = base64.b64encode(b"TestKey1234567890123==").decode()
    request = (
        f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n"
        f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key_b64}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    conn = socket.create_connection((host, port), timeout=3.0)
    conn.sendall(request.encode())

    raw = b""
    conn.settimeout(3.0)
    while True:
        chunk = conn.recv(4096)
        raw += chunk
        idx = raw.find(b"\r\n\r\n")
        if idx >= 0:
            overflow = raw[idx + 4:]
            break
        if not chunk:
            overflow = b""
            break

    assert b"101 Switching Protocols" in raw
    return _WsClient(conn, initial_buf=overflow)


# ---------------------------------------------------------------------------
# Integration-style tests: real server + socket client
# ---------------------------------------------------------------------------

class TestWebCodecsServerHandshake:
    def setup_method(self):
        self.port = _free_port()
        self.srv  = WebCodecsServer(host="127.0.0.1", port=self.port)
        self.srv.start()
        time.sleep(0.05)

    def teardown_method(self):
        self.srv.stop()

    def test_client_receives_init_json(self):
        self.srv.push_init("avc1.640028", 1920, 1080)
        c = _ws_connect("127.0.0.1", self.port)
        try:
            msg = json.loads(c.read_text())
            assert msg == {"type": "init", "codec": "avc1.640028", "width": 1920, "height": 1080}
        finally:
            c.close()

    def test_init_description_base64(self):
        avcc = b"\x01\x64\x00\x28\xFF"
        self.srv.push_init("avc1.640028", 3840, 2160, avcc=avcc)
        c = _ws_connect("127.0.0.1", self.port)
        try:
            msg = json.loads(c.read_text())
            assert base64.b64decode(msg["description"]) == avcc
        finally:
            c.close()

    def test_client_receives_binary_packet(self):
        self.srv.push_init("avc1", 640, 480)
        c = _ws_connect("127.0.0.1", self.port)
        try:
            c.read_text()          # consume init
            time.sleep(0.01)
            pkt = b"\x00\x00\x00\x01\x67\xAB\xCD"
            self.srv.push_packet(pkt, pts_us=123456, is_keyframe=True)
            payload = c.read_binary()
            assert payload[0] == 0x01                           # keyframe flag
            assert struct.unpack("<Q", payload[1:9])[0] == 123456
            assert payload[9:] == pkt
        finally:
            c.close()

    def test_two_clients_both_receive_packet(self):
        self.srv.push_init("avc1", 640, 480)
        c1 = _ws_connect("127.0.0.1", self.port)
        c2 = _ws_connect("127.0.0.1", self.port)
        try:
            c1.read_text(); c2.read_text()
            time.sleep(0.01)
            pkt = b"\xFF\xEE"
            self.srv.push_packet(pkt, pts_us=0, is_keyframe=False)
            assert c1.read_binary()[9:] == pkt
            assert c2.read_binary()[9:] == pkt
        finally:
            c1.close(); c2.close()

    def test_dead_client_removed_on_broadcast(self):
        self.srv.push_init("avc1", 640, 480)
        c = _ws_connect("127.0.0.1", self.port)
        c.read_text()
        time.sleep(0.01)
        c.close()
        time.sleep(0.05)
        # push_packet must not crash when broadcasting to closed socket
        self.srv.push_packet(b"\x00", pts_us=0, is_keyframe=False)

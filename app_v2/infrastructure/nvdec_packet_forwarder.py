"""PyFFmpegDemuxer-based raw video packet tap for WebCodecs streaming.

Opens a **second** connection to the same RTSP/HTTP source alongside the
NvdecDecoder's own internal decode loop.  The two connections are independent;
the camera / HTTP server must support concurrent clients (all common cases do).

The forwarder:
  1. Opens a ``PyFFmpegDemuxer`` in a daemon background thread.
  2. On the first keyframe that contains SPS + PPS (H.264), builds the
     AVCDecoderConfigurationRecord (avcC box) and calls
     ``ws_server.push_init()`` so the browser can configure ``VideoDecoder``.
  3. Calls ``ws_server.push_packet()`` for every subsequent packet.

If ``PyNvCodec`` / ``PyFFmpegDemuxer`` is unavailable or the codec is not
H.264/H.265, the forwarder exits silently — MJPEG fallback remains active.
"""
from __future__ import annotations

import importlib
import struct
import threading
import time
from typing import Any

from logger.filtered_logger import LogChannel, info as log_info, warning as log_warning

# ---------------------------------------------------------------------------
# Annex-B bitstream helpers (public so tests can import them directly)
# ---------------------------------------------------------------------------

def find_start_codes(data: bytes) -> list[tuple[int, int]]:
    """Return ``(byte_offset, start_code_len)`` for every Annex-B start code."""
    result: list[tuple[int, int]] = []
    i = 0
    n = len(data)
    while i < n - 2:
        b0, b1, b2 = data[i], data[i + 1], data[i + 2]
        if b0 == 0 and b1 == 0:
            if b2 == 0 and i + 3 < n and data[i + 3] == 1:
                result.append((i, 4))
                i += 4
                continue
            if b2 == 1:
                result.append((i, 3))
                i += 3
                continue
        i += 1
    return result


def split_nalus(data: bytes) -> list[bytes]:
    """Split Annex-B bitstream into raw NAL unit bodies (start codes stripped)."""
    scs = find_start_codes(data)
    out: list[bytes] = []
    for j, (pos, sc_len) in enumerate(scs):
        start = pos + sc_len
        end = scs[j + 1][0] if j + 1 < len(scs) else len(data)
        if start < end:
            out.append(data[start:end])
    return out


def extract_avcc(packet: bytes) -> tuple[str, bytes | None]:
    """Extract ``(codec_string, avcC_box)`` from an Annex-B H.264 keyframe.

    Returns
    -------
    codec_str
        WebCodecs codec string like ``"avc1.640028"`` derived from the SPS.
        Falls back to ``"avc1"`` when SPS cannot be located.
    avcc
        Raw AVCDecoderConfigurationRecord bytes, or ``None`` if SPS/PPS are
        missing from the packet.
    """
    sps_body: bytes | None = None
    pps_body: bytes | None = None

    for nal in split_nalus(packet):
        if not nal:
            continue
        nal_type = nal[0] & 0x1F
        if nal_type == 7 and sps_body is None:    # SPS
            sps_body = nal
        elif nal_type == 8 and pps_body is None:  # PPS
            pps_body = nal

    if sps_body is None or pps_body is None or len(sps_body) < 4:
        return "avc1", None

    profile = sps_body[1]
    compat  = sps_body[2]
    level   = sps_body[3]
    codec_str = f"avc1.{profile:02X}{compat:02X}{level:02X}"

    # Build AVCDecoderConfigurationRecord (ISO 14496-15 §5.3.3.1.2)
    avcc = (
        bytes([0x01, profile, compat, level, 0xFF, 0xE1])
        + struct.pack(">H", len(sps_body))
        + sps_body
        + bytes([0x01])
        + struct.pack(">H", len(pps_body))
        + pps_body
    )
    return codec_str, avcc


def is_keyframe(packet: bytes, pkt_data: Any) -> bool:
    """Return True if the packet is a keyframe (IDR or CRA for H.265).

    Checks ``PacketData`` attributes first, then falls back to NAL unit scan.
    """
    if pkt_data is not None:
        for attr in ("bKeyFrame", "key", "IsKeyFrame", "KeyFrame"):
            val = getattr(pkt_data, attr, None)
            if val is not None:
                return bool(val)

    for nal in split_nalus(packet):
        if not nal:
            continue
        nal_type = nal[0] & 0x1F
        if nal_type == 5:   # H.264 IDR slice
            return True
        hevc_type = (nal[0] >> 1) & 0x3F
        if hevc_type in (19, 20, 21):   # HEVC IDR_W_RADL / IDR_N_LP / CRA_NUT
            return True
    return False


# ---------------------------------------------------------------------------
# NvdecPacketForwarder
# ---------------------------------------------------------------------------

class NvdecPacketForwarder:
    """Demux compressed video packets and forward them to a WebCodecsServer.

    Opens a ``PyFFmpegDemuxer`` in a daemon thread.  Runs alongside the main
    ``NvdecDecoder`` without interfering with inference.
    """

    def __init__(
        self,
        stream_url: str,
        ws_server: Any,
        decoder_opts: dict | None = None,
    ) -> None:
        self._url = stream_url
        self._ws_server = ws_server
        self._opts = dict(decoder_opts or {})
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="nvdec-pkt-fwd"
        )
        self._thread.start()
        log_info(LogChannel.GLOBAL, f"NvdecPacketForwarder started for {self._url}")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Private: demux loop
    # ------------------------------------------------------------------

    # Maximum delay (seconds) between reconnection attempts.
    _MAX_RECONNECT_DELAY = 8.0

    def _run(self) -> None:
        try:
            import numpy as np  # type: ignore[import]
            nvc = importlib.import_module("PyNvCodec")
        except ImportError as exc:
            log_warning(LogChannel.GLOBAL, f"PyNvCodec/numpy unavailable — WebCodecs packet forwarder disabled: {exc}")
            return

        reconnect_delay = 2.0  # seconds; doubles on each failure up to _MAX_RECONNECT_DELAY

        while self._running:
            eof = self._run_demux_once(nvc)
            if not self._running:
                break
            if eof:
                # Clean EOF (stream ended); stop without reconnecting.
                log_info(LogChannel.GLOBAL, "NvdecPacketForwarder: stream EOF — stopping")
                break
            # Transient error — reconnect after a short pause.
            log_info(LogChannel.GLOBAL, f"NvdecPacketForwarder: reconnecting in {reconnect_delay:.1f} s…")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, self._MAX_RECONNECT_DELAY)

        log_info(LogChannel.GLOBAL, "NvdecPacketForwarder run loop exited")

    def _run_demux_once(self, nvc: Any) -> bool:
        """Run one demux session.  Returns True on clean EOF, False on error."""
        import numpy as np  # type: ignore[import]

        demuxer = self._create_demuxer(nvc)
        if demuxer is None:
            return False  # Error (not EOF)

        # Read stream dimensions from the demuxer.
        width  = self._demuxer_int(demuxer, "Width")
        height = self._demuxer_int(demuxer, "Height")

        pkt_data_cls = getattr(nvc, "PacketData", None)
        pkt_data = pkt_data_cls() if pkt_data_cls is not None else None

        packet = np.zeros(shape=(0,), dtype=np.uint8)

        avcc_sent = False
        t0_ns = time.time_ns()

        while self._running:
            try:
                if pkt_data is not None:
                    success = demuxer.DemuxSinglePacket(packet, pkt_data)
                else:
                    success = demuxer.DemuxSinglePacket(packet)
            except Exception as exc:
                log_warning(LogChannel.GLOBAL, f"DemuxSinglePacket error: {exc}")
                return False  # Transient error — caller will reconnect

            if not success:
                return True  # Clean EOF

            pkt_bytes = bytes(packet)
            if not pkt_bytes:
                continue

            is_kf = is_keyframe(pkt_bytes, pkt_data)

            # On the first keyframe with SPS+PPS, send init message.
            if is_kf and not avcc_sent:
                codec_str, avcc = extract_avcc(pkt_bytes)
                self._ws_server.push_init(codec_str, width, height, avcc)
                avcc_sent = True

            pts = self._pts_us(pkt_data, t0_ns)
            self._ws_server.push_packet(pkt_bytes, pts, is_kf)

        return True  # Stopped cleanly via self._running = False

    def _create_demuxer(self, nvc: Any) -> Any | None:
        """Try multiple ``PyFFmpegDemuxer`` constructor signatures."""
        for args in [
            (self._url, self._opts),
            (self._url,),
        ]:
            try:
                dmx = nvc.PyFFmpegDemuxer(*args)
                log_info(LogChannel.GLOBAL, f"PyFFmpegDemuxer opened (nargs={len(args)})")
                return dmx
            except Exception as exc:
                log_warning(LogChannel.GLOBAL, f"PyFFmpegDemuxer({args[:1]!r}…) failed: {exc}")
        return None

    @staticmethod
    def _demuxer_int(demuxer: Any, attr: str) -> int:
        """Read a numeric attribute from the demuxer (method or property)."""
        val = getattr(demuxer, attr, None)
        if callable(val):
            try:
                return int(val())
            except Exception:
                return 0
        if isinstance(val, int):
            return val
        return 0

    @staticmethod
    def _pts_us(pkt_data: Any, t0_ns: int) -> int:
        """Convert PacketData PTS to microseconds, falling back to wall clock."""
        if pkt_data is not None:
            for attr in ("pts", "PTS", "Pts"):
                val = getattr(pkt_data, attr, None)
                if isinstance(val, (int, float)) and val > 0:
                    # H.264 RTSP streams use 90 kHz timebase by convention.
                    # Heuristic: values < 1e12 are likely ticks, not nanoseconds.
                    if val < 1e12:
                        return int(val * 1_000_000 / 90_000)
                    # Assume nanoseconds → convert to microseconds.
                    return int(val // 1_000)
        return (time.time_ns() - t0_ns) // 1_000

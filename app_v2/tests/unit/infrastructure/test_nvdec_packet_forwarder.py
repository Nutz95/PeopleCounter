"""Unit tests for app_v2.infrastructure.nvdec_packet_forwarder."""
from __future__ import annotations

import struct
import time
from unittest.mock import MagicMock, patch

import pytest

from app_v2.infrastructure.nvdec_packet_forwarder import (
    NvdecPacketForwarder,
    extract_avcc,
    find_start_codes,
    is_keyframe,
    split_nalus,
)

# ---------------------------------------------------------------------------
# Annex-B / NAL unit helpers
# ---------------------------------------------------------------------------

class TestFindStartCodes:
    def test_single_4byte_start_code(self):
        data = b"\x00\x00\x00\x01\x67\xAB"
        scs = find_start_codes(data)
        assert scs == [(0, 4)]

    def test_single_3byte_start_code(self):
        data = b"\x00\x00\x01\x61\xAB"
        scs = find_start_codes(data)
        assert scs == [(0, 3)]

    def test_multiple_start_codes(self):
        data = (
            b"\x00\x00\x00\x01\x67\xAB"    # SPS at 0
            b"\x00\x00\x00\x01\x68\xCD"    # PPS at 6
            b"\x00\x00\x00\x01\x65\xEF"    # IDR at 12
        )
        scs = find_start_codes(data)
        assert len(scs) == 3
        assert scs[0][0] == 0
        assert scs[1][0] == 6
        assert scs[2][0] == 12

    def test_no_start_codes(self):
        assert find_start_codes(b"\xAB\xCD\xEF\x12") == []

    def test_empty_data(self):
        assert find_start_codes(b"") == []


class TestSplitNalus:
    def test_two_nalus(self):
        data = b"\x00\x00\x00\x01\x67\xAB\x00\x00\x00\x01\x68\xCD"
        nalus = split_nalus(data)
        assert len(nalus) == 2
        assert nalus[0] == b"\x67\xAB"
        assert nalus[1] == b"\x68\xCD"

    def test_empty_input(self):
        assert split_nalus(b"") == []


class TestExtractAvcc:
    def _make_sps(self, profile=100, compat=0, level=40) -> bytes:
        # Minimal valid SPS NAL: type byte + 3 profile/level bytes + padding
        return bytes([0x67, profile, compat, level, 0x00, 0x00])

    def _make_pps(self) -> bytes:
        return bytes([0x68, 0xEB, 0xEF, 0x20])

    def _annex_b(self, *nals: bytes) -> bytes:
        return b"".join(b"\x00\x00\x00\x01" + n for n in nals)

    def test_no_sps_returns_generic_codec(self):
        # Packet with no SPS/PPS
        idr = bytes([0x65, 0x88, 0x84, 0x00, 0x33])
        packet = self._annex_b(idr)
        codec_str, avcc = extract_avcc(packet)
        assert codec_str == "avc1"
        assert avcc is None

    def test_codec_string_from_sps(self):
        sps = self._make_sps(profile=100, compat=0, level=40)   # High 4.0
        pps = self._make_pps()
        packet = self._annex_b(sps, pps)
        codec_str, _ = extract_avcc(packet)
        assert codec_str == "avc1.640028"   # 100=0x64, 0=0x00, 40=0x28

    def test_avcc_structure(self):
        sps = self._make_sps(profile=100, compat=0, level=40)
        pps = self._make_pps()
        packet = self._annex_b(sps, pps)
        _, avcc = extract_avcc(packet)
        assert avcc is not None
        # configurationVersion
        assert avcc[0] == 0x01
        # profile, compat, level from SPS
        assert avcc[1:4] == bytes([100, 0, 40])
        # lengthSizeMinusOne = 0xFF
        assert avcc[4] == 0xFF
        # numSPS marker (0xE1 = 0b111_00001 = 1 SPS)
        assert avcc[5] == 0xE1
        # SPS length
        sps_len = struct.unpack(">H", avcc[6:8])[0]
        assert sps_len == len(sps)
        # SPS body
        assert avcc[8:8 + sps_len] == sps
        # numPPS
        assert avcc[8 + sps_len] == 0x01
        # PPS length
        pps_len = struct.unpack(">H", avcc[9 + sps_len:11 + sps_len])[0]
        assert pps_len == len(pps)

    def test_sps_too_short_returns_none(self):
        short_sps = bytes([0x67, 0x64])   # only 2 bytes — not enough for profile/level
        pps = self._make_pps()
        packet = self._annex_b(short_sps, pps)
        _, avcc = extract_avcc(packet)
        assert avcc is None

    def test_idr_with_sps_pps(self):
        sps = self._make_sps()
        pps = self._make_pps()
        idr = bytes([0x65, 0x88, 0x84])
        packet = self._annex_b(sps, pps, idr)
        codec_str, avcc = extract_avcc(packet)
        assert avcc is not None            # SPS+PPS present → avcC built


class TestIsKeyframe:
    def _make_packet(self, *nal_types: int) -> bytes:
        out = b""
        for nt in nal_types:
            out += b"\x00\x00\x00\x01" + bytes([nt]) + b"\x00\x00"
        return out

    def test_idr_nal_is_keyframe(self):
        packet = self._make_packet(0x65)   # NAL type 5 = IDR (0x65 & 0x1F == 5)
        assert is_keyframe(packet, None) is True

    def test_p_frame_nal_is_not_keyframe(self):
        packet = self._make_packet(0x61)   # NAL type 1 = non-IDR slice
        assert is_keyframe(packet, None) is False

    def test_pkt_data_attr_bKeyFrame_true(self):
        pkt_data = MagicMock()
        pkt_data.bKeyFrame = True
        del pkt_data.key
        del pkt_data.IsKeyFrame
        del pkt_data.KeyFrame
        assert is_keyframe(b"", pkt_data) is True

    def test_pkt_data_attr_bKeyFrame_false(self):
        pkt_data = MagicMock()
        pkt_data.bKeyFrame = False
        del pkt_data.key
        del pkt_data.IsKeyFrame
        del pkt_data.KeyFrame
        assert is_keyframe(b"", pkt_data) is False

    def test_pkt_data_takes_priority_over_scan(self):
        # pkt_data says it's NOT a keyframe, but packet has IDR NAL
        pkt_data = MagicMock()
        pkt_data.bKeyFrame = False
        del pkt_data.key
        del pkt_data.IsKeyFrame
        del pkt_data.KeyFrame
        idr_packet = self._make_packet(0x65)
        # pkt_data wins
        assert is_keyframe(idr_packet, pkt_data) is False

    def test_none_pkt_data_falls_back_to_scan(self):
        idr_packet = self._make_packet(0x65)
        assert is_keyframe(idr_packet, None) is True


# ---------------------------------------------------------------------------
# NvdecPacketForwarder constructor / start / stop
# ---------------------------------------------------------------------------

class TestNvdecPacketForwarder:
    def test_start_stops_without_pynvcodec(self):
        """Forwarder should exit its thread gracefully when PyNvCodec is missing."""
        ws_mock = MagicMock()
        fwd = NvdecPacketForwarder("rtsp://nowhere/stream", ws_mock)
        with patch.dict("sys.modules", {"PyNvCodec": None}):
            fwd.start()
            fwd.stop()
            # Give the daemon thread time to exit.
            if fwd._thread:
                fwd._thread.join(timeout=2.0)
        # ws_server should not have been called
        ws_mock.push_init.assert_not_called()
        ws_mock.push_packet.assert_not_called()

    def test_stop_before_start_is_safe(self):
        ws_mock = MagicMock()
        fwd = NvdecPacketForwarder("rtsp://nowhere/stream", ws_mock)
        fwd.stop()   # must not raise

    def test_pts_us_wall_clock_fallback(self):
        t0 = time.time_ns()
        pts = NvdecPacketForwarder._pts_us(None, t0)
        assert pts >= 0
        assert pts < 1_000_000  # started less than 1 second ago

    def test_pts_us_from_pkt_data(self):
        pkt_data = MagicMock()
        pkt_data.pts = 900_000    # 10 seconds at 90 kHz
        del pkt_data.PTS
        del pkt_data.Pts
        pts = NvdecPacketForwarder._pts_us(pkt_data, 0)
        assert pts == 10_000_000  # 10 s in µs

    def test_demuxer_int_callable(self):
        obj = MagicMock()
        obj.Width = MagicMock(return_value=1920)
        result = NvdecPacketForwarder._demuxer_int(obj, "Width")
        assert result == 1920

    def test_demuxer_int_attribute(self):
        obj = MagicMock()
        obj.Width = 3840
        result = NvdecPacketForwarder._demuxer_int(obj, "Width")
        assert result == 3840

    def test_demuxer_int_missing(self):
        obj = MagicMock(spec=[])
        result = NvdecPacketForwarder._demuxer_int(obj, "Width")
        assert result == 0

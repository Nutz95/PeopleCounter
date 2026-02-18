"""Unit tests for app_v2.infrastructure.engine_stream_writer (Opt #6 — IStreamWriter).

TRT is mocked so tests run without a GPU / TRT installation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock tensorrt (IStreamWriter base class) before the import
# ---------------------------------------------------------------------------
_fake_trt = MagicMock()

class _FakeIStreamWriter:
    """Minimal stand-in for trt.IStreamWriter."""
    def __init__(self):
        pass

_fake_trt.IStreamWriter = _FakeIStreamWriter
sys.modules.setdefault("tensorrt", _fake_trt)

from app_v2.infrastructure.engine_stream_writer import (  # noqa: E402
    FileStreamWriter,
    BytesStreamWriter,
)


# ---------------------------------------------------------------------------
# FileStreamWriter tests
# ---------------------------------------------------------------------------


class TestFileStreamWriter:
    def test_write_creates_file(self, tmp_path: Path):
        out = tmp_path / "engine.trt"
        writer = FileStreamWriter(out)
        writer.write(b"hello")
        writer.write(b" world")
        writer.close()

        assert out.exists()
        assert out.read_bytes() == b"hello world"

    def test_bytes_written_property(self, tmp_path: Path):
        out = tmp_path / "engine.trt"
        writer = FileStreamWriter(out)
        writer.write(b"abc")
        writer.write(b"de")
        writer.close()

        assert writer.bytes_written == 5

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "deep" / "nested" / "dir" / "engine.trt"
        writer = FileStreamWriter(out)
        writer.write(b"\x00\x01\x02")
        writer.close()

        assert out.exists()

    def test_context_manager_closes_file(self, tmp_path: Path):
        out = tmp_path / "engine.trt"
        with FileStreamWriter(out) as writer:
            writer.write(b"data")

        assert out.read_bytes() == b"data"

    def test_write_returns_chunk_length(self, tmp_path: Path):
        out = tmp_path / "engine.trt"
        writer = FileStreamWriter(out)
        chunk = b"chunk of bytes"
        result = writer.write(chunk)
        writer.close()

        assert result == len(chunk)

    def test_path_property(self, tmp_path: Path):
        out = tmp_path / "engine.trt"
        writer = FileStreamWriter(out)
        assert writer.path == out
        writer.close()

    def test_empty_write_produces_empty_file(self, tmp_path: Path):
        out = tmp_path / "engine.trt"
        with FileStreamWriter(out) as writer:
            pass  # write nothing

        assert out.exists()
        assert out.read_bytes() == b""
        assert writer.bytes_written == 0


# ---------------------------------------------------------------------------
# BytesStreamWriter tests
# ---------------------------------------------------------------------------


class TestBytesStreamWriter:
    def test_collects_chunks_in_order(self):
        writer = BytesStreamWriter()
        writer.write(b"part1")
        writer.write(b"part2")
        writer.write(b"part3")
        assert writer.bytes == b"part1part2part3"

    def test_bytes_written_property(self):
        writer = BytesStreamWriter()
        writer.write(b"abc")
        writer.write(b"de")
        assert writer.bytes_written == 5

    def test_empty_writer(self):
        writer = BytesStreamWriter()
        assert writer.bytes == b""
        assert writer.bytes_written == 0

    def test_write_returns_chunk_length(self):
        writer = BytesStreamWriter()
        chunk = b"test data"
        result = writer.write(chunk)
        assert result == len(chunk)

    def test_bytes_is_immutable_copy(self):
        writer = BytesStreamWriter()
        writer.write(b"hello")
        result = writer.bytes
        assert isinstance(result, bytes)
        # Further writes should not mutate already-retrieved bytes
        writer.write(b" world")
        assert result == b"hello"  # the bytes() snapshot is independent

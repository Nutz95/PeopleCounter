from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app_v2.infrastructure.timing_cache_manager import TimingCacheManager


class TestTimingCacheManagerInit:
    def test_path_stored(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "timing_cache.bin"
        manager = TimingCacheManager(cache_path)
        assert manager.path == cache_path

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        manager = TimingCacheManager(str(tmp_path / "cache.bin"))
        assert isinstance(manager.path, Path)


class TestTimingCacheManagerExists:
    def test_not_exists_when_no_file(self, tmp_path: Path) -> None:
        manager = TimingCacheManager(tmp_path / "missing.bin")
        assert not manager.exists()

    def test_not_exists_when_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        manager = TimingCacheManager(p)
        assert not manager.exists()

    def test_exists_when_file_has_content(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.bin"
        p.write_bytes(b"\x00" * 64)
        manager = TimingCacheManager(p)
        assert manager.exists()


class TestTimingCacheManagerInvalidate:
    def test_invalidate_deletes_file(self, tmp_path: Path) -> None:
        p = tmp_path / "cache.bin"
        p.write_bytes(b"data")
        manager = TimingCacheManager(p)
        manager.invalidate()
        assert not p.exists()

    def test_invalidate_is_idempotent(self, tmp_path: Path) -> None:
        manager = TimingCacheManager(tmp_path / "nonexistent.bin")
        manager.invalidate()  # should not raise


class TestTimingCacheManagerLoadIntoConfig:
    def test_raises_when_trt_unavailable(self, tmp_path: Path) -> None:
        manager = TimingCacheManager(tmp_path / "cache.bin")
        with patch("app_v2.infrastructure.timing_cache_manager.trt", None):
            with pytest.raises(RuntimeError, match="TensorRT is not available"):
                manager.load_into_config(MagicMock())

    def test_creates_empty_cache_when_no_file(self, tmp_path: Path) -> None:
        manager = TimingCacheManager(tmp_path / "cache.bin")
        mock_trt = MagicMock()
        mock_config = MagicMock()
        mock_cache = MagicMock()
        mock_config.create_timing_cache.return_value = mock_cache

        with patch("app_v2.infrastructure.timing_cache_manager.trt", mock_trt):
            result = manager.load_into_config(mock_config)

        mock_config.create_timing_cache.assert_called_once_with(b"")
        mock_config.set_timing_cache.assert_called_once_with(mock_cache, ignore_mismatch=False)
        assert result is mock_cache

    def test_loads_existing_cache_bytes(self, tmp_path: Path) -> None:
        cache_data = b"fake_trt_timing_cache_blob"
        p = tmp_path / "cache.bin"
        p.write_bytes(cache_data)
        manager = TimingCacheManager(p)
        mock_trt = MagicMock()
        mock_config = MagicMock()

        with patch("app_v2.infrastructure.timing_cache_manager.trt", mock_trt):
            manager.load_into_config(mock_config)

        mock_config.create_timing_cache.assert_called_once_with(cache_data)


class TestTimingCacheManagerSaveFromConfig:
    def test_raises_when_trt_unavailable(self, tmp_path: Path) -> None:
        manager = TimingCacheManager(tmp_path / "cache.bin")
        with patch("app_v2.infrastructure.timing_cache_manager.trt", None):
            with pytest.raises(RuntimeError, match="TensorRT is not available"):
                manager.save_from_config(MagicMock())

    def test_writes_serialized_blob_to_disk(self, tmp_path: Path) -> None:
        expected_blob = b"serialized_timing_data"
        p = tmp_path / "cache.bin"
        manager = TimingCacheManager(p)

        mock_trt = MagicMock()
        mock_config = MagicMock()
        mock_cache = MagicMock()
        mock_cache.serialize.return_value = expected_blob
        mock_config.get_timing_cache.return_value = mock_cache

        with patch("app_v2.infrastructure.timing_cache_manager.trt", mock_trt):
            bytes_written = manager.save_from_config(mock_config)

        assert p.read_bytes() == expected_blob
        assert bytes_written == len(expected_blob)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        p = tmp_path / "deep" / "nested" / "cache.bin"
        manager = TimingCacheManager(p)

        mock_trt = MagicMock()
        mock_config = MagicMock()
        mock_cache = MagicMock()
        mock_cache.serialize.return_value = b"data"
        mock_config.get_timing_cache.return_value = mock_cache

        with patch("app_v2.infrastructure.timing_cache_manager.trt", mock_trt):
            manager.save_from_config(mock_config)

        assert p.exists()

    def test_roundtrip_load_save(self, tmp_path: Path) -> None:
        """Save then load should return the same blob."""
        blob = b"roundtrip_test_blob_xyz"
        p = tmp_path / "cache.bin"
        manager = TimingCacheManager(p)

        # --- save ---
        mock_trt = MagicMock()
        save_config = MagicMock()
        save_cache = MagicMock()
        save_cache.serialize.return_value = blob
        save_config.get_timing_cache.return_value = save_cache

        with patch("app_v2.infrastructure.timing_cache_manager.trt", mock_trt):
            manager.save_from_config(save_config)

        # --- load ---
        load_config = MagicMock()
        with patch("app_v2.infrastructure.timing_cache_manager.trt", mock_trt):
            manager.load_into_config(load_config)

        load_config.create_timing_cache.assert_called_once_with(blob)

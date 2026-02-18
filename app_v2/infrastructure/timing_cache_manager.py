from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tensorrt as trt
except Exception:  # pragma: no cover
    trt = None  # type: ignore[assignment]


class TimingCacheManager:
    """Persists TensorRT timing cache to disk to enable deterministic, faster rebuilds.

    Usage during engine build:
        manager = TimingCacheManager("models/tensorrt/timing_cache.bin")
        cache = manager.load_into_config(builder_config)
        # ... build engine ...
        manager.save_from_config(builder_config)
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)

    @property
    def path(self) -> Path:
        return self._path

    def load_into_config(self, builder_config: Any) -> Any:
        """Load existing cache from disk (or create empty) and attach to builder_config.

        Returns the ITimingCache object so the caller can inspect it if needed.
        Raises RuntimeError if TensorRT is unavailable.
        """
        if trt is None:
            raise RuntimeError("TensorRT is not available")

        blob = self._read_blob()
        cache = builder_config.create_timing_cache(blob)
        builder_config.set_timing_cache(cache, ignore_mismatch=False)
        source = "disk" if blob else "empty"
        print(f"[TimingCacheManager] Loaded timing cache from {source} ({len(blob)} bytes)")
        return cache

    def save_from_config(self, builder_config: Any) -> int:
        """Serialize the timing cache embedded in builder_config back to disk.

        Returns the number of bytes written.
        """
        if trt is None:
            raise RuntimeError("TensorRT is not available")

        cache = builder_config.get_timing_cache()
        blob = cache.serialize()
        raw: bytes = bytes(blob)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(raw)
        print(f"[TimingCacheManager] Saved timing cache to {self._path} ({len(raw)} bytes)")
        return len(raw)

    def exists(self) -> bool:
        """Return True when a persisted cache file is present on disk."""
        return self._path.exists() and self._path.stat().st_size > 0

    def invalidate(self) -> None:
        """Delete the on-disk cache (force full re-profiling on next build)."""
        if self._path.exists():
            self._path.unlink()
            print(f"[TimingCacheManager] Invalidated cache at {self._path}")

    def _read_blob(self) -> bytes:
        if self._path.exists():
            return self._path.read_bytes()
        return b""

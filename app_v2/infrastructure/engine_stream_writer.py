"""TensorRT IStreamWriter implementations (Opt #6).

TensorRT 10+ provides ``builder.build_serialized_network_to_stream(network, config, writer)``
as an alternative to ``build_serialized_network()``.  The streaming API avoids
holding the **full** serialised engine bytes in RAM before writing to disk —
each chunk is forwarded to the writer immediately, which lowers peak memory for
large engines (FP8-qdq, INT8, etc.).

Two concrete implementations are provided:

``FileStreamWriter``
    Streams bytes directly to a file.  Peak heap usage = chunk size, not engine
    size.  Best for large models where the full bytes object would cause GC
    pressure or OOM.

``BytesStreamWriter``
    Collects all chunks into a ``bytearray``.  Semantically equivalent to the
    original ``build_serialized_network()`` path but goes through the streaming
    interface; useful when the caller needs the engine bytes in memory (e.g. for
    an immediate ``deserialize_cuda_engine()`` call without touching disk).

Usage::

    from app_v2.infrastructure.engine_stream_writer import FileStreamWriter

    writer = FileStreamWriter(engine_path)
    ok = builder.build_serialized_network_to_stream(network, config, writer)
    if not ok:
        raise RuntimeError("build_serialized_network_to_stream failed")
    print(f"Written {writer.bytes_written} bytes to {engine_path}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import tensorrt as trt  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)


class FileStreamWriter(trt.IStreamWriter):
    """Stream engine bytes directly to a file — no full-engine in-memory copy.

    Each chunk delivered by TRT is ``write()`` appended to the file handle
    immediately, so peak RSS stays proportional to the chunk size rather than
    the full engine size.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        trt.IStreamWriter.__init__(self)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("wb")
        self._bytes_written = 0

    # ------------------------------------------------------------------
    # trt.IStreamWriter interface
    # ------------------------------------------------------------------

    def write(self, data: bytes) -> int:  # type: ignore[override]
        n = self._handle.write(data)
        self._bytes_written += n
        return n

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()
            _log.debug(
                "[IStreamWriter] Wrote %d bytes to '%s'",
                self._bytes_written,
                self._path,
            )

    def __enter__(self) -> "FileStreamWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def bytes_written(self) -> int:
        return self._bytes_written


class BytesStreamWriter(trt.IStreamWriter):
    """Collect serialised engine bytes in memory via the streaming interface.

    Equivalent to ``build_serialized_network()`` but exercises the same
    ``IStreamWriter`` code-path.  Primarily useful for tests and for cases
    where the engine must be immediately deserialised without a disk round-trip.
    """

    def __init__(self) -> None:
        trt.IStreamWriter.__init__(self)
        self._buf = bytearray()

    def write(self, data: bytes) -> int:  # type: ignore[override]
        self._buf.extend(data)
        return len(data)

    @property
    def bytes(self) -> bytes:
        return bytes(self._buf)

    @property
    def bytes_written(self) -> int:
        return len(self._buf)

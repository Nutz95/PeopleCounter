"""FileStreamWriter — stream serialised TRT engine bytes directly to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import tensorrt as trt  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)


class FileStreamWriter(trt.IStreamWriter):
    """Stream engine bytes directly to a file — no full-engine in-memory copy.

    Each chunk delivered by TRT is ``write()``-appended to the file handle
    immediately, so peak RSS stays proportional to the chunk size rather than
    the full engine size.

    Supports the context-manager protocol::

        with FileStreamWriter(engine_path) as writer:
            ok = builder.build_serialized_network_to_stream(network, config, writer)
        # file is flushed and closed on __exit__
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
                "[FileStreamWriter] Wrote %d bytes to '%s'",
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

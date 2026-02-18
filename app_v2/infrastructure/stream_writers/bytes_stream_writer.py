"""BytesStreamWriter — collect serialised TRT engine bytes in memory."""

from __future__ import annotations

import tensorrt as trt  # type: ignore[import-untyped]


class BytesStreamWriter(trt.IStreamWriter):
    """Collect serialised engine bytes in memory via the streaming interface.

    Equivalent to ``build_serialized_network()`` but exercises the same
    ``IStreamWriter`` code-path.  Primarily useful for tests and for cases
    where the engine must be immediately deserialised without a disk round-trip.

    Example::

        writer = BytesStreamWriter()
        ok = builder.build_serialized_network_to_stream(network, config, writer)
        engine = runtime.deserialize_cuda_engine(writer.bytes)
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

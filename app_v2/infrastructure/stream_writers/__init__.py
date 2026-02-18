"""stream_writers — TensorRT IStreamWriter implementations (Opt #6).

TensorRT 10+ provides ``builder.build_serialized_network_to_stream(network, config, writer)``
as an alternative to ``build_serialized_network()``.  The streaming API avoids
holding the full serialised engine bytes in RAM before writing to disk —
each chunk is forwarded to the writer immediately, lowering peak memory for
large engines.

Available implementations:

- :class:`FileStreamWriter` — streams bytes directly to a file
- :class:`BytesStreamWriter` — collects bytes in memory (useful for tests or in-memory deserialisation)
"""

from app_v2.infrastructure.stream_writers.bytes_stream_writer import BytesStreamWriter
from app_v2.infrastructure.stream_writers.file_stream_writer import FileStreamWriter

__all__ = ["FileStreamWriter", "BytesStreamWriter"]

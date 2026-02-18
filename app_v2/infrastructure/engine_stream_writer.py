"""Backward-compatibility shim — import from stream_writers sub-package instead.

.. deprecated::
    Import directly from :mod:`app_v2.infrastructure.stream_writers`.
    This module is kept to avoid breaking existing callers during migration.

Example migration::

    # old
    from app_v2.infrastructure.engine_stream_writer import FileStreamWriter
    # new
    from app_v2.infrastructure.stream_writers import FileStreamWriter
"""
from app_v2.infrastructure.stream_writers import BytesStreamWriter, FileStreamWriter

__all__ = ["FileStreamWriter", "BytesStreamWriter"]

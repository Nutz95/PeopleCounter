from enum import Enum


class FusionStrategyType(Enum):
    """Enumerates supported fusion strategies for result publication."""

    STRICT_SYNC = "STRICT_SYNC"
    ASYNC_OVERLAY = "ASYNC_OVERLAY"
    RAW_STREAM_WITH_METADATA = "RAW_STREAM_WITH_METADATA"

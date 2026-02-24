"""RawStreamFusionStrategy — publish each model result immediately as it arrives."""
from __future__ import annotations

from typing import Any, Sequence

from app_v2.enums import FusionStrategyType
from app_v2.core.strategies.base import FusionStrategy


class RawStreamFusionStrategy(FusionStrategy):
    """Video stream is fully decoupled from inference.

    The NVDEC ring-buffer slot is released *before* inference begins (see
    PipelineOrchestrator.run), so the decoder keeps feeding frames at full
    camera FPS regardless of inference latency.  Inference results are shipped
    as lightweight SSE metadata events the moment they are ready — the browser
    layers them over the already-decoded video.
    """

    def __init__(self) -> None:
        super().__init__(FusionStrategyType.RAW_STREAM_WITH_METADATA)

    def should_publish(self, frame_id: int, collected_ids: Sequence[int]) -> bool:
        # One model result is enough to trigger a publish.
        return len(collected_ids) >= 1

    def merge(self, payloads: Sequence[Any]) -> Sequence[Any]:
        return list(payloads)

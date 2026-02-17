from __future__ import annotations

PREPROCESS_MODEL_STAGE_PREFIX = "preprocess_model_"
PREPROCESS_STREAM_MODEL_PREFIX = "preprocess_stream_model_"
PREPROCESS_STREAM_CUDA_MODEL_PREFIX = "preprocess_stream_cuda_model_"

PREPROCESS_MODEL_SUM_MS = "preprocess_model_sum_ms"
PREPROCESS_MODEL_MAX_MS = "preprocess_model_max_ms"
PREPROCESS_CRITICAL_PATH_MS = "preprocess_critical_path_ms"
PREPROCESS_SERIAL_OVERHEAD_MS = "preprocess_serial_overhead_ms"
PREPROCESS_PARALLEL_EFFICIENCY = "preprocess_parallel_efficiency"

FUSION_WAIT_MS = "fusion_wait_ms"
OVERLAY_LAG_MS = "overlay_lag_ms"
END_TO_END_MS = "end_to_end_ms"
FRAME_TIMESTAMP_NS = "frame_timestamp_ns"
INFERENCE_MODEL_STAGE_PREFIX = "inference_model_"
INFERENCE_MODEL_SUM_MS = "inference_model_sum_ms"
INFERENCE_MODEL_MAX_MS = "inference_model_max_ms"


def preprocess_model_stage_name(model_name: str) -> str:
    return f"{PREPROCESS_MODEL_STAGE_PREFIX}{model_name}"


def preprocess_model_metric_key(model_name: str) -> str:
    return f"{PREPROCESS_MODEL_STAGE_PREFIX}{model_name}_ms"


def preprocess_stream_model_key(model_name: str) -> str:
    return f"{PREPROCESS_STREAM_MODEL_PREFIX}{model_name}"


def preprocess_stream_cuda_model_key(model_name: str) -> str:
    return f"{PREPROCESS_STREAM_CUDA_MODEL_PREFIX}{model_name}"


def inference_model_metric_key(model_name: str) -> str:
    return f"{INFERENCE_MODEL_STAGE_PREFIX}{model_name}_ms"

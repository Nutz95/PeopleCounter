# GPU Preprocess Components

This document describes the v2 preprocessing chain added for model-input planning and compatibility with existing inference APIs.

## Components

- `InputSpecRegistry` (`app_v2/application/input_spec_registry.py`)
  - Builds input specs from pipeline-like config.
  - Supports at least:
    - `yolo_global`: one global letterbox input (`640x640` by default)
    - `yolo_tiles`: tiled inputs (`640x640`) with overlap (default `0.2`)

- `GpuPreprocessPlanner` (`app_v2/application/gpu_preprocess_planner.py`)
  - Converts frame size + `InputSpec` into concrete `PreprocessTask` entries.
  - Produces per-model `PreprocessPlan` metadata (`mode`, tile rows/cols, overlap, task count).

- `GpuPreprocessor` (`app_v2/infrastructure/gpu_preprocessor.py`)
  - Implements `Preprocessor`.
  - `configure(metadata)` loads specs through the registry.
  - `build_output(frame_id, frame)` returns `PreprocessOutput` containing plans and model-scoped inputs.
  - `process(frame_id, frame)` returns flattened inputs for compatibility with `InferenceModel.infer(frame_id, inputs)`.

- `PreprocessOutput` (`app_v2/core/preprocess_types.py`)
  - Stores model plans and model inputs for one frame.
  - Exposes `flatten_inputs(model_name=None)` to produce inference-compatible sequences.

## Backward compatibility

`CudaPreprocessor` now wraps the new implementation and keeps the same public methods (`configure`, `process`) so `PipelineOrchestrator` can continue working unchanged.

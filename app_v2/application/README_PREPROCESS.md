# GPU Preprocess Components

This document describes the v2 preprocessing chain added for model-input planning and compatibility with existing inference APIs.

## Components

- `InputSpecRegistry` (`app_v2/application/input_spec_registry.py`)
  - Builds input specs from the `preprocess` block in `app_v2/config/pipeline.yaml`.
  - Requires each entry to define `target_width`, `target_height`, `mode`, and `overlap` so changes to model inputs happen through configuration instead of hard-coded defaults.

- `GpuPreprocessPlanner` (`app_v2/application/gpu_preprocess_planner.py`)
  - Converts frame size + `InputSpec` into concrete `PreprocessTask` entries.
  - Produces per-model `PreprocessPlan` metadata (`mode`, tile rows/cols, overlap, task count).

- `GpuPreprocessor` (`app_v2/infrastructure/gpu_preprocessor.py`)
  - Implements `Preprocessor`.
  - `configure(metadata)` loads specs through the registry.
  - Calls the CUDA preprocessing kernels defined under `app_v2/kernels/preprocess.py` so buffers stay on-device instead of duplicating entire `GpuFrame`s.
  - `build_output(frame_id, frame)` returns `PreprocessOutput` containing plans and model-scoped inputs.
  - `process(frame_id, frame)` returns flattened inputs for compatibility with `InferenceModel.infer(frame_id, inputs)`.

- `PreprocessOutput` (`app_v2/core/preprocessor_types/preprocess_output.py`)
  - Stores model plans and model inputs for one frame.
  - Exposes `flatten_inputs(model_name=None)` to produce inference-compatible sequences.
  - Now also carries a `FrameTelemetry` object so downstream consumers can show GPU stage timings without forcing CPU synchronization.

## Backward compatibility

`CudaPreprocessor` now wraps the new implementation and keeps the same public methods (`configure`, `process`) so `PipelineOrchestrator` can continue working unchanged.

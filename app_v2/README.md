# PeopleCounter v2

PeopleCounter v2 is a GPU-first rewrite that targets TensorRT-only inference with a clean, modular architecture. It brings the legacy processing into a disciplined frame_id-based pipeline so each model can publish results independently while a dedicated fusion strategy merges them for the Flask streamer.

## Key highlights

- **TensorRT-only inference**: YOLO segmentation, YOLO tiling, and density models all execute inside TensorRT execution contexts.
- **GPU preprocessing**: The `CudaPreprocessor` lives in the infrastructure layer so every data movement is consolidated on CUDA streams defined in `config/pipeline.yaml`.
- **Fusion strategies**: The architecture supports `STRICT_SYNC`, `ASYNC_OVERLAY`, and `RAW_STREAM_WITH_METADATA` fusion modes driven by the pipeline config.
- **Modular layers**: Core interfaces (`FrameSource`, `InferenceModel`, `Postprocessor`, etc.) keep the application layer decoupled from infrastructure details.

## Configuration

- `config/pipeline.yaml` defines which models are enabled, the fusion strategy, tiling metadata, and the CUDA stream indices consumed by `CudaStreamPool`.
- `config/log.yaml` controls the filtered logger channels (`GLOBAL`, `YOLO`, `DENSITY`) so you can mute noisy lanes without touching the code.
- `plans/app_v2_migration_plan.md` tracks the remaining scaffolding and documentation updates so the rewrite stays in sync.

## Getting started

1. Run `./1_prepare.sh` at the repo root to build and prepopulate the Docker image (`people-counter:gpu-final`).
2. Run `./3_run_app.sh --app-version v2 rtsp://<camera-url>` to launch the v2 orchestrator inside the prepared image.
3. The Flask server exposes the same web UI at http://localhost:5000 once inference logic is implemented.

The skeleton in this folder includes config loaders, interfaces, infrastructure stubs, and orchestration scaffolding; replace the stub methods in each class with real TensorRT execution code to finish the port.

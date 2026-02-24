# PeopleCounter v2

PeopleCounter v2 is a GPU-first rewrite that targets TensorRT-only inference with a clean, modular architecture. It brings the legacy processing into a disciplined frame_id-based pipeline so each model can publish results independently while a dedicated fusion strategy merges them for the Flask streamer.

## Key highlights

- **TensorRT-only inference**: YOLO segmentation, YOLO tiling, and density models all execute inside TensorRT execution contexts.
- **GPU preprocessing**: The `CudaPreprocessor` lives in the infrastructure layer so every data movement is consolidated on CUDA streams defined in `config/pipeline.yaml`.
- **Fusion strategies**: The architecture supports `STRICT_SYNC`, `ASYNC_OVERLAY`, and `RAW_STREAM_WITH_METADATA` fusion modes driven by the pipeline config.
- **Modular layers**: Core interfaces (`FrameSource`, `InferenceModel`, `Postprocessor`, etc.) keep the application layer decoupled from infrastructure details.

## Configuration

- `config/pipeline.yaml` defines which models are enabled, the fusion strategy, tiling metadata, and the CUDA stream indices consumed by `CudaStreamPool`.
- `config/log.yaml` controls the filtered logger channels (`GLOBAL`, `YOLO`, `DENSITY`, `NVDEC`). Toggle `NVDEC_DEBUG_LOGS` (or flip the `nvdec` flag in the same YAML) when you want to see decode diagnostics without overflowing the final console.
- `plans/app_v2_migration_plan.md` tracks the remaining scaffolding and documentation updates so the rewrite stays in sync.

## Documentation

- Architecture overview and progress diagrams: `app_v2/docs/README_ARCHI.md`
- Detailed preprocess pipeline and metrics: `app_v2/docs/README_PREPROCESS.md`
- Performance analysis (FPS, delay, overlay lag): `app_v2/docs/README_PERFORMANCE_ANALYSIS.md`

## Getting started

1. Run `./1_prepare.sh` at the repo root to build and prepopulate the Docker image (`people-counter:gpu-final`).
2. Run `./2_prepare_nvdec.sh` to build the NVDEC-enabled image layer with VPF/PyNvCodec; this step is mandatory so the orchestrator can decode directly on the GPU and the tests execute inside `people-counter:gpu-final-nvdec`.

	The installer now prefers a repo-local copy under `external/Video_Codec_SDK_13.0.37` and copies that folder's `libnvcuvid.so` and headers into `/usr/local/cuda`. When that directory is absent, it downloads the archive at https://developer.nvidia.com/downloads/video-codec-sdk/13.0.37/video_codec_sdk_13.0.37.zip (you will normally need to authenticate with a NVIDIA account), extracts it into `externals/`, and stages the codec files before building PyNvCodec. If you already have the SDK archive or want to use a different version, set `VIDEO_CODEC_SDK_DIR=/path/to/Video_Codec_SDK` and optionally override `VIDEO_CODEC_SDK_DOWNLOAD_URL`/`VIDEO_CODEC_SDK_VERSION` before running this prep step.
3. Run `./4_run_app.sh --app-version v2 rtsp://<camera-url>` to launch the v2 orchestrator inside the prepared image. (exemple:`./4_run_app.sh 'http://192.168.x.x:5002/video_feed' --app-version v2`)
4. Run `./5_run_tests.sh` (after each implementation pass) so the orchestrator is compiled and the current pytest suite executes inside the NVDEC-ready container, ensuring the GPU loop stays validated.
5. The Flask server exposes the same web UI at http://localhost:5000 once inference logic is implemented.

 The skeleton in this folder now wires NVDEC decode, CUDA preprocessing, and TensorRT inference into a single `PipelineOrchestrator`. It also includes a GPU-first test harness (`app_v2/tests` + `./5_run_tests.sh`) so every change to the scheduler, fusion strategy, or publisher runs inside the same NVDEC image used for deployment.

### Debugging & testing

- Set `NVDEC_TEST_STREAM_URL` (or edit `app_v2/config/test_config.yaml`) to your Windows bridge RTSP endpoint when you want to validate the real decoder; `pytest app_v2/tests/integration/nvdec/test_nvdec_decode.py` will consume that stream and assert on the GPU-resident NV12 surfaces.
- Use `plans/nvdec_performance_baseline.md` as the capture sheet for your first timing run so the early `PerformanceTracker` metrics and `nvidia-smi` snapshots can serve as a reference when TensorRT strips in the true prediction payloads.
- Use `app_v2/tests/integration/pipeline/test_pipeline_metrics_integration.py` as the evolving integration baseline for end-to-end stage timings and tensor-pool metrics.

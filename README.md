# PeopleCounter

PeopleCounter counts people on a live camera stream (up to 4K) by combining YOLO and LWCC/density models with sparse mask overlays and latency-aware metrics. The legacy pipeline is now housed under `app_v1/`, while `app_v2/` contains the new TensorRT-only, GPU-first orchestrator described in the `app_v2/README.md` family of docs.

## Quick start

### 1. Clone & install Python dependencies

- Clone the repository on your host (Windows or Linux) and navigate inside the source tree.
- Create and activate a Python 3.11 virtual environment:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Linux/WSL/macOS
```
- Install the runtime requirements:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Prepare the AI models

- Run `./setup.sh` (WSL/Git Bash recommended). It downloads Ultralytics weights and converts them to ONNX/TensorRT `.engine` files inside `models/`.
- CUDA/TensorRT is required to generate `.engine` files locally; if drivers are missing, the script still produces ONNX copies.

### 3. Set up the Windows → WSL bridge

- Clone the repository on your Windows host as well.
- Run the helper in `windows/setup_and_run.bat`; it
  1. creates `venv_bridge`,
  2. installs Flask/OpenCV, and
  3. launches `camera_bridge.py` that exposes the MJPEG stream (e.g. `http://192.168.1.62:5002/video_feed`).
- Once the bridge is running, open WSL, `cd` into the repo, and launch:

```bash
./run_app.sh --profile rtx_extreme http://<windows-ip>:5002/video_feed
```

This pulls the correct Docker image, generates any missing models inside the container, and connects to the bridge stream.

### 4. Run PeopleCounter

We have split the workflow into six stages:

1. `./0_build_image.sh` performs the heavy Docker image build (`people-counter:gpu-final`), so you only spend that hour once.
2. `./1_prepare.sh` layers the apt/pip dependencies on top of the already built image.
3. `./2_prepare_nvdec.sh` builds and installs VPF + PyNvCodec from source in a **new image layer**, committing `people-counter:gpu-final-nvdec`. NVDEC support is now required to run the `app_v2` orchestrator and the GPU test harness. The NVDEC layer also patches the FFmpeg demuxer so codecs that aren't H.264/HEVC skip the Annex-B bitstream filter, preventing the "unknown filter by name" failure that plagued the MJPEG bridge stream.

  The installer first checks whether you placed `external/Video_Codec_SDK_13.0.37` inside the repository; if so, that folder is used. When it is missing, the script downloads the official archive from https://developer.nvidia.com/downloads/video-codec-sdk/13.0.37/video_codec_sdk_13.0.37.zip (a NVIDIA account sign-in is typically required), extracts it into `externals/`, and copies `libnvcuvid.so` plus the headers into `/usr/local/cuda` so PyNvCodec can build. If you already provisioned the SDK yourself (e.g., offline or due to corporate mirrors), set `VIDEO_CODEC_SDK_DIR=/path/to/Video_Codec_SDK` before running `./2_prepare_nvdec.sh` so the script will use that copy instead of downloading another archive.
4. `./3_prepare_models.sh` runs `prepare_models.py` inside the prepared image so you can rebuild TensorRT/ONNX assets without re-running the installs.
5. `./4_run_app.sh --app-version v1 <source>` launches the legacy pipeline from `app_v1/` with the existing worker graph. Use `--app-version v2` to start the new GPU-first orchestration in `app_v2/`, which now decodes via NVDEC, preprocesses on CUDA, executes the TensorRT engines, fuses metadata, and publishes through the Flask server.
6. `./5_run_tests.sh` compiles `app_v2` and runs `pytest app_v2/tests` inside the NVDEC-ready `people-counter:gpu-final-nvdec` image so each implementation pass runs on the same GPU stack that ships to production.

The `app_v2` orchestrator now drives NVDEC decode, CUDA preprocessing, TensorRT inference, and metadata fusion before the Flask streamer publishes the merged masks; see `app_v2/README.md` for the clean-architecture overview.

Use `IMAGE_NAME=people-counter:gpu-final-nvdec` with `./4_run_app.sh` or `./5_run_tests.sh` when running on the optional NVDEC-enabled image.

`run_app.sh` is kept as a lightweight wrapper for backwards compatibility; it now forwards all arguments to `./4_run_app.sh`.

Pass any additional camera URL or resolution arguments after the profile (e.g., `./4_run_app.sh --app-version v2 --profile rtx_extreme http://...`).
The recommended way to change backends remains modifying `.env` profiles as detailed below.

### 5. Build the Docker image (optional)

- Regenerate the Docker runtime with `./build_image.sh` from inside WSL.
- Prerequisites: install Docker Desktop or Engine inside WSL, install the NVIDIA Container Toolkit (`nvidia-container-toolkit`), and ensure `nvidia-smi` works in WSL (requires NVIDIA drivers that expose CUDA inside the subsystem).
- Verifying GPUs inside the container can be done with:

```bash
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

## Configuration via `.env`

- Each profile lives under `scripts/configs/<profile>.env` (examples: `rtx_extreme.env`, `cpu_fallback.env`, `balanced_tri_chip.env`). `run_app.sh --profile rtx_extreme` sources the file before launching the pipeline.
- Common variables you can override:
  - `YOLO_BACKEND`, `YOLO_MODEL`, `YOLO_DEVICE`
  - `LWCC_BACKEND`, `OPENVINO_DEVICE`, `LWCC_THRESHOLD`
  - `YOLO_TILING`, `DENSITY_TILING`, `DENSITY_THRESHOLD`
  - `YOLO_USE_GPU_PREPROC`, `YOLO_USE_GPU_POST`, `YOLO_PIPELINE_MODE`
  - `YOLO_SEG` to toggle segmentation models
  - `DEBUG_TILING` to log tiles and `YOLO_CONF` to adjust thresholds
- The `.env` files can also include overrides for `EXTREME_DEBUG`, `CAMERA_URL`, and the MQTT broker when running in distributed mode. Leave `EXTREME_DEBUG` unset to keep the console focused on the `[MASK TIMING]` timeline, or set it to `1` to surface the `[GPU PERF]`/average logs.
- When you need granular guidance on each flag, consult [README_PARAMETERS.md](README_PARAMETERS.md) for a parameter-by-parameter breakdown.

## Observability & metrics

- The web UI surface now exposes a “mask timings” card (created/sent/received/displayed times) plus a latency history graph that paints the 25–30 fps target band. The chart and console use the same `[MASK TIMING]` payload stream so regressions appear in both places.
- A `[MASK TIMING]` log line records backend creation and send latencies, breaking the delay into creation, send, and total segments so you can trace any regressions in the pipeline.
- The metrics payload now carries `density_heatmap_payload` alongside the YOLO mask metadata so the browser can draw density overlays without rewiring the server.
- Masks are downscaled and aligned to the client canvas before compositing, so overlays only tint the detected zones rather than the entire feed. More architecture detail is in the dedicated doc below.
- After the NVDEC decoder delivers GPU frames you can capture the first baseline run with `plans/nvdec_performance_baseline.md` and, optionally, set `NVDEC_TEST_STREAM_URL` so `pytest app_v2/tests/test_nvdec_decode.py` verifies the real stream before TensorRT starts producing predictions.

## Important model notes

- TensorRT engines are built with CUDA 13.1 + TensorRT 10.14 on a RTX 5060 Ti. A working CUDA/TensorRT stack is required to reproduce `.engine` files locally.
- PyTorch exports target Opset 18 so that Ultralytics stays compatible with TensorRT conversion and the automatic converter doesn’t downgrade the model.
- OpenVINO IR artifacts are stored under `models/openvino/`; the conversion pipeline has moved to `convert_pth_to_openvino.py` in case you want to re-export from different backends.

## See also

- [README_DOCKER.md](README_DOCKER.md) : Docker build, runtime, and profiling documentation (includes the latest architecture diagrams for the containerized pipeline).
- [app_v2/README.md](app_v2/README.md) : Clean architecture overview for the v2 orchestrator.
- [README_ARCHITECTURE.md](README_ARCHITECTURE.md) : In-depth architecture, masking/metrics/tiling flows, mermaid diagrams, and `.env` reference for the legacy stack.
- [plans/app_v2_migration_plan.md](plans/app_v2_migration_plan.md) : Tracks the remaining v2 scaffolding, configs, and docs updates so the clean rewrite stays on schedule.
- [windows/setup_and_run.bat](windows/setup_and_run.bat) : Windows helper to spin up the camera bridge that exports the MJPEG stream consumed by WSL.

The main README remains the entry point: it should invite contributors to try the project, link to the Docker guide for profiles, the architecture doc for the deep dive, and reference the plans that must be kept synchronized.

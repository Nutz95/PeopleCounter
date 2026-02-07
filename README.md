# PeopleCounter

PeopleCounter counts people on a live camera stream (up to 4K) by combining YOLO and LWCC/density models with sparse mask overlays and latency-aware metrics.

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

- `run_app.sh` wraps the entire pipeline and sources `scripts/configs/<profile>.env` before starting the app. Pass any additional camera URL or resolution arguments after the profile.
- The script accepts the same signature as the old `run_people_counter_*` helpers (`[RESOLUTION] [MODEL] ...`), but the recommended way to change backends is via `.env` profiles (see below).

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
- The `.env` files can also include overrides for `EXTREME_DEBUG`, `CAMERA_URL`, and the MQTT broker when running in distributed mode.

## Observability & métriques

- The web UI surface now exposes a “mask timings” card (created/sent/received/displayed times) plus the “YOLO internal (ms)” chart so you can correlate backend FPS to frontend update cadence. It relies on `camera_app_pipeline.py` propagating `created_at`/`created_at_ts` metadata from `yolo_seg_people_counter.py` and on the adaptive polling in `static/js/app.js`.
- A `[MASK TIMING]` log line records backend creation and send latencies, which explains why overlays refresh around 1 Hz even if YOLO emits more frames; the UI divides those timings into creation, send, and total latency segments.
- Masks are downscaled and aligned to the client canvas before compositing, so overlays only tint the detected zones rather than the entire feed. More architecture detail is in the dedicated doc below.

## Important model notes

- TensorRT engines are built with CUDA 13.1 + TensorRT 10.14 on a RTX 5060 Ti. A working CUDA/TensorRT stack is required to reproduce `.engine` files locally.
- PyTorch exports target Opset 18 so that Ultralytics stays compatible with TensorRT conversion and the automatic converter doesn’t downgrade the model.
- OpenVINO IR artifacts are stored under `models/openvino/`; the conversion pipeline has moved to `convert_pth_to_openvino.py` in case you want to re-export from different backends.

## Voir aussi

- [README_DOCKER.md](README_DOCKER.md) : Docker build, runtime, and profiling documentation (includes the latest architecture diagrams for the containerized pipeline).
- [README_ARCHITECTURE.md](README_ARCHITECTURE.md) : In-depth architecture, masking/metrics/tiling flows, mermaid diagrams, density model layout, and `.env` reference.
- [plans/documentation-refresh-plan.md](plans/documentation-refresh-plan.md), [plans/mask_overlay_roadmap.md](plans/mask_overlay_roadmap.md), [plans/mask_timing-plan.md](plans/mask_timing-plan.md) : Track the doc refresh, mask overlay fixes, and timing instrumentation work.
- [windows/setup_and_run.bat](windows/setup_and_run.bat) : Windows helper to spin up the camera bridge that exports the MJPEG stream consumed by WSL.

Le README principal reste le point d’entrée : il doit donner envie d’essayer le projet, pointer vers la doc Docker pour les profils, vers l’architecture pour les détails techniques, et renvoyer vers les plans qui doivent toujours être mis à jour en parallèle.

## Support

Ouvrez une issue si vous rencontrez des problèmes, en précisant OS, GPU, driver, et commande utilisée.
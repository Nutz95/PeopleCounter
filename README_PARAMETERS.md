# Parameter Guide

This reference explains the purpose of each environment variable or `.env` flag that configures the PeopleCounter pipeline. Use the profiles located under `scripts/configs/<profile>.env` as examples and adjust one variable at a time when iterating.

## YOLO / segmentation

| Variable | Description | Effect |
| --- | --- | --- |
| `YOLO_BACKEND` | Selects the inference backend for YOLO (TensorRT, OpenVINO, PyTorch) | Controls which engine (`tensorrt_native`, `openvino_native`, `torch`, etc.) powers the detection stage.
| `YOLO_MODEL` | Path to the segmentation model (`.engine`, `.onnx`, `.pt`) | Determines accuracy/speed; choose `yolo26s-seg.engine` for RTX, `yolo26s-seg.onnx` for CPU fallback.
| `YOLO_DEVICE` | Explicit device to drive YOLO | Use `cuda`, `GPU`, `NPU`, or `CPU` to force a specific chip when multiple are available.
| `YOLO_USE_GPU_PREPROC` | Enables GPU-based preprocessing/kernels (`0` or `1`) | Offloads resize/crop into CUDA, reducing CPU copies when set to `1`.
| `YOLO_USE_GPU_POST` | Enables GPU rendering of fused masks | Keeps mask drawing on CUDA, bypassing CPU blend operations.
| `YOLO_PIPELINE_MODE` | High-level mode selector (`auto`, `gpu`, `gpu_full`, `cpu`) | Controls whether tiling, inference, and mask rendering stay on the GPU or fall back to CPU.
| `YOLO_CONF` | Default confidence threshold | Fine-tune detection sensitivity; lower values yield more boxes at the cost of false positives.
| `YOLO_SEG` | Enables segmentation-specific mask outputs (`1`/`0`) | Required when loading `-seg` models; keeps classic detection pipelines disabled when `0`.

## Density / LWCC

| Variable | Description | Effect |
| --- | --- | --- |
| `LWCC_BACKEND` | Kernel for the density model (`tensorrt`, `openvino`, `torch`) | Matches the available hardware for crowd-density estimates.
| `OPENVINO_DEVICE` | Device used by the OpenVINO backend (`GPU`, `NPU`, `CPU`) | Aligns the density pipeline with the desired silicon (use `GPU` or `NPU` when available).
| `DENSITY_TILING` | Enables tiled density processing (`1`/`0`) | Tiling breaks the frame into a grid (2x2) for higher precision with per-tile passes.
| `DENSITY_THRESHOLD` | Minimum crowd value per tile before it is considered occupied | Adjusts responsiveness of the density overlay; lower thresholds show smaller crowds.
| `LWCC_THRESHOLD` | Global density threshold passed to `PeopleCounterProcessor` | Controls the grouping of nearby detections when summarizing crowd counts.

## Tiling & capture

| Variable | Description | Effect |
| --- | --- | --- |
| `YOLO_TILING` | Enables YOLO tiling (`1`/`0`) | Splits 4K frames into `640×640` crops so tensor cores run at optimal batch sizes.
| `DEBUG_TILING` | Logs tile boundaries to stderr when truthy | Useful when analyzing leftover padding or overlapping tile edges.
| `CAMERA_URL` | Custom camera stream URL | Overrides the default `/dev/video0` with an IP/USB feed from Windows or RTSP devices.
| `CAMERA_FRAMERATE` | Soft limit for capture framerate | Caps the ingest rate before inference to avoid GPU overload.

## Infrastructure & logging

| Variable | Description | Effect |
| --- | --- | --- |
| `EXTREME_DEBUG` | Enables verbose logging (`1`) | Dumps pipeline timings, mask coverage, and backend decisions into the console.
| `MQTT_HOST`, `MQTT_PORT` | MQTT broker connection details | Used when broadcasting frames or receiving control events from other services.
| `SENTRY_DSN` | Optional Sentry configuration | Sends errors/metrics to Sentry when provided.
| `RUN_INFERENCE_ONLY` | Skips UI composition when truthy | Ideal for headless benchmarking/external dashboards.
| `FORCE_MASK_ENCODING` | Forces mask encoding path to run (`1`) | Useful during testing to isolate `_encode_mask_blob` changes.

Each profile in `scripts/configs` bundles a curated combination of these keys. When debugging performance, tweak only one variable at a time and restart `run_app.sh --profile <profile>` to isolate the impact.

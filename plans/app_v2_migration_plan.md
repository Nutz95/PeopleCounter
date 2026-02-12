# App v2 Migration Plan

## Technical context

- Python + Docker on CUDA 13.x (RTX 5060 Ti) with TensorRT-only inference
- Local 4K stream decoded via NVDEC so frames can stay on GPU, feeding YOLO segmentation (640×640), YOLO tiling (batch 640×640), and density tiles (~6×3 at 640×720)
- Objectives: 100% TensorRT + GPU preprocessing, multi-stream CUDA, clean SOLID layers, fusion strategy toggle (STRICT_SYNC, ASYNC_OVERLAY, RAW_STREAM_WITH_METADATA), zero PyTorch fallback, no GPU blending (masks handled by frontend)
- Configuration-driven via `app_v2/config/pipeline.yaml` (models/streams/fusion) and `config/log.yaml` (log channels)

## Status Checklist
- [x] Step 1: Split the build/prep/run workflow into `0_build_image.sh`, `1_prepare.sh`, `2_prepare_nvdec.sh`, `3_prepare_models.sh`, `4_run_app.sh`, and `5_run_tests.sh` so the image is prepared once and reused.
- [x] Step 2: Move the legacy application, logger, and shared configs into `app_v1/` while keeping `models/`, `logger/`, and `scripts/configs/` at the repo root. This migration is mandatory for the current dev cycle; we no longer keep aliases or compatibility wrappers for the old script names once they are renamed.
- [x] Step 3: Inventory current `app_v2/` scaffolding, verify it uses shared logging channels, and extend the config surface (`pipeline.yaml`, `log.yaml`).
- [x] Step 4: Implement the core/interface/infrastructure/application skeletons described in the clean‑architecture plan with clear responsibilities and method signatures, covering the NVDEC scheduler, YOLO overlap tracking, density tile batching, and the per-frame performance tracker.
- [x] Step 5: Build the GPU-first test harness by adding `app_v2/tests`, scripting `5_run_tests.sh`, and ensuring each run compiles `app_v2` before executing `pytest` inside `people-counter:gpu-final` so regressions surface early.
- [x] Step 6: Update documentation (`README.md`, `README_DOCKER.md`, `app_v2/README.md`, `app_v2/README_ARCHI.md`) with the new scripts, configs, and fusion diagrams (GPU/CPU annotations). Ensure every development step refreshes the docs.
- [x] Step 7: Keep the plan, diagrams, and reference docs synchronized so future contributors can track progress before implementing TensorRT specifics.

> Current focus: Observability of the NVDEC/TensorRT stack and keeping the docs/plans synchronized as we add TensorRT specifics (Step 6/7).
 - Update `export_density_to_onnx.py` so the generated engine expects 640×720 density tiles without overlap and matches the GPU batching logic.
 - Design how YOLO tiling overlap metadata is propagated to the fusion layer so the UI can assemble masks seamlessly.
 - Record the inaugural NVDEC/TensorRT baseline in `plans/nvdec_performance_baseline.md` so we can compare future edits.

## Processing Chain Migration Status (Color)

- 🟢 **Done**
	- NVDEC decoding path produces GPU-native `GpuFrame` through `NvdecDecoder` and `GpuRingBuffer`.
	- Preprocess component split introduced: `InputSpecRegistry`, `GpuPreprocessPlanner`, `GpuPreprocessor`, and `PreprocessOutput`.
	- Default preprocess specs supported for `yolo_global` (single 640x640 letterbox) and `yolo_tiles` (640x640 tiling, overlap default `0.2`).
	- Backward compatibility preserved through `CudaPreprocessor` wrapper so `PipelineOrchestrator` keeps its existing flow.

- 🟡 **Tested / In progress**
	- Unit coverage added for preprocess planning behavior (`yolo_global`, `yolo_tiles`).
	- Integration skeleton added for `NVDEC -> preprocess` with runtime skips when stream URL or `PyNvCodec` is unavailable.
	- Ring buffer long-run stress loop added to validate slot recycling over many producer/consumer cycles.

- 🔴 **Pending**
	- Replace placeholder preprocess payload replication with real GPU transform kernels (letterbox/tiling tensors).
	- Model-specific routing in orchestrator/preprocess handoff to avoid passing unrelated flattened inputs to each model.
	- End-to-end validation with real TensorRT model outputs and fusion impact under sustained throughput.


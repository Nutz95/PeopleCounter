# App v2 Migration Plan

## Technical context

- Python + Docker on CUDA 13.x (RTX 5060 Ti) with TensorRT-only inference
- Local 4K stream decoded via NVDEC so frames can stay on GPU, feeding YOLO segmentation (640×640), YOLO tiling (batch 640×640), and density tiles (~6×3 at 640×720)
- Objectives: 100% TensorRT + GPU preprocessing, multi-stream CUDA, clean SOLID layers, fusion strategy toggle (STRICT_SYNC, ASYNC_OVERLAY, RAW_STREAM_WITH_METADATA), zero PyTorch fallback, no GPU blending (masks handled by frontend)
- Configuration-driven via `app_v2/config/pipeline.yaml` (models/streams/fusion) and `config/log.yaml` (log channels)

## Status Checklist
- [x] Step 1: Split the build/prep/run workflow into `0_build_image.sh`, `1_prepare.sh`, `2_prepare_models.sh`, and `3_run_app.sh` so the image is prepared once and reused.
- [x] Step 2: Move the legacy application, logger, and shared configs into `app_v1/` while keeping `models/`, `logger/`, and `scripts/configs/` at the repo root.
- [x] Step 3: Inventory current `app_v2/` scaffolding, verify it uses shared logging channels, and extend the config surface (`pipeline.yaml`, `log.yaml`).
- [ ] Step 4: Implement the core/interface/infrastructure/application skeletons described in the clean‑architecture plan with clear responsibilities and method signatures, covering the NVDEC scheduler, YOLO overlap tracking, density tile batching, and the per-frame performance tracker.
- [ ] Step 5: Build the GPU-first test harness by adding `app_v2/tests`, scripting `4_run_tests.sh`, and ensuring each run compiles `app_v2` before executing `pytest` inside `people-counter:gpu-final` so regressions surface early.
- [ ] Step 6: Update documentation (`README.md`, `README_DOCKER.md`, `app_v2/README.md`, `app_v2/README_ARCHI.md`) with the new scripts, configs, and fusion diagrams (GPU/CPU annotations). Ensure every development step refreshes the docs.
- [ ] Step 7: Keep the plan, diagrams, and reference docs synchronized so future contributors can track progress before implementing TensorRT specifics.

## Supporting tasks
 - Update `export_density_to_onnx.py` so the generated engine expects 640×720 density tiles without overlap and matches the GPU batching logic.
 - Design how YOLO tiling overlap metadata is propagated to the fusion layer so the UI can assemble masks seamlessly.

> Each checkbox also triggers a doc update; the README(s) must stay aligned with the actual scripts/configs mentioned in each step.
>
> Current focus: Step 4 (core/infrastructure/application skeletons) and Step 5 (GPU build/test harness) while keeping docs synchronized.

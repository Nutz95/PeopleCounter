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
	- Default preprocess specs supported for `yolo_global` (single 640x640 letterbox) and `yolo_tiles` (640x640 tiling, overlap default `0.2`) now live in `config/pipeline.yaml` `preprocess` entries.
	- Preprocess kernels live under `app_v2/kernels`, and `GpuPreprocessor` now calls them while telemetry captures GPU timings per frame.
	- Backward compatibility preserved through `CudaPreprocessor` wrapper so `PipelineOrchestrator` keeps its existing flow.

- 🟡 **Tested / In progress**
	- Unit coverage added for preprocess planning behavior (`yolo_global`, `yolo_tiles`).
	- Integration skeleton added for `NVDEC -> preprocess` with runtime skips when stream URL or `PyNvCodec` is unavailable.
	- Ring buffer long-run stress loop added to validate slot recycling over many producer/consumer cycles.

- 🔴 **Pending**
	- Replace placeholder preprocess payload replication with real GPU transform kernels (letterbox/tiling tensors).
	- Model-specific routing in orchestrator/preprocess handoff to avoid passing unrelated flattened inputs to each model.
	- End-to-end validation with real TensorRT model outputs and fusion impact under sustained throughput.

## Update — 2026-02-17 (E2E perf checkpoint)

### What is validated now

- Full app_v2 suite is green in Docker (`42 passed`) using `./5_run_tests.sh --app-version v2`.
- E2E report generation now includes:
  - preprocess decomposition,
  - YOLO global vs YOLO tiles inference comparison,
  - multi-frame statistics (avg, p50, p90, max, N),
  - stage histograms.
- E2E integration test warmup is now configured to skip the first **5 frames** before computing history stats.

### Current bottleneck summary

- Decode and preprocess are now mostly within expected envelopes.
- Main latency pressure is in:
  - `inference_model_yolo_global_ms`,
  - `inference_model_yolo_tiles_ms`,
  - `fusion_wait_ms` / `overlay_lag_ms`,
  - and therefore `end_to_end_ms`.

### Metric collection correction (important)

- E2E inference report aggregation was corrected to use **per-frame values** (steady-state) instead of a **max over all frames** that over-weighted warmup spikes.
- Warmup is now skipped for the first 5 frames in the e2e history view.
- Integration assertions now check YOLO prediction statuses so `gpu_unavailable` regressions are surfaced quickly.

### Action plan (next iterations)

1. **Per-model TRT profiling pass (short-term)**
	- Capture TensorRT execution timeline for YOLO global and YOLO tiles independently.
	- Record effective input shapes, optimization profile selection, and enqueue/execute timings.
	- Verify no hidden synchronization points around stream boundaries.

2. **YOLO tiles strategy optimization**
	- Add configurable tiles downsampling/stride policy and compare accuracy/perf tradeoff.
	- Evaluate reduced tile count and/or adaptive tile selection from motion/ROI heuristics.
	- Keep current strict shape guard enabled.

3. **Fusion backpressure reduction**
	- Instrument queue depths and waiting reasons at fusion boundaries.
	- Confirm publish cadence and ensure no avoidable blocking on late branches.
	- Evaluate strategy-specific timeout/partial-publish policy in controlled benchmark mode.

4. **Engine-level optimization track**
	- Rebuild YOLO engines with verified precision/profile settings for 640x640 throughput.
	- Compare baseline across at least two calibrated TensorRT builds and retain best config.
	- Document reproducible build flags and expected perf envelope.

5. **Architecture hardening for sustained throughput**
	- Finalize model-specific preprocess routing end-to-end (no irrelevant tensors per model).
	- Expand perf regression checks to include p90/p95 thresholds over steady-state windows.
	- Keep reports split by scenario (`preprocess` vs `e2e`) for clean diagnostics.


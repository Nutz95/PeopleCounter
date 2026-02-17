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

## Update — 2026-02-17 (Instrumentation + split tiles attempt)

### What was implemented

**Phase 1: Detailed TRT profiling instrumentation ✅**
- Extended `TensorRTExecutionContext.execute()` to measure:
  - `prepare_batch_ms`: input tensor concatenation/preparation time
  - `enqueue_ms`: TRT `execute_async_v3` enqueue overhead
  - `stream_sync_ms`: stream synchronization wait time
  - `decode_ms`: post-inference decoding time (measured in model classes)
- Propagated these metrics through `YoloGlobalTRT` and `YoloTilingTRT` payloads.
- Updated E2E test to collect and display detailed breakdown per model (`yolo_global`, `yolo_tiles`).
- All 42 tests pass with instrumentation active.

**Phase 2: Parallel tiles split implementation ✅**
- Added config `yolo_tiles_parallel` in `pipeline.yaml` (enabled, groups).
- Created `YoloTilingParallelTRT` class that:
  - Splits tiles into N groups (default 2: moitié/moitié).
  - Creates N independent TRT execution contexts (one per group, no shared state).
  - Executes groups in parallel using `ThreadPoolExecutor` with dedicated CUDA streams (`model:yolo_tiles:g0`, `g1`).
  - Merges detections and aggregates metrics post-execution.
- Modified `ModelBuilder` to instantiate `YoloTilingParallelTRT` conditionally based on config.
- Backward compatibility validated: all tests pass with split OFF (default).
- Forward compatibility validated: all tests pass with split ON.

### Performance results

**Baseline (split OFF, yolo26n-seg.engine)**
- `inference_model_yolo_global_ms`: ~6.3ms ✅ (within 15ms budget)
- `inference_model_yolo_tiles_ms`: ~33ms 🔴 (exceeds 20ms budget by 65%)
- `end_to_end_ms`: ~63ms 🔴 (exceeds 33ms target at 4K@30FPS)

**Split tiles x2 (split ON, 2 groups)**
- `inference_model_yolo_global_ms`: ~9.4ms ✅ (still within budget but higher variance)
- `inference_model_yolo_tiles_ms`: ~23ms 🔴 (reduced from ~33ms, ~30% gain, but still exceeds 20ms budget)
- `end_to_end_ms`: ~49ms 🔴 (reduced from ~63ms, ~22% gain, but still exceeds target)

**Key observations**
- Parallel split provides measurable gain (~10ms reduction on tiles, ~14ms on e2e).
- However, target of `<= 10ms` for tiles inference not reached (gap: 23ms vs 10ms).
- ThreadPoolExecutor coordination overhead may limit parallel efficiency (Python GIL contention).
- GPU compute is underutilized due to small per-group batch sizes and synchronization barriers.

### Decision

- Keep feature **OFF by default** (`yolo_tiles_parallel.enabled: false`) as it doesn't meet the 10ms target.
- Feature is production-ready and **activable via config** for future experimentation.
- Further optimization paths:
  1. **Engine-level**: Rebuild with INT8/FP16 profiles optimized for batch efficiency.
  2. **Architecture**: Replace ThreadPoolExecutor with native CUDA stream async pattern (remove Python GIL).
  3. **Batching strategy**: Tune tile overlap/count to reduce total compute while preserving coverage.

### Next steps

1. **Immediate**: Document profiling metrics in HTML reports (prepare/enqueue/sync/decode breakdown).
2. **Short-term**: Benchmark engine rebuild with calibrated INT8 quantization for yolo26n.
3. **Medium-term**: Profile pure CUDA stream async coordination (no ThreadPool) to eliminate GIL overhead.
4. **Long-term**: Adaptive tiling strategy based on ROI/motion heuristics (reduce unnecessary tiles).

## Update — 2026-02-17 (Scaling investigation)

### Observed behavior: Parallel split does NOT scale linearly

**Performance degradation with more groups:**
- 2 groups → **21.6ms** ✓ (~30% gain vs baseline 33ms)
- 4 groups → **28.4ms** ✗ (WORSE than 2 groups!)
- 8 groups → **54.5ms** ✗✗ (WORSE than baseline!)

**Side effect on yolo_global:**
- More tile groups → yolo_global inference also slows down
- This indicates system-wide resource contention, not just tiles issue

### Root cause analysis

**1. Python GIL (Global Interpreter Lock)**
- `ThreadPoolExecutor` does NOT provide true parallelism in Python
- Threads are scheduled sequentially by GIL with context switching overhead
- Overhead GROWS linearly with number of threads
- **Evidence**: Wall-clock time increases despite "parallel" execution

**2. GPU resource contention**
- 4-8 concurrent CUDA streams compete for GPU Streaming Multiprocessors (SMs)
- Memory bandwidth saturation with multiple concurrent kernel launches
- TensorRT context switching overhead

**3. Batch size fragmentation**
- Baseline: 32 tiles in 1 batch → efficient GPU utilization
- 2 groups: 16 tiles/batch → still good batch efficiency
- 4 groups: 8 tiles/batch → reduced batch amortization
- 8 groups: 4 tiles/batch → loses batch efficiency completely
- **TensorRT penalty**: Small batches don't saturate GPU, waste kernel launch overhead

**4. TRT Runtime overhead**
- Each group creates a separate TRT execution context
- More contexts → more memory allocation, more logger warnings
- Fragmented GPU memory → reduced cache efficiency

### Why 2 groups is the sweet spot

- **Balance**: Enough parallelism to hide latency, not too much overhead
- **Batch size**: 16 tiles/group still benefits from batch efficiency
- **GPU utilization**: 2 streams can saturate GPU without excessive contention
- **Python overhead**: 2 threads manageable by GIL scheduler

### Recommended actions

**Priority 1: INT8 Quantization (IMMEDIATE)**
- Use [prepare_yolo_int8.py](prepare_yolo_int8.py) script to build calibrated INT8 engine
- Download COCO val2017 subset (500-1000 images sufficient)
- Expected gain: 2-4× speedup on RTX 5060 Ti (INT8 Tensor Cores)
- Command:
  ```bash
  python prepare_yolo_int8.py \
    --model models/pt/yolo26n-seg.pt \
    --download-coco \
    --num-calibration-images 500 \
    --output models/tensorrt/yolo26n-seg-int8.engine
  ```

**Priority 2: CUDA Stream Pure Async (MEDIUM-TERM)**
- Replace `ThreadPoolExecutor` with native CUDA stream pattern
- Modify `TensorRTExecutionContext.execute()` to support non-blocking mode
- Use `cudaEventRecord` + `cudaEventSynchronize` for coordination
- Expected gain: Eliminate Python GIL overhead completely

**Priority 3: Constrain groups to 2-3 maximum**
- Add validation in config: `assert 1 <= groups <= 3`
- Document scaling limitation in config comments
- Sweet spot: 2 groups for current architecture

**NOT RECOMMENDED (for now):**
- Adaptive tiling: Adds complexity, uncertain ROI
- More than 3 groups: Proven to degrade performance
- Engine rebuild without INT8: Won't reach 10ms target

### TRT Logger warnings

The warnings about logger registration are **benign** but indicate architectural issue:
- Each `TensorRTEngineLoader` creates a separate runtime
- TensorRT expects single global logger per process
- Solution: Share single TRT runtime across all contexts (future refactor)




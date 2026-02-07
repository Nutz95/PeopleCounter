## Plan: Mask timing telemetry

TL;DR mask payloads now carry creation/network timestamps and the UI consumes them for the adaptive mask timing card; the next phase focuses on presenting a latency graph, cleaning up console logs, and documenting progress in the new performance plan.

**Completed work**
1. Emitted `created_at`/`created_at_ts` metadata from `yolo_seg_people_counter.py` and encoded it in `_encode_mask_blob`.
2. `camera_app_pipeline.py` aggregates the new `yolo_mask_payload_*` fields, records `[MASK TIMING]` logs, and includes `mask_sent_at` before sending the payload.
3. The web UI (`static/js/app.js`) now shows creation/send/display latencies and adapts polling based on the reported YOLO FPS, so overlay timing issues are visible to users.

**Next steps**
1. Hand off the latency graph, console log cleanup, and 25–30 fps target narrative to [plans/performance-latency-plan.md], which now guides the telemetry story for the mask timing card.
2. Continue recording any future mask timing observations in the performance plan so this document can serve as the historical record of the instrumentation rollout.
3. Capture end-to-end GPU work via the CUDA profiler helpers by setting `ENABLE_CUDA_PROFILING=1` (and `YOLO_USE_GPU_PREPROC=1`, `YOLO_USE_GPU_POST=1`, and the density pipeline’s GPU flags), then run `nsys profile --trace=cuda --capture-range=cudaProfilerApi --output=people_counter ./run_people_counter.sh` so the new `cudaProfilerStart/Stop` hooks emit a clean timerange for crop/fuse/infer.

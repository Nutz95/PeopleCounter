## Plan: Mask timing telemetry

TL;DR mask payloads now carry creation/network timestamps and the UI consumes them for the adaptive mask timing card; the next phase focuses on presenting a latency graph, cleaning up console logs, and documenting progress in the new performance plan.

**Completed work**
1. Emitted `created_at`/`created_at_ts` metadata from `yolo_seg_people_counter.py` and encoded it in `_encode_mask_blob`.
2. `camera_app_pipeline.py` aggregates the new `yolo_mask_payload_*` fields, records `[MASK TIMING]` logs, and includes `mask_sent_at` before sending the payload.
3. The web UI (`static/js/app.js`) now shows creation/send/display latencies and adapts polling based on the reported YOLO FPS, so overlay timing issues are visible to users.

**Next steps**
1. Implement the performance/latency plan (see `plans/performance-latency-plan.md`) to render a graph of total mask/YOLO latency per frame and highlight regressions.
2. Clean up console logging so only `[MASK TIMING]` lines remain, centralize verbose diagnostics via `EXTREME_DEBUG`, and document the remaining observation points in `README_PARAMETERS.md`.
3. Provide a `mask_timing-plan` update that references the new performance plan and explains how the metrics now tie into the mask overlay behavior.

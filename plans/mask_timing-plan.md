## Plan: Mask Timing Instrumentation

TL;DR add timestamps to mask payloads on the backend, surface those timestamps in the metrics response, and make the web UI adapt its polling interval to the reported YOLO FPS so we can trace why masks currently refresh only once per second.

**Phases 1**
1. **Phase 1: Instrument mask payloads**
    - **Objective:** Stamp each mask payload with a mask creation timestamp and propagate it into the metrics dictionary.
    - **Files/Functions to Modify/Create:** `yolo_seg_people_counter.py` (`_build_mask_payload`), `camera_app_pipeline.py` (metrics builder and mask attachment), `web_server.py` (metrics endpoint if needed).
    - **Tests to Write:** None (manual verification only).
    - **Steps:**
        1. When `_build_mask_payload` succeeds, add `created_at`/`mask_timestamp` fields and log each creation event for traceability.
        2. Have `CameraAppPipeline` copy that timestamp into the metrics payload, append a `mask_sent_at` timestamp before publishing, and log both to surface the delay.
        3. Show both timestamps or the delta in the backend logs so the issue is visible before touching the frontend polling logic.

**Open Questions**
1. Should we also track `mask_sent_at` timestamp before the metrics response is emitted?

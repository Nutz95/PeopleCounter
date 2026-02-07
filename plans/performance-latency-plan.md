## Plan: Performance & latency instrumentation

TL;DR: visualize the mask/YOLO latency budget, consolidate log output, and document the performance goals so the telemetry cards match the 25–30 fps target once the pipeline stabilizes.

## Status
- Phase 1 (latency graph + target bands) now ships in the UI and draws from the `[MASK TIMING]` payloads.
- Phase 2 (log cleanup via `EXTREME_DEBUG`, README/parameter updates, and plan alignment) is complete, so this document now tracks the 25–30 fps narrative.

**Phases**
1. **Phase 1: Surface latency breakdowns**
    - **Objective:** Add a latency graph or timeline so operators can spot which stage (creation, send, display) is taking the longest.
    - **Files/Functions to Modify/Create:** `static/js/app.js` (graph rendering), `templates/index.html` (latency card), `camera_app_pipeline.py` (metrics payload fields), possibly CSS for the new graph.
    - **Tests to Write:** Manual verification of the UI graph and verifying logs show the new metrics.
    - **Steps:**
        1. Define a new UI component (canvas or chart library) that plots the latest mask/YOLO latencies over time, grouping creation/send/display.
        2. Ensure the backend metrics include a total latency field and push it to the UI along with the existing timestamps.
        3. Update the `static/js` helpers to maintain a rolling buffer of latencies and refresh the graph at the adaptive polling interval.
2. **Phase 2: Clean logs & document goals**
    - **Objective:** Simplify console output, tie the plans to the new performance metrics, and capture the 25–30 fps target in the docs.
    - **Files/Functions to Modify/Create:** `camera_app_pipeline.py` (log lines), `plans/mask_overlay_roadmap.md`, `plans/mask_timing-plan.md`, `README_PARAMETERS.md`, `README_ARCHITECTURE.md`, `README.md` (where the performance target is referenced).
    - **Tests to Write:** None (focus on readability and docs).
    - **Steps:**
        1. Remove redundant console noise, leaving `[MASK TIMING]` as the primary perf line, and gate extra details behind `EXTREME_DEBUG`.
        2. Expand the overlay and timing plans to mention the latency graph and the 25–30 fps plateau, so future updates can track it.
        3. Document the performance targets in the READMEs and parameter guide, linking back to this plan and the mask timing metrics.

**Open Questions**
1. Do we want the latency graph to live inside the existing "mask timings" card or on a separate dashboard panel?
2. Should the performance plan expose the 25–30 fps target as a warning threshold (e.g., color the graph red below 25 fps)?
3. Are there additional log lines (beyond `[MASK TIMING]`) that should stay for diagnostics even after the cleanup?
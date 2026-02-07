## Plan: Documentation refresh and roadmap alignment

This plan captures the README rewrite, the new architecture guide, the parameter reference, and the roadmap synchronization so the mask overlay, metrics instrumentation, and docs stay traceable.

**Phases**
1. **Phase 1: Restructure README.md and README_DOCKER.md**
    - **Objective:** Provide an inviting quick-start entry that explains the installation flow (Windows+WSL bridge, `.env` profiles, Docker image build, model setup) and point readers to more specific guides instead of duplicating outdated sections.
    - **Files/Functions to Modify/Create:** `README.md`, `README_DOCKER.md`
    - **Tests to Write:** None (documentation changes only).
    - **Steps:**
        1. Restructure `README.md` to describe the install steps, clarify the `.env` profiles, add the Windows bridge workflow, and link to the new architecture/parameter/plan docs.
        2. Rewrite `README_DOCKER.md` so it stays container-focused (build instructions, prerequisites) and points to `README_ARCHITECTURE.md` for the full system picture.
        3. Mention in both README files that any change touching masks, architecture, or metrics must be mirrored across the plans (`plans/documentation-refresh-plan.md`, `plans/mask_overlay_roadmap.md`, `plans/mask_timing-plan.md`, `plans/performance-latency-plan.md`).
2. **Phase 2: Create README_ARCHITECTURE.md and README_PARAMETERS.md**
    - **Objective:** Offer a deep dive into the CUDA/CPU mask+metrics pipeline, the mask timeline, and a dedicated parameter guide that explains how each `.env` variable shapes the experience.
    - **Files/Functions to Modify/Create:** `README_ARCHITECTURE.md`, `README_PARAMETERS.md`
    - **Tests to Write:** None (documentation changes only).
    - **Steps:**
        1. Document the host/capture/pipeline flow with mermaid diagrams, mask creation, telemetry propagation, and CPU vs GPU responsibilities.
        2. Create a parameter reference listing every backend flag, density knob, tiling option, and infrastructure toggle, then link back to this guide from the main README and Docker guide.
3. **Phase 3: Sync roadmaps and performance plan**
    - **Objective:** Update the overlay/timing plans, add the new performance/latency plan, and ensure the README pointers (including the new parameter guide) reflect the completed work plus the outstanding latency graph/log cleanup goals.
    - **Files/Functions to Modify/Create:** `plans/mask_overlay_roadmap.md`, `plans/mask_timing-plan.md`, `plans/performance-latency-plan.md`
    - **Tests to Write:** None (documentation changes only).
    - **Steps:**
        1. Capture the mask overlay refinements, telemetry card, and parameter document references in `plans/mask_overlay_roadmap.md` so future work knows what’s done and what remains (density re-enable, performance targets).
        2. Update `plans/mask_timing-plan.md` to mention the existing timestamps, UI card, and the future latency graph/log cleanup tasks now tracked in `plans/performance-latency-plan.md`.
        3. Ensure every README points to the relevant plans (Documentation refresh, overlay, timing, performance) so the documentation and roadmap stay aligned across updates.

**Open Questions**
1. Should the README list the specialized docs (Docker, architecture, parameters, density, YOLO) in a quick reference table with links to their respective plans?
2. Where is the best place to note that the entire set of READMEs must stay in English once the translation pass begins?
3. Does the performance plan need to expose the 25–30 fps target as a warning level or keep it descriptive only?

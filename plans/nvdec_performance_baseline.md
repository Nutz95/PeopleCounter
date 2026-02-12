# NVDEC Pipeline Baseline

Now that the NVDEC decoder and clean-architecture loop are wired together, we can capture the first end-to-end baseline before the inference outputs are polished. Follow these steps whenever you need a reference point for future optimizations:

1. **Prepare the test stream**
   - Ensure your Windows bridge stream is running and reachable (the one that feeds `camera_bridge.py`).
   - Export the RTSP URL into the environment variable `NVDEC_TEST_STREAM_URL`, e.g. `export NVDEC_TEST_STREAM_URL=rtsp://192.168.1.62:5002/video_feed`.

2. **Run the integration test suite**
   - Rebuild the NVDEC image if needed (`./1_prepare.sh` then `./2_prepare_nvdec.sh`).
   - Execute `./5_run_tests.sh --app-version v2`. This runs `pytest app_v2/tests`, including `test_nvdec_decode.py`, inside `people-counter:gpu-final-nvdec` with the real stream enabled.
   - Capture the pytest logs (save `pytest.log` or use tooling) so you know how long decode + scheduler + aggregator took in this baseline run.

3. **Gather runtime metrics**
   - Launch the orchestrator against the same stream: `./4_run_app.sh --app-version v2 --profile rtx_extreme "$NVDEC_TEST_STREAM_URL"`.
   - Watch the `[MASK TIMING]` and `[GPU PERF]` lines; they record the `PerformanceTracker` timings for NVDEC, preprocessors, and each inference model.
   - Use `nvidia-smi` (inside the container) and the Flask `/api/last` endpoint to correlate frame IDs with GPU/latency utilization.

4. **Record the baseline**
   - Copy the relevant log lines (timestamps, frame IDs, latencies) into this document or a dedicated entry in `plans/` so future runs can compare.
   - Note the container image/tag, CUDA version, and whether `NVDEC_TEST_STREAM_URL` was reachable.

When the inference engines are ready, rerun these steps and update the recorded metrics so we always have the latest performance reference.

// ── sse.js ─────────────────────────────────────────────────────────────────
// Server-Sent Events connection: receives inference telemetry + overlay data
// and drives all per-frame metric/overlay updates.
// Depends on: state.js, charts.js, overlays.js, controls.js
// ──────────────────────────────────────────────────────────────────────────

const sse = new EventSource('/api/stream');

sse.addEventListener('message', e => {
  let msg;
  try { msg = JSON.parse(e.data); } catch { return; }
  if (!msg || !msg.frame_id) return;

  // ── Passthrough heartbeat ─────────────────────────────────────
  // No inference payload — just count FPS from the frame flow.
  if (msg.passthrough) {
    const now = Date.now();
    fpsTs.push(now);
    while (fpsTs.length && now - fpsTs[0] > 2000) fpsTs.shift();
    const fps = fpsTs.length > 1
      ? Math.round((fpsTs.length - 1) / ((now - fpsTs[0]) / 1000))
      : 0;
    $connStatus.textContent  = '⬤ Live';
    $connStatus.style.color  = 'var(--accent)';
    $fpsChip.textContent     = fps ? `${fps} fps` : '— fps';
    $latencyChip.textContent = `frame ${msg.frame_id}`;
    $latencyChip.style.background = 'rgba(42,223,165,0.12)';
    $frameLabel.textContent  = `frame ${msg.frame_id}`;
    $e2eValue.textContent    = '—';
    $fpsDerived.textContent  = fps ? `${fps} fps` : '— fps';
    $countValue.textContent  = '—';
    $preprocValue.textContent = '—';
    $tilesValue.textContent  = '—';
    $globalValue.textContent = '—';
    $fusionValue.textContent = '—';

    if (window.__benchmarkCapture && window.__benchmarkCapture.isActive()) {
      window.__benchmarkCapture.push({
        frame_id: msg.frame_id,
        mode: _activeMode,
        sync_mode: _activeSyncMode,
        passthrough: 1,
        count: null,
        fps,
        e2e_ms: null,
        preproc_ms: null,
        infer_tiles_ms: null,
        infer_global_ms: null,
        infer_critical_ms: null,
        fusion_ms: null,
        postdecode_ms: null,
        decode_stage_filter_ms: null,
        decode_stage_nms_ms: null,
        decode_stage_export_ms: null,
        decode_stage_pack_ms: null,
        other_ms: null,
        nvdec_ms: null,
        src_wait_ms: null,
        src_age_ms: null,
        src_copy_sync_ms: null,
        trt_prepare_ms: null,
        trt_sync_ms: null,
        decode_ms: null,
        video_encode_ms: null,
        video_encode_wait_ms: null,
        video_encode_copy_ms: null,
        video_encode_push_ms: null,
        publish_total_ms: null,
      });
    }
    return;
  }

  const payload = Array.isArray(msg.payload) ? msg.payload : [];
  const tel     = (payload.find(p => p && p.telemetry) || {}).telemetry || {};
  if (typeof tel.server_side_overlay_active === 'number') {
    window.__serverSideOverlayActive = tel.server_side_overlay_active > 0.5;
  }
  if (typeof tel.server_side_heatmap_active === 'number') {
    window.__serverSideHeatmapActive = tel.server_side_heatmap_active > 0.5;
  }
  const asyncNoOverlayMode = _activeSyncMode !== 'sync';
  const serverSideHeatmapActive = asyncNoOverlayMode || !!window.__serverSideHeatmapActive || !!window.__serverSideOverlayActive;

  // Update cached video source dimensions whenever the server reports them.
  if (tel.frame_width  > 0) _videoSrcW = tel.frame_width;
  if (tel.frame_height > 0) _videoSrcH = tel.frame_height;

  // ── Person count ─────────────────────────────────────────────
  let count = 0;
  const hasPackedForFrame = (
    _packedDetections != null
    && _packedDetectionsFrameId === msg.frame_id
    && _packedDetectionsRowWidth >= 3
    && _packedDetections.length >= _packedDetectionsRowWidth
  );
  const hasPackedCentersForFrame = hasPackedForFrame && _packedDetectionsRowWidth === 3;
  if (hasPackedForFrame) {
    count = Math.max(count, Math.floor(_packedDetections.length / _packedDetectionsRowWidth));
  }
  for (const p of payload) {
    if (!p) continue;
    if (Array.isArray(p.detections))          count = Math.max(count, p.detections.length);
    if (typeof p.detection_count === 'number') count = Math.max(count, p.detection_count);
    if (typeof p.count === 'number')          count = Math.max(count, p.count);
    if (typeof p.hotspot_count === 'number')  count = Math.max(count, p.hotspot_count);
    if (typeof p.density_count === 'number')
      count = Math.max(count, Math.round(p.density_count));
  }

  // ── Metrics from telemetry snapshot ──────────────────────────
  const e2e      = +(tel.end_to_end_ms                  ?? 0);
  const preproc  = +(tel.preprocess_ms                  ?? 0);
  const preprocDispatch = +(tel.preprocess_dispatch_ms ?? 0);
  const preprocSync = +(tel.preprocess_sync_ms ?? 0);
  const preprocPlanYolo = +(tel.preprocess_plan_ms_yolo ?? 0);
  const preprocKernelYolo = +(tel.preprocess_kernel_ms_yolo ?? 0);
  const preprocTileCountYolo = +(tel.preprocess_tile_count_yolo ?? 0);
  const preprocPlanCrowd = +(tel.preprocess_plan_ms_crowd ?? 0);
  const preprocKernelCrowd = +(tel.preprocess_kernel_ms_crowd ?? 0);
  const preprocTileCountCrowd = +(tel.preprocess_tile_count_crowd ?? 0);
    // Pick inference time across all model variants (yolo, crowd, density).
    const tiles   = +(tel.inference_model_yolo_tiles_ms   ?? tel.inference_model_crowd_tiles_ms  ?? 0);
    const global_ = +(tel.inference_model_yolo_global_ms  ?? tel.inference_model_crowd_global_ms ?? 0);
  const fusion   = +(tel.fusion_wait_ms                 ?? 0);
  const postDecode = +(tel.decode_model_sum_ms          ?? 0);
  const decodeFilter = +(tel.decode_stage_filter_sum_ms ?? 0);
  const decodeNms = +(tel.decode_stage_nms_sum_ms ?? 0);
  const decodeExport = +(tel.decode_stage_export_sum_ms ?? 0);
  const decodePack = +(tel.decode_stage_pack_sum_ms ?? 0);
  const nvdec = +(tel.nvdec_ms ?? 0);
  const videoEncode = +(tel.video_encode_last_ms ?? 0);
  const videoEncodeWait = +(tel.video_encode_wait_event_ms ?? 0);
  const videoEncodeCopy = +(tel.video_encode_cpu_copy_ms ?? 0);
  const videoEncodePush = +(tel.video_encode_push_ms ?? 0);
  const videoInflight = +(tel.video_encode_inflight ?? 0);
  const videoStashed = +(tel.video_encode_stashed ?? 0);
  const publishTotal = +(tel.server_publish_total_ms ?? 0);
  const srcWait = +(tel.frame_source_wait_latest_ms ?? 0);
  const srcAge = +(tel.frame_source_age_at_consume_ms ?? 0);
  const srcCopySync = +(tel.frame_source_copy_sync_ms ?? 0);
  let trtPrepare = +(tel.prepare_batch_ms ?? 0);
  let trtSync = +(tel.stream_sync_ms ?? 0);
  for (const p of payload) {
    if (!p || typeof p !== 'object') continue;
    const pb = +(p.prepare_batch_ms ?? 0);
    const ss = +(p.stream_sync_ms ?? 0);
    if (pb > trtPrepare) trtPrepare = pb;
    if (ss > trtSync) trtSync = ss;
  }
  const decodeMs = +(tel.decode_model_sum_ms ?? 0);

  const hasTelemetry = e2e > 0 || tiles > 0 || global_ > 0;

  // ── FPS ──────────────────────────────────────────────────────
  const now = Date.now();
  if (hasTelemetry) {
    fpsTs.push(now);
    while (fpsTs.length && now - fpsTs[0] > 2000) fpsTs.shift();
  }
  const fps = fpsTs.length > 1
    ? Math.round((fpsTs.length - 1) / ((now - fpsTs[0]) / 1000))
    : 0;

  // ── Compute rolling averages ──────────────────────────────────
  const avgE2e     = hasTelemetry ? rollAvg(_smooth.e2e,    e2e)     : (e2eHist.length ? e2eHist[e2eHist.length - 1] : 0);
  const avgTiles   = hasTelemetry ? rollAvg(_smooth.tiles,  tiles)   : 0;
  const avgGlobal  = hasTelemetry ? rollAvg(_smooth.global, global_) : 0;
  const avgPreproc = hasTelemetry ? rollAvg(_smooth.preproc, preproc): 0;
  const avgFusion  = hasTelemetry ? rollAvg(_smooth.fusion, fusion)  : 0;
  const avgPostDecode  = hasTelemetry ? rollAvg(_smooth.decode,  postDecode) : 0;
  const avgDecodeFilter = hasTelemetry ? rollAvg(_smooth.decodeFilter, decodeFilter) : 0;
  const avgDecodeNms = hasTelemetry ? rollAvg(_smooth.decodeNms, decodeNms) : 0;
  const avgDecodeExport = hasTelemetry ? rollAvg(_smooth.decodeExport, decodeExport) : 0;
  const avgDecodePack = hasTelemetry ? rollAvg(_smooth.decodePack, decodePack) : 0;
  const avgNvdec   = hasTelemetry ? rollAvg(_smooth.nvdec, nvdec) : 0;
  const avgEncode  = hasTelemetry ? rollAvg(_smooth.encode, videoEncode) : 0;
  const avgPublish = hasTelemetry ? rollAvg(_smooth.publish, publishTotal) : 0;
  const inferCritical = Math.max(tiles, global_);
  // E2E starts at frame.timestamp_ns, which is set right after NVDEC unblock.
  // Therefore nvdec_ms is an upstream metric (informative) and MUST NOT be
  // subtracted from E2E component accounting.
  const e2eAccounted = preproc + inferCritical + fusion + postDecode;
  const otherMs = Math.max(0, e2e - e2eAccounted);
  const avgOther = hasTelemetry ? rollAvg(_smooth.other, otherMs) : 0;

  // ── Decompose other_ms into sub-metrics ──────────────────────
  // These are sampled from telemetry to understand where the gap goes.
  const otherPublish = +(tel.server_publish_total_ms ?? 0);
  const otherEncode = +(tel.video_encode_last_ms ?? 0);
  const otherEncodeWait = +(tel.video_encode_wait_event_ms ?? 0);
  const otherEncodeCopy = +(tel.video_encode_cpu_copy_ms ?? 0);
  const otherEncodePush = +(tel.video_encode_push_ms ?? 0);
  const orchestratorFlatten = +(tel.orchestrator_flatten_ms ?? 0);
  const orchestratorRegister = +(tel.orchestrator_register_ms ?? 0);
  const orchestratorCollect = +(tel.orchestrator_collect_ms ?? 0);
  const aggregatorCollectLatency = +(tel.aggregator_collect_latency_ms ?? 0);
  const serverJsonEncode = +(tel.server_json_encode_ms ?? 0);
  const serverSsePublish = +(tel.server_sse_publish_ms ?? 0);
  const serverCompactPayload = +(tel.server_compact_payload_ms ?? 0);
  const serverMetaWsPush = +(tel.server_meta_ws_push_ms ?? 0);
  const serverTelemetryUpdate = +(tel.server_telemetry_update_ms ?? 0);
  const serverLockHold = +(tel.server_lock_hold_ms ?? 0);
  const serverLockAcquired = +(tel.server_lock_acquired ?? 0);
  const otherCollect = Math.max(0, otherMs - (otherPublish + otherEncode));

  // ── Update DOM ───────────────────────────────────────────────
  $connStatus.textContent = '⬤ Live';
  $connStatus.style.color = 'var(--accent)';

  $countValue.textContent = count;
  rollAvgWin(_smooth.count, count, COUNT_SMOOTH_WIN);
  $frameLabel.textContent = `frame ${msg.frame_id}`;

  $e2eValue.textContent     = avgE2e.toFixed(1);
  $fpsDerived.textContent   = fps ? `${fps} fps` : '— fps';
  $preprocValue.textContent = avgPreproc.toFixed(2);
  $tilesValue.textContent   = avgTiles.toFixed(2);
  $globalValue.textContent  = avgGlobal.toFixed(2);
  $fusionValue.textContent  = avgFusion.toFixed(2);

  $latencyChip.textContent = `${avgE2e.toFixed(1)} ms`;
  $fpsChip.textContent     = fps ? `${fps} fps` : '— fps';
  $latencyChip.style.background =
    avgE2e <= TARGET_MS
      ? 'rgba(42,223,165,0.18)'
      : avgE2e <= TARGET_MS * 1.2
      ? 'rgba(245,158,11,0.22)'
      : 'rgba(239,68,68,0.22)';

  $ovPreprocess.textContent = `preprocess ${avgPreproc.toFixed(1)} ms`;
  $ovGlobal.textContent     = `global  ${avgGlobal.toFixed(1)} ms`;
  $ovTiles.textContent      = `tiles   ${avgTiles.toFixed(1)} ms  postdecode ${avgPostDecode.toFixed(1)} ms [f ${avgDecodeFilter.toFixed(1)} / nms ${avgDecodeNms.toFixed(1)} / xfer ${avgDecodeExport.toFixed(1)} / pack ${avgDecodePack.toFixed(1)}]  nvdec(upstream) ${avgNvdec.toFixed(1)} ms`;
  if ($ovOther) {
    const queueState = videoInflight > 0 ? (videoStashed > 0 ? 'enc:busy+stash' : 'enc:busy') : 'enc:idle';
    $ovOther.textContent = `other(e2e gap) ${avgOther.toFixed(1)} ms  fusion ${avgFusion.toFixed(2)} ms  pub ${avgPublish.toFixed(2)} ms  enc ${avgEncode.toFixed(1)} ms (${queueState}, wait ${videoEncodeWait.toFixed(1)} / copy ${videoEncodeCopy.toFixed(1)} / push ${videoEncodePush.toFixed(1)} ms)`;
  }

  if (window.__benchmarkCapture && window.__benchmarkCapture.isActive()) {
    window.__benchmarkCapture.push({
      frame_id: msg.frame_id,
      mode: _activeMode,
      sync_mode: _activeSyncMode,
      passthrough: 0,
      count,
      fps,
      e2e_ms: +e2e.toFixed(4),
      preproc_ms: +preproc.toFixed(4),
      preproc_dispatch_ms: +preprocDispatch.toFixed(4),
      preproc_sync_ms: +preprocSync.toFixed(4),
      preproc_plan_yolo_ms: +preprocPlanYolo.toFixed(4),
      preproc_kernel_yolo_ms: +preprocKernelYolo.toFixed(4),
      preproc_tile_count_yolo: +preprocTileCountYolo.toFixed(0),
      preproc_plan_crowd_ms: +preprocPlanCrowd.toFixed(4),
      preproc_kernel_crowd_ms: +preprocKernelCrowd.toFixed(4),
      preproc_tile_count_crowd: +preprocTileCountCrowd.toFixed(0),
      infer_tiles_ms: +tiles.toFixed(4),
      infer_global_ms: +global_.toFixed(4),
      infer_critical_ms: +inferCritical.toFixed(4),
      fusion_ms: +fusion.toFixed(4),
      postdecode_ms: +postDecode.toFixed(4),
      decode_stage_filter_ms: +decodeFilter.toFixed(4),
      decode_stage_nms_ms: +decodeNms.toFixed(4),
      decode_stage_export_ms: +decodeExport.toFixed(4),
      decode_stage_pack_ms: +decodePack.toFixed(4),
      other_ms: +otherMs.toFixed(4),
      other_publish_ms: +otherPublish.toFixed(4),
      other_encode_ms: +otherEncode.toFixed(4),
      other_encode_wait_ms: +otherEncodeWait.toFixed(4),
      other_encode_copy_ms: +otherEncodeCopy.toFixed(4),
      other_encode_push_ms: +otherEncodePush.toFixed(4),
      other_collect_ms: +otherCollect.toFixed(4),
      orchestrator_flatten_ms: +orchestratorFlatten.toFixed(4),
      orchestrator_register_ms: +orchestratorRegister.toFixed(4),
      orchestrator_collect_ms: +orchestratorCollect.toFixed(4),
      aggregator_collect_latency_ms: +aggregatorCollectLatency.toFixed(4),
      server_json_encode_ms: +serverJsonEncode.toFixed(4),
      server_sse_publish_ms: +serverSsePublish.toFixed(4),
      server_compact_payload_ms: +serverCompactPayload.toFixed(4),
      server_meta_ws_push_ms: +serverMetaWsPush.toFixed(4),
      server_telemetry_update_ms: +serverTelemetryUpdate.toFixed(4),
      server_lock_hold_ms: +serverLockHold.toFixed(4),
      server_lock_acquired: +serverLockAcquired.toFixed(4),
      nvdec_ms: +nvdec.toFixed(4),
      src_wait_ms: +srcWait.toFixed(4),
      src_age_ms: +srcAge.toFixed(4),
      src_copy_sync_ms: +srcCopySync.toFixed(4),
      trt_prepare_ms: +trtPrepare.toFixed(4),
      trt_sync_ms: +trtSync.toFixed(4),
      decode_ms: +decodeMs.toFixed(4),
      video_encode_ms: +videoEncode.toFixed(4),
      video_encode_wait_ms: +videoEncodeWait.toFixed(4),
      video_encode_copy_ms: +videoEncodeCopy.toFixed(4),
      video_encode_push_ms: +videoEncodePush.toFixed(4),
      publish_total_ms: +publishTotal.toFixed(4),
    });
  }

  // ── Overlays ─────────────────────────────────────────────────
  if (!previewOnly && !serverSideHeatmapActive) {
    if (showMask) {
      if (hasPackedForFrame) drawMaskPacked(_packedDetections);
      else drawMask(payload);
    }
      else if (!window._webglOverlayActive) maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (showSeg)  drawSegMask(payload);
    else segCtx.clearRect(0, 0, segCanvas.width, segCanvas.height);
  }
  if (!previewOnly && showHeatmap && !serverSideHeatmapActive) {
    if (hasPackedCentersForFrame) {
      // Dense P2PNet path: prefer packed centers transport over JSON hotspots.
      if (!window._webglOverlayActive) drawHeatmapPacked(_packedDetections);
      else heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
    } else {
      drawHeatmap(payload);
    }
  } else {
    heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
    if (serverSideHeatmapActive) {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      segCtx.clearRect(0, 0, segCanvas.width, segCanvas.height);
    }
  }

  // ── History & latency chart ───────────────────────────────────
  if (hasTelemetry) {
    e2eHist.push(e2e);
    tilesHist.push(tiles);
    if (e2eHist.length   > MAX_HISTORY) e2eHist.shift();
    if (tilesHist.length > MAX_HISTORY) tilesHist.shift();
    drawChart();
  }
});

sse.onerror = () => {
  $connStatus.textContent = '⬤ Reconnecting…';
  $connStatus.style.color = '#f59e0b';
};

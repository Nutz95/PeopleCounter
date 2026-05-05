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
    return;
  }

  const payload = Array.isArray(msg.payload) ? msg.payload : [];
  const tel     = (payload.find(p => p && p.telemetry) || {}).telemetry || {};

  // Update cached video source dimensions whenever the server reports them.
  if (tel.frame_width  > 0) _videoSrcW = tel.frame_width;
  if (tel.frame_height > 0) _videoSrcH = tel.frame_height;

  // ── Person count ─────────────────────────────────────────────
  let count = 0;
  for (const p of payload) {
    if (!p) continue;
    if (Array.isArray(p.detections))          count = Math.max(count, p.detections.length);
    if (typeof p.count === 'number')          count = Math.max(count, p.count);
    if (typeof p.hotspot_count === 'number')  count = Math.max(count, p.hotspot_count);
    else if (typeof p.density_count === 'number')
      count = Math.max(count, Math.round(p.density_count));
  }

  // ── Metrics from telemetry snapshot ──────────────────────────
  const e2e      = +(tel.end_to_end_ms                  ?? 0);
  const preproc  = +(tel.preprocess_ms                  ?? 0);
  const tiles    = +(tel.inference_model_yolo_tiles_ms  ?? 0);
  const global_  = +(tel.inference_model_yolo_global_ms ?? 0);
  const fusion   = +(tel.fusion_wait_ms                 ?? 0);
  const decodeSum = +(tel.decode_model_sum_ms           ?? 0);

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
  const avgDecode  = hasTelemetry ? rollAvg(_smooth.decode,  decodeSum) : 0;

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
  $ovTiles.textContent      = `tiles   ${avgTiles.toFixed(1)} ms  decode ${avgDecode.toFixed(1)} ms`;

  // ── Overlays ─────────────────────────────────────────────────
  if (!previewOnly) {
    if (showMask) drawMask(payload);
    else maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (showSeg)  drawSegMask(payload);
    else segCtx.clearRect(0, 0, segCanvas.width, segCanvas.height);
  }
  if (!previewOnly && showHeatmap) drawHeatmap(payload);
  else heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);

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

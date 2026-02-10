const baseUrl = window.location.origin;
const debugToggle = document.getElementById('debugToggle');
const fpsBadge = document.getElementById('fpsBadge');
const fpsStatus = document.getElementById('fpsStatus');
const profileStart = document.getElementById('profileStart');
const profileStop = document.getElementById('profileStop');
const profileClear = document.getElementById('profileClear');
const profileViewSelect = document.getElementById('profileViewSelect');
const profileViewBadge = document.getElementById('profileViewBadge');
const profileLog = document.getElementById('profileLog');
const fpsValue = document.getElementById('fpsValue');
const profileState = document.getElementById('profileState');
const videoWrapper = document.getElementById('videoWrapper');
const perfSummary = document.getElementById('perfSummary');
const captureTimeLabel = document.getElementById('captureTimeLabel');
const overlayToggle = document.getElementById('overlayToggle');
const overlayStatus = document.getElementById('overlayStatus');
const pipelineModeSelect = document.getElementById('pipelineModeSelect');
const pipelineModeEffective = document.getElementById('pipelineModeEffective');
const pipelineReportStatus = document.getElementById('pipelineReportStatus');
const pipelineHelperStatus = document.getElementById('pipelineHelperStatus');
const densityStatusValue = document.getElementById('densityStatusValue');
const densityHeatmapStatus = document.getElementById('densityHeatmapStatus');
const videoHistoryList = document.getElementById('videoHistoryList');
const historyChartShell = document.getElementById('historyChartShell');
const historyChart = document.getElementById('historyChart');
const historyCtx = historyChart ? historyChart.getContext('2d') : null;
const yoloInternalValue = document.getElementById('yoloInternalValue');
const yoloInternalTotalValue = document.getElementById('yoloInternalTotalValue');
const yoloCropValue = document.getElementById('yoloCropValue');
const yoloFuseValue = document.getElementById('yoloFuseValue');
const overlayComposeValue = document.getElementById('overlayComposeValue');
const overlayPreviewValue = document.getElementById('overlayPreviewValue');
const overlayCpuPreviewValue = document.getElementById('overlayCpuPreviewValue');
const overlayDrawGpuValue = document.getElementById('overlayDrawGpuValue');
const overlayDrawKernelValue = document.getElementById('overlayDrawKernelValue');
const overlayDrawBlendValue = document.getElementById('overlayDrawBlendValue');
const overlayDrawConvertValue = document.getElementById('overlayDrawConvertValue');
const overlayDrawReturnedValue = document.getElementById('overlayDrawReturnedValue');
const overlayRendererFallbackValue = document.getElementById('overlayRendererFallbackValue');
const graphEmpty = document.getElementById('graphEmpty');
const maskLatencyValue = document.getElementById('maskLatencyValue');
const maskTimingNote = document.getElementById('maskTimingNote');
const maskTimingExtra = document.getElementById('maskTimingExtra');
const maskTimingTotal = document.getElementById('maskTimingTotal');
const maskOverlay = document.getElementById('maskOverlay');
const maskToggle = document.getElementById('maskToggle');
const densityToggle = document.getElementById('densityToggle');
const densityTileCountValue = document.getElementById('densityTileCount');
const videoStream = document.getElementById('videoStream');
const maskCtx = maskOverlay ? maskOverlay.getContext('2d') : null;
const maskLatencyChart = document.getElementById('maskLatencyChart');
const maskLatencyCtx = maskLatencyChart ? maskLatencyChart.getContext('2d') : null;
const metricsInterval = 1100;
const METRICS_MIN_INTERVAL = 100;
let metricsTimer = null;
const MAX_LOGS = 6;
let maskVisible = true;
let lastMaskPayload = null;
let lastDensityPayload = null;
let densityOverlayEnabled = false;
let maskRenderToken = 0;
let lastMaskTiming = { created: null, sent: null, latency: null };
let maskReceivedIso = null;
let maskDisplayedIso = null;
let lastMetricsSnapshot = null;
const maskLatencyHistory = [];
const MAX_MASK_LATENCY_SAMPLES = 36;
const FPS_TARGET_HIGH_MS = 1000 / 30;
const FPS_TARGET_LOW_MS = 1000 / 25;

function updateMaskButton(enabled) {
  if (!maskToggle) {
    return;
  }
  const text = enabled ? 'Hide mask overlay' : 'Show mask overlay';
  maskToggle.textContent = text;
  maskToggle.dataset.state = enabled ? 'on' : 'off';
}

function isDensityPayloadAvailable() {
  return Boolean(lastDensityPayload && Array.isArray(lastDensityPayload.tiles) && lastDensityPayload.tiles.length);
}

function isDisplayingDensityOverlay() {
  return densityOverlayEnabled && isDensityPayloadAvailable();
}

function getActiveMaskPayload() {
  if (isDisplayingDensityOverlay()) {
    return lastDensityPayload;
  }
  return lastMaskPayload || lastDensityPayload;
}

function updateDensityButton(enabled) {
  if (!densityToggle) {
    return;
  }
  const label = enabled ? 'Show YOLO overlay' : 'Show density overlay';
  densityToggle.textContent = label;
  densityToggle.dataset.state = enabled ? 'on' : 'off';
}

function applyActiveMaskTiming(data) {
  if (!data) {
    return;
  }
  if (isDisplayingDensityOverlay()) {
    lastMaskTiming = {
      created: data.density_heatmap_payload_created_at || null,
      sent: data.density_heatmap_payload_sent_at || null,
      latency: typeof data.density_heatmap_payload_latency_ms === 'number' ? data.density_heatmap_payload_latency_ms : null,
    };
  } else {
    lastMaskTiming = {
      created: data.yolo_mask_payload_created_at || null,
      sent: data.yolo_mask_payload_sent_at || null,
      latency: typeof data.yolo_mask_payload_latency_ms === 'number' ? data.yolo_mask_payload_latency_ms : null,
    };
  }
}

function computeMetricsInterval(fps) {
  if (typeof fps !== 'number' || !Number.isFinite(fps) || fps <= 0) {
    return metricsInterval;
  }
  const interval = 1000 / fps;
  return Math.max(METRICS_MIN_INTERVAL, Math.min(metricsInterval, interval));
}

function scheduleMetricsRefresh(interval) {
  if (metricsTimer) {
    clearTimeout(metricsTimer);
  }
  metricsTimer = window.setTimeout(refreshMetrics, interval);
}

function formatTimestamp(value) {
  if (!value) {
    return '—';
  }
  try {
    const parsed = Date.parse(value);
    if (Number.isNaN(parsed)) {
      return value;
    }
    const date = new Date(parsed);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch (error) {
    return value;
  }
}

function formatLatencyValue(value) {
  return (typeof value === 'number' && Number.isFinite(value)) ? `${value.toFixed(1)}ms` : '—';
}

function updateMaskTimingCard() {
  if (maskLatencyValue) {
    maskLatencyValue.textContent = formatLatencyValue(lastMaskTiming.latency);
  }
  if (maskTimingNote) {
    const created = formatTimestamp(lastMaskTiming.created);
    const sent = formatTimestamp(lastMaskTiming.sent);
    maskTimingNote.textContent = `created: ${created} · sent: ${sent}`;
  }
  if (maskTimingExtra) {
    const received = formatTimestamp(maskReceivedIso);
    const displayed = formatTimestamp(maskDisplayedIso);
    maskTimingExtra.textContent = `received: ${received} · displayed: ${displayed}`;
  }
  if (maskTimingTotal) {
    const totalMs = computeDisplayLatency();
    maskTimingTotal.textContent = `total latency: ${formatLatencyValue(totalMs)}`;
  }
}

function computeDisplayLatency() {
  const createdTs = lastMaskTiming.created ? Date.parse(lastMaskTiming.created) : NaN;
  const displayedTs = maskDisplayedIso ? Date.parse(maskDisplayedIso) : NaN;
  if (!Number.isFinite(createdTs) || !Number.isFinite(displayedTs)) {
    return null;
  }
  return displayedTs - createdTs;
}

function recordMaskLatencySample() {
  const createdTs = lastMaskTiming.created ? Date.parse(lastMaskTiming.created) : NaN;
  const sentTs = lastMaskTiming.sent ? Date.parse(lastMaskTiming.sent) : NaN;
  const receivedTs = maskReceivedIso ? Date.parse(maskReceivedIso) : NaN;
  const displayedTs = maskDisplayedIso ? Date.parse(maskDisplayedIso) : NaN;
  if (!Number.isFinite(createdTs) || !Number.isFinite(sentTs)) {
    return;
  }
  const sendLatency = Math.max(0, sentTs - createdTs);
  const backendLatency = (typeof lastMaskTiming.latency === 'number' && Number.isFinite(lastMaskTiming.latency))
    ? lastMaskTiming.latency
    : sendLatency;
  const renderLatency = Number.isFinite(displayedTs) && Number.isFinite(receivedTs)
    ? Math.max(0, displayedTs - receivedTs)
    : null;
  const totalLatency = Number.isFinite(displayedTs)
    ? Math.max(0, displayedTs - createdTs)
    : null;
  maskLatencyHistory.push({
    backend: backendLatency,
    render: renderLatency,
    total: totalLatency,
    payload: sendLatency,
  });
  if (maskLatencyHistory.length > MAX_MASK_LATENCY_SAMPLES) {
    maskLatencyHistory.shift();
  }
  drawMaskLatencyGraph();
}

function drawMaskLatencyGraph() {
  if (!maskLatencyChart || !maskLatencyCtx) {
    return;
  }
  const bounds = maskLatencyChart.getBoundingClientRect();
  const width = bounds.width;
  const height = bounds.height;
  if (width <= 0 || height <= 0) {
    return;
  }
  const dpr = window.devicePixelRatio || 1;
  maskLatencyChart.width = width * dpr;
  maskLatencyChart.height = height * dpr;
  maskLatencyCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  maskLatencyCtx.clearRect(0, 0, width, height);
  if (!maskLatencyHistory.length) {
    maskLatencyCtx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    maskLatencyCtx.fillRect(0, 0, width, height);
    maskLatencyCtx.fillStyle = 'rgba(255, 255, 255, 0.35)';
    maskLatencyCtx.font = '0.8rem "Space Grotesk", "Poppins", sans-serif';
    maskLatencyCtx.textAlign = 'center';
    maskLatencyCtx.fillText('Waiting for mask timeline…', width / 2, height / 2 + 4);
    return;
  }
  const accentColor = (getComputedStyle(document.documentElement).getPropertyValue('--accent') || '#2adfa5').trim() || '#2adfa5';
  const renderColor = 'rgba(6, 189, 255, 0.95)';
  const flattened = maskLatencyHistory.flatMap((sample) => [
    Number(sample.backend) || 0,
    Number(sample.render) || 0,
    Number(sample.total) || 0,
  ]);
  const maxSample = Math.max(...flattened, FPS_TARGET_LOW_MS, FPS_TARGET_HIGH_MS);
  const scaleMax = Math.max(maxSample * 1.1, FPS_TARGET_LOW_MS + 10, 12);
  const steps = Math.max(maskLatencyHistory.length - 1, 1);
  const stepX = width / steps;
  const clampY = (value) => height - ((value / scaleMax) * (height - 24)) - 12;
  maskLatencyCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  maskLatencyCtx.lineWidth = 1;
  for (let idx = 1; idx < 4; idx += 1) {
    const y = (height / 4) * idx;
    maskLatencyCtx.beginPath();
    maskLatencyCtx.moveTo(0, y);
    maskLatencyCtx.lineTo(width, y);
    maskLatencyCtx.stroke();
  }
  const targetLines = [
    { value: FPS_TARGET_HIGH_MS, label: '30 fps', color: 'rgba(255, 255, 255, 0.45)' },
    { value: FPS_TARGET_LOW_MS, label: '25 fps', color: 'rgba(255, 255, 255, 0.25)' },
  ];
  maskLatencyCtx.save();
  maskLatencyCtx.setLineDash([5, 4]);
  maskLatencyCtx.lineWidth = 1;
  targetLines.forEach(({ value, label, color }) => {
    const y = clampY(value);
    maskLatencyCtx.strokeStyle = color;
    maskLatencyCtx.beginPath();
    maskLatencyCtx.moveTo(0, y);
    maskLatencyCtx.lineTo(width, y);
    maskLatencyCtx.stroke();
    maskLatencyCtx.fillStyle = color;
    maskLatencyCtx.textAlign = 'right';
    maskLatencyCtx.textBaseline = 'bottom';
    maskLatencyCtx.font = '0.7rem "Space Grotesk", sans-serif';
    maskLatencyCtx.fillText(label, width - 6, y - 4);
  });
  maskLatencyCtx.restore();
  const drawSeries = (key, color, lineWidth) => {
    let started = false;
    maskLatencyCtx.beginPath();
    maskLatencyCtx.lineWidth = lineWidth;
    maskLatencyCtx.strokeStyle = color;
    maskLatencyHistory.forEach((sample, idx) => {
      const value = Number(sample[key]);
      if (!Number.isFinite(value)) {
        started = false;
        return;
      }
      const x = idx * stepX;
      const y = clampY(value);
      if (!started) {
        maskLatencyCtx.moveTo(x, y);
        started = true;
        return;
      }
      maskLatencyCtx.lineTo(x, y);
    });
    if (started) {
      maskLatencyCtx.stroke();
    }
  };
  drawSeries('backend', accentColor, 2.4);
  drawSeries('render', renderColor, 1.6);
}

function getMaskDisplayGeometry(payload) {
  if (!videoWrapper) {
    return null;
  }
  const wrapperRect = videoWrapper.getBoundingClientRect();
  if (wrapperRect.width <= 0 || wrapperRect.height <= 0) {
    return null;
  }
  let displayWidth = wrapperRect.width;
  let displayHeight = wrapperRect.height;
  let offsetX = 0;
  let offsetY = 0;
  const frameWidth = payload && payload.frame_width
    ? payload.frame_width
    : (payload && payload.width && payload.scale_x ? payload.width * payload.scale_x : null);
  const frameHeight = payload && payload.frame_height
    ? payload.frame_height
    : (payload && payload.height && payload.scale_y ? payload.height * payload.scale_y : null);
  if (frameWidth > 0 && frameHeight > 0) {
    const frameRatio = frameWidth / frameHeight;
    const wrapperRatio = wrapperRect.width / wrapperRect.height;
    if (frameRatio > wrapperRatio) {
      displayWidth = wrapperRect.width;
      displayHeight = wrapperRect.width / frameRatio;
    } else {
      displayHeight = wrapperRect.height;
      displayWidth = wrapperRect.height * frameRatio;
    }
    offsetX = (wrapperRect.width - displayWidth) / 2;
    offsetY = (wrapperRect.height - displayHeight) / 2;
  }
  return {
    width: Math.max(1, displayWidth),
    height: Math.max(1, displayHeight),
    left: Math.max(0, offsetX),
    top: Math.max(0, offsetY),
  };
}

function syncMaskCanvasSize() {
  if (!maskOverlay || !videoStream) {
    return;
  }
  const fallbackWidth = Math.max(1, Math.floor(videoStream.clientWidth || 0));
  const fallbackHeight = Math.max(1, Math.floor(videoStream.clientHeight || 0));
  const geometry = getMaskDisplayGeometry(getActiveMaskPayload()) || {
    width: fallbackWidth,
    height: fallbackHeight,
    left: 0,
    top: 0,
  };
  const canvasWidth = Math.max(1, Math.round(geometry.width));
  const canvasHeight = Math.max(1, Math.round(geometry.height));
  maskOverlay.width = canvasWidth;
  maskOverlay.height = canvasHeight;
  maskOverlay.style.width = `${geometry.width}px`;
  maskOverlay.style.height = `${geometry.height}px`;
  maskOverlay.style.left = `${geometry.left}px`;
  maskOverlay.style.top = `${geometry.top}px`;
}

function renderMaskOverlay() {
  if (!maskCtx || !maskOverlay) {
    return;
  }
  maskCtx.clearRect(0, 0, maskOverlay.width, maskOverlay.height);
  if (!maskVisible) {
    return;
  }
  syncMaskCanvasSize();
  const payload = getActiveMaskPayload();
  if (!payload) {
    return;
  }
  if (isDisplayingDensityOverlay()) {
    renderDensityOverlay(payload);
  } else {
    renderYoloOverlay(payload);
  }
}

function renderYoloOverlay(payload) {
  if (!payload || !payload.blob || !maskOverlay) {
    return;
  }
  const width = maskOverlay.width;
  const height = maskOverlay.height;
  if (!width || !height) {
    return;
  }
  const token = ++maskRenderToken;
  const img = new Image();
  img.onload = () => {
    if (token !== maskRenderToken) {
      return;
    }
    maskCtx.clearRect(0, 0, width, height);
    maskCtx.save();
    maskCtx.imageSmoothingEnabled = false;
    maskCtx.webkitImageSmoothingEnabled = false;
    maskCtx.mozImageSmoothingEnabled = false;
    maskCtx.globalAlpha = 0.55;
    maskCtx.drawImage(img, 0, 0, width, height);
    maskCtx.globalCompositeOperation = 'source-in';
    maskCtx.fillStyle = 'rgba(255, 95, 66, 0.85)';
    maskCtx.fillRect(0, 0, width, height);
    maskCtx.restore();
    maskDisplayedIso = new Date().toISOString();
    recordMaskLatencySample();
    updateMaskTimingCard();
  };
  img.onerror = () => {
    if (token === maskRenderToken) {
      maskCtx.clearRect(0, 0, width, height);
    }
  };
  const format = payload.format || 'png';
  img.src = `data:image/${format};base64,${payload.blob}`;
}

function renderDensityOverlay(payload) {
  if (!payload || !Array.isArray(payload.tiles) || !payload.tiles.length || !maskOverlay) {
    return;
  }
  const width = maskOverlay.width;
  const height = maskOverlay.height;
  if (!width || !height) {
    return;
  }
  const frameWidth = payload.frame_width || 0;
  const frameHeight = payload.frame_height || 0;
  const scaleX = frameWidth > 0 ? width / frameWidth : 1;
  const scaleY = frameHeight > 0 ? height / frameHeight : 1;
  const tiles = payload.tiles;
  const totalTiles = tiles.length;
  const token = ++maskRenderToken;
  let completed = 0;
  const finalize = () => {
    if (token !== maskRenderToken) {
      return;
    }
    maskDisplayedIso = new Date().toISOString();
    recordMaskLatencySample();
    updateMaskTimingCard();
  };
  const handleTileComplete = () => {
    completed += 1;
    if (completed >= totalTiles) {
      finalize();
    }
  };
  maskCtx.clearRect(0, 0, width, height);
  tiles.forEach((tile) => {
    const blob = tile.blob;
    if (!blob) {
      handleTileComplete();
      return;
    }
    const img = new Image();
    img.onload = () => {
      if (token !== maskRenderToken) {
        return;
      }
      const coords = Array.isArray(tile.coords) ? tile.coords : [0, 0, frameWidth, frameHeight];
      const left = coords[0] || 0;
      const top = coords[1] || 0;
      const right = coords[2] || frameWidth;
      const bottom = coords[3] || frameHeight;
      const drawX = Math.max(0, left * scaleX);
      const drawY = Math.max(0, top * scaleY);
      const tileW = Math.max(0, (right - left) * scaleX);
      const tileH = Math.max(0, (bottom - top) * scaleY);
      if (tileW <= 0 || tileH <= 0) {
        handleTileComplete();
        return;
      }
      maskCtx.save();
      maskCtx.imageSmoothingEnabled = false;
      maskCtx.webkitImageSmoothingEnabled = false;
      maskCtx.mozImageSmoothingEnabled = false;
      maskCtx.globalAlpha = 0.55;
      maskCtx.drawImage(img, drawX, drawY, tileW, tileH);
      maskCtx.globalCompositeOperation = 'source-in';
      maskCtx.fillStyle = 'rgba(255, 95, 66, 0.85)';
      maskCtx.fillRect(drawX, drawY, tileW, tileH);
      maskCtx.restore();
      handleTileComplete();
    };
    img.onerror = () => {
      if (token !== maskRenderToken) {
        return;
      }
      handleTileComplete();
    };
    const format = tile.format || 'png';
    img.src = `data:image/${format};base64,${blob}`;
  });
}

function sendControl(payload) {
  fetch(baseUrl + '/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).catch(console.error);
}

function setProfileButtons(active) {
  profileStart.disabled = active;
  profileStop.disabled = !active;
  profileState.textContent = active ? 'Profiling active' : 'Idle';
}

function formatStage(prefix, value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return `${prefix} ${value.toFixed(1)}ms`;
  }
  return null;
}

function formatMsValue(value) {
  return (typeof value === 'number' && Number.isFinite(value)) ? `${value.toFixed(1)}ms` : '—';
}

function updateOverlayButton(enabled) {
  const label = enabled ? 'Hide metric chart' : 'Show metric chart';
  overlayToggle.textContent = label;
  overlayToggle.dataset.state = enabled ? 'on' : 'off';
  overlayStatus.textContent = `Overlay: ${enabled ? 'on' : 'off'}`;
  if (historyChartShell) {
    historyChartShell.style.display = enabled ? '' : 'none';
  }
}

function updateVideoHistory(records = []) {
  const recent = (records || []).slice(-4).reverse();
  if (!recent.length) {
    videoHistoryList.innerHTML = '<p class="history-empty">Waiting for data...</p>';
    return;
  }
  videoHistoryList.innerHTML = '';
  recent.forEach((record) => {
    const entry = document.createElement('div');
    entry.className = 'history-entry';
    const description = document.createElement('span');
    const timestamp = document.createElement('span');
    const yoloVal = record.yolo_count !== undefined ? Number(record.yolo_count).toFixed(0) : '—';
    const densityVal = record.density_count !== undefined ? Number(record.density_count).toFixed(1) : '—';
    const avgVal = record.average !== undefined ? Number(record.average).toFixed(1) : '—';
    description.textContent = `YOLO ${yoloVal} · Dens ${densityVal} · Avg ${avgVal}`;
    timestamp.textContent = new Date(record.timestamp || Date.now()).toLocaleTimeString();
    entry.append(description, timestamp);
    videoHistoryList.appendChild(entry);
  });
}

function drawHistoryChart(records = []) {
  if (!historyChart || !historyCtx) {
    return;
  }
  const bounds = historyChartShell.getBoundingClientRect();
  const width = bounds.width;
  const height = bounds.height;
  const dpr = window.devicePixelRatio || 1;
  if (width <= 0 || height <= 0) {
    return;
  }
  historyChart.width = width * dpr;
  historyChart.height = height * dpr;
  historyCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  historyCtx.clearRect(0, 0, width, height);
  if (!records.length) {
    if (graphEmpty) {
      graphEmpty.style.display = 'block';
    }
    historyCtx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    historyCtx.fillRect(0, 0, width, height);
    return;
  }
  if (graphEmpty) {
    graphEmpty.style.display = 'none';
  }
  const flattened = records.flatMap((row) => [Number(row.yolo_count || 0), Number(row.density_count || 0), Number(row.average || 0)]);
  const maxVal = Math.max(5, ...flattened);
  const maxPoints = Math.max(records.length - 1, 1);
  const stepX = width / maxPoints;
  const clampY = (value) => height - ((value / maxVal) * (height - 24)) - 12;
  const drawLine = (values, color, thickness) => {
    historyCtx.beginPath();
    historyCtx.lineWidth = thickness;
    historyCtx.strokeStyle = color;
    values.forEach((val, idx) => {
      const x = idx * stepX;
      const y = clampY(val);
      if (idx === 0) {
        historyCtx.moveTo(x, y);
      } else {
        historyCtx.lineTo(x, y);
      }
    });
    historyCtx.stroke();
  };
  const yoloValues = records.map((entry) => Number(entry.yolo_count || 0));
  const densityValues = records.map((entry) => Number(entry.density_count || 0));
  const avgValues = records.map((entry) => Number(entry.average || 0));
  // Draw light grid
  historyCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  historyCtx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (height / 4) * i;
    historyCtx.beginPath();
    historyCtx.moveTo(0, y);
    historyCtx.lineTo(width, y);
    historyCtx.stroke();
  }
  drawLine(avgValues, 'rgba(255, 255, 0, 0.95)', 2.4);
  drawLine(densityValues, 'rgba(0, 0, 255, 0.8)', 1.5);
  drawLine(yoloValues, 'rgba(0, 255, 0, 0.8)', 1.5);
}

debugToggle.addEventListener('change', () => sendControl({ debug: debugToggle.checked }));
profileStart.addEventListener('click', () => {
  sendControl({ profile_action: 'start' });
  setProfileButtons(true);
});
profileStop.addEventListener('click', () => {
  sendControl({ profile_action: 'stop' });
  setProfileButtons(false);
});
profileClear.addEventListener('click', () => sendControl({ profile_action: 'clear' }));
profileViewSelect.addEventListener('change', (event) => {
  const value = event.target.value;
  profileViewBadge.textContent = value.replace('_', ' ');
  sendControl({ profile_name: value });
});
overlayToggle.addEventListener('click', () => {
  const nextState = overlayToggle.dataset.state !== 'on';
  updateOverlayButton(nextState);
  sendControl({ overlay: nextState });
});
if (maskToggle) {
  maskToggle.addEventListener('click', () => {
    maskVisible = !maskVisible;
    updateMaskButton(maskVisible);
    renderMaskOverlay();
  });
}
if (densityToggle) {
  densityToggle.addEventListener('click', () => {
    densityOverlayEnabled = !densityOverlayEnabled;
    updateDensityButton(densityOverlayEnabled);
    if (lastMetricsSnapshot) {
      applyActiveMaskTiming(lastMetricsSnapshot);
    }
    renderMaskOverlay();
    updateMaskTimingCard();
  });
}
if (pipelineModeSelect) {
  pipelineModeSelect.addEventListener('change', (event) => {
    sendControl({ pipeline_mode: event.target.value });
  });
}
updateOverlayButton(overlayToggle.dataset.state === 'on');
updateMaskButton(maskVisible);
updateDensityButton(densityOverlayEnabled);
if (videoStream) {
  videoStream.addEventListener('load', () => {
    syncMaskCanvasSize();
    renderMaskOverlay();
  });
}
window.addEventListener('resize', () => {
  syncMaskCanvasSize();
  renderMaskOverlay();
  drawMaskLatencyGraph();
});

async function refreshMetrics() {
  try {
    const response = await fetch(baseUrl + '/api/metrics', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Unable to reach metrics');
    }
    const data = await response.json();
    document.getElementById('yoloCountValue').textContent = data.yolo_count !== undefined ? data.yolo_count.toFixed(0) : '—';
    document.getElementById('densityCountValue').textContent = data.density_count !== undefined ? data.density_count.toFixed(1) : '—';
    document.getElementById('avgValue').textContent = data.average !== undefined ? data.average.toFixed(1) : '—';
    fpsValue.textContent = data.fps !== undefined ? data.fps.toFixed(1) : '—';
    document.getElementById('cpuValue').textContent = data.cpu_usage !== undefined ? data.cpu_usage.toFixed(1) + '%' : '—';
    document.getElementById('yoloTimeValue').textContent = data.yolo_time !== undefined ? (data.yolo_time * 1000).toFixed(1) + 'ms' : '—';
    document.getElementById('densityTimeValue').textContent = data.density_time !== undefined ? (data.density_time * 1000).toFixed(1) + 'ms' : '—';
    document.getElementById('totalTimeValue').textContent = data.total_time !== undefined ? (data.total_time * 1000).toFixed(1) + 'ms' : '—';
    document.getElementById('yoloDeviceLabel').textContent = 'Device: ' + (data.yolo_device || '—');
    const densityDeviceEl = document.getElementById('densityDeviceLabel');
    if (densityDeviceEl) {
      densityDeviceEl.textContent = data.density_device || '—';
    }
    if (densityStatusValue) {
      const statusText = typeof data.density_enabled === 'boolean'
        ? (data.density_enabled ? 'enabled' : 'disabled')
        : '—';
      densityStatusValue.textContent = statusText;
    }
    if (densityHeatmapStatus) {
      densityHeatmapStatus.textContent = data.density_ready ? 'ready' : 'inactive';
    }
    if (densityTileCountValue) {
      const tileCount = Number.isFinite(data.density_tile_count) ? data.density_tile_count : null;
      densityTileCountValue.textContent = tileCount !== null ? tileCount : '—';
    }
    fpsBadge.textContent = 'FPS ' + (data.fps !== undefined ? data.fps.toFixed(1) : '--');
    fpsStatus.textContent = `FPS ${data.fps !== undefined ? data.fps.toFixed(1) : '--'} · ${data.profile_active ? 'Profiling' : 'Live'}`;
    debugToggle.checked = Boolean(data.debug_mode);
    setProfileButtons(Boolean(data.profile_active));
    updateOverlayButton(Boolean(data.graph_overlay));
    updateVideoHistory(data.history || []);
    drawHistoryChart(data.history || []);
    const activeProfileName = data.profile_name || data.active_profile_view;
    if (activeProfileName) {
      profileViewBadge.textContent = activeProfileName.replace('_', ' ');
      const matchingOption = [...profileViewSelect.options].find((opt) => opt.value === activeProfileName);
      if (matchingOption) {
        profileViewSelect.value = activeProfileName;
      }
    }
    if (pipelineModeSelect && typeof data.yolo_pipeline_mode === 'string') {
      pipelineModeSelect.value = data.yolo_pipeline_mode;
    }
    const pipelineReport = data.yolo_pipeline_report || {};
    if (pipelineModeEffective) {
      const effective = data.yolo_pipeline_mode_effective || data.yolo_pipeline_mode;
      const displayMap = {
        auto: 'Auto (preferred)',
        gpu_full: 'Full GPU pipeline',
        gpu: 'GPU preprocess + render',
        cpu: 'CPU fallback',
      };
      const label = effective ? displayMap[effective] || effective.replace('_', ' ') : '—';
      const fallbackNote = pipelineReport.last_gpu_full_success === false ? ' (fallbacked to CPU tiler)' :
        (pipelineReport.last_gpu_full_success === true ? ' (GPU tiler ran)' : '');
      pipelineModeEffective.textContent = `Effective pipeline: ${label}${fallbackNote}`;
    }
    if (pipelineReportStatus) {
      const statusParts = [];
      if (pipelineReport.active_mode) {
        statusParts.push(pipelineReport.active_mode === 'gpu_full' ? 'gpu_full active' : pipelineReport.active_mode);
      }
      if (pipelineReport.last_gpu_full_success === true) {
        statusParts.push('GPU tiler succeeded');
      } else if (pipelineReport.last_gpu_full_success === false) {
        const errMsg = pipelineReport.last_gpu_full_error ? `error: ${pipelineReport.last_gpu_full_error}` : 'error';
        statusParts.push(`GPU tiler failed (${errMsg})`);
      }
      if (!statusParts.length) {
        statusParts.push('Awaiting GPU pipeline data…');
      }
      pipelineReportStatus.textContent = `Pipeline status: ${statusParts.join(' · ')}`;
    }
    if (pipelineHelperStatus) {
      const helperState = pipelineReport.helpers_available;
      const helperText = helperState === true ? 'available' : helperState === false ? 'missing' : 'unknown';
      pipelineHelperStatus.textContent = `CUDA helpers: ${helperText}`;
    }
    const captureMs = data.capture_ms;
    captureTimeLabel.textContent = `Capture: ${typeof captureMs === 'number' ? captureMs.toFixed(1) : '—'}ms`;
    const yoloStages = [
      formatStage('pre', data.yolo_pre_ms),
      formatStage('gpu', data.yolo_gpu_ms),
      formatStage('post', data.yolo_post_ms)
    ].filter(Boolean).join(' · ');
    const densityStages = [
      formatStage('pre', data.density_pre_ms),
      formatStage('gpu', data.density_gpu_ms),
      formatStage('post', data.density_post_ms)
    ].filter(Boolean).join(' · ');
    const pipelinePre = formatStage('Pipeline pre', data.pipeline_pre_ms);
    const pipelinePost = formatStage('Pipeline post', data.pipeline_post_ms);
    const pipelineLine = [pipelinePre, pipelinePost].filter(Boolean).join(' · ');
    perfSummary.innerHTML = `
      <div class="perf-line">${pipelineLine || 'Pipeline —'}</div>
      <div class="perf-line">YOLO ${yoloStages || '—'}</div>
      <div class="perf-line">DENS ${densityStages || '—'}</div>
    `;
    if (yoloInternalValue) {
      const infLabel = formatMsValue(data.yolo_internal_inf_ms);
      const drawLabel = formatMsValue(data.yolo_internal_draw_ms);
      yoloInternalValue.textContent = `Inf: ${infLabel} · Draw: ${drawLabel}`;
    }
    if (yoloInternalTotalValue) {
      yoloInternalTotalValue.textContent = formatMsValue(data.yolo_internal_total_ms);
    }
    if (yoloCropValue) {
      yoloCropValue.textContent = formatMsValue(data.yolo_internal_crop_ms);
    }
    if (yoloFuseValue) {
      yoloFuseValue.textContent = formatMsValue(data.yolo_internal_fuse_ms);
    }
    if (overlayComposeValue) {
      overlayComposeValue.textContent = formatMsValue(data.overlay_compose_ms);
    }
    if (overlayDrawGpuValue) {
      overlayDrawGpuValue.textContent = formatMsValue(data.overlay_draw_total_ms);
    }
    if (overlayDrawKernelValue) {
      overlayDrawKernelValue.textContent = formatMsValue(data.overlay_draw_kernel_ms);
    }
    if (overlayDrawBlendValue) {
      overlayDrawBlendValue.textContent = formatMsValue(data.overlay_draw_blend_ms);
    }
    if (overlayDrawConvertValue) {
      overlayDrawConvertValue.textContent = formatMsValue(data.overlay_draw_convert_ms);
    }
    if (overlayDrawReturnedValue) {
      const tensorFlag = data.overlay_draw_returned_tensor ? 'yes' : 'no';
      overlayDrawReturnedValue.textContent = tensorFlag;
    }
    if (overlayRendererFallbackValue) {
      overlayRendererFallbackValue.textContent = data.overlay_renderer_fallback || '—';
    }
    if (overlayPreviewValue) {
      overlayPreviewValue.textContent = formatMsValue(data.overlay_preview_ms);
    }
    if (overlayCpuPreviewValue) {
      overlayCpuPreviewValue.textContent = formatMsValue(data.overlay_cpu_preview_ms);
    }
    updateProfileLog(data.profile_log || []);
    lastMaskPayload = data.yolo_mask_payload || null;
    lastDensityPayload = data.density_heatmap_payload || null;
    lastMetricsSnapshot = data;
    applyActiveMaskTiming(data);
    maskReceivedIso = new Date().toISOString();
    maskDisplayedIso = null;
    renderMaskOverlay();
    const maskLatency = lastMaskTiming.latency;
    if (typeof maskLatency === 'number') {
      console.debug(`[MASK TIMING] created=${lastMaskTiming.created || '—'} sent=${lastMaskTiming.sent || '—'} latency=${maskLatency.toFixed(1)}ms`);
    }
    updateMaskTimingCard();
    const nextInterval = computeMetricsInterval(data.fps);
    scheduleMetricsRefresh(nextInterval);
  } catch (error) {
    fpsStatus.textContent = 'Waiting for metrics...';
    console.warn('Metrics fetch failed', error);
    scheduleMetricsRefresh(metricsInterval);
  }
}

function updateProfileLog(entries) {
  if (!entries.length) {
    profileLog.innerHTML = '<p class="log-empty">Profiles will appear here when profiling is active.</p>';
    return;
  }
  profileLog.innerHTML = '';
  entries.slice(-MAX_LOGS).reverse().forEach((record) => {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const label = document.createElement('span');
    const time = new Date(record.timestamp || Date.now()).toLocaleTimeString();
    label.className = 'log-time';
    label.textContent = time;
    const metrics = document.createElement('span');
    const yoloVal = Number(record.yolo_count || 0);
    const densityVal = Number(record.density_count || 0);
    const fpsVal = Number(record.fps || 0);
    metrics.textContent = `YOLO ${yoloVal.toFixed(0)} · Dens ${densityVal.toFixed(1)} · FPS ${fpsVal.toFixed(1)}`;
    const profileBadge = document.createElement('span');
    profileBadge.className = 'log-profile';
    profileBadge.textContent = (record.profile_name || 'Live Metrics').replace('_', ' ');
    metrics.append(profileBadge);
    entry.append(label, metrics);
    profileLog.appendChild(entry);
  });
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    videoWrapper.requestFullscreen().catch(() => { });
  } else {
    document.exitFullscreen().catch(() => { });
  }
}

document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);
document.getElementById('fitBtn').addEventListener('click', () => {
  document.exitFullscreen().catch(() => { });
  videoWrapper.scrollIntoView({ behavior: 'smooth' });
});
window.addEventListener('keydown', (event) => {
  if (event.code === 'KeyF') {
    toggleFullscreen();
  }
});
document.addEventListener('DOMContentLoaded', () => {
  syncMaskCanvasSize();
  renderMaskOverlay();
  drawMaskLatencyGraph();
  refreshMetrics();
});

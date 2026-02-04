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
const videoHistoryList = document.getElementById('videoHistoryList');
const historyChartShell = document.getElementById('historyChartShell');
const historyChart = document.getElementById('historyChart');
const historyCtx = historyChart ? historyChart.getContext('2d') : null;
const yoloInternalValue = document.getElementById('yoloInternalValue');
const yoloInternalTotalValue = document.getElementById('yoloInternalTotalValue');
const graphEmpty = document.getElementById('graphEmpty');
const metricsInterval = 1100;
const MAX_LOGS = 6;

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
if (pipelineModeSelect) {
  pipelineModeSelect.addEventListener('change', (event) => {
    sendControl({ pipeline_mode: event.target.value });
  });
}
updateOverlayButton(overlayToggle.dataset.state === 'on');

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
    document.getElementById('densityDeviceLabel').textContent = 'Device: ' + (data.density_device || '—');
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
    if (pipelineModeEffective) {
      const effective = data.yolo_pipeline_mode_effective || data.yolo_pipeline_mode;
      const displayMap = {
        auto: 'Auto (preferred)',
        gpu_full: 'Full GPU pipeline',
        gpu: 'GPU preprocess + render',
        cpu: 'CPU fallback',
      };
      const label = effective ? displayMap[effective] || effective.replace('_', ' ') : '—';
      pipelineModeEffective.textContent = `Effective pipeline: ${label}`;
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
    updateProfileLog(data.profile_log || []);
  } catch (error) {
    fpsStatus.textContent = 'Waiting for metrics...';
    console.warn('Metrics fetch failed', error);
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
  refreshMetrics();
  setInterval(refreshMetrics, metricsInterval);
});

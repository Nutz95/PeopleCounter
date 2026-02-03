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
const videoHistoryList = document.getElementById('videoHistoryList');
const metricsInterval = 1100;
const MAX_LOGS = 6;

function sendControl(payload) {
  fetch(baseUrl + '/api/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
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

function updateOverlayButton(enabled) {
  const label = enabled ? 'Hide overlay graph' : 'Show overlay graph';
  overlayToggle.textContent = label;
  overlayToggle.dataset.state = enabled ? 'on' : 'off';
  overlayStatus.textContent = `Overlay: ${enabled ? 'on' : 'off'}`;
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

debugToggle.addEventListener('change', () => sendControl({debug: debugToggle.checked}));
profileStart.addEventListener('click', () => {
  sendControl({profile_action: 'start'});
  setProfileButtons(true);
});
profileStop.addEventListener('click', () => {
  sendControl({profile_action: 'stop'});
  setProfileButtons(false);
});
profileClear.addEventListener('click', () => sendControl({profile_action: 'clear'}));
profileViewSelect.addEventListener('change', (event) => {
  const value = event.target.value;
  profileViewBadge.textContent = value.replace('_', ' ');
  sendControl({profile_name: value});
});
overlayToggle.addEventListener('click', () => {
  const nextState = overlayToggle.dataset.state !== 'on';
  updateOverlayButton(nextState);
  sendControl({overlay: nextState});
});
updateOverlayButton(overlayToggle.dataset.state === 'on');

async function refreshMetrics() {
  try {
    const response = await fetch(baseUrl + '/api/metrics', {cache: 'no-store'});
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
    const activeProfileName = data.profile_name || data.active_profile_view;
    if (activeProfileName) {
      profileViewBadge.textContent = activeProfileName.replace('_', ' ');
      const matchingOption = [...profileViewSelect.options].find((opt) => opt.value === activeProfileName);
      if (matchingOption) {
        profileViewSelect.value = activeProfileName;
      }
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
    perfSummary.textContent = `YOLO ${yoloStages || '—'} · DENS ${densityStages || '—'}`;
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
    videoWrapper.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);
document.getElementById('fitBtn').addEventListener('click', () => {
  document.exitFullscreen().catch(() => {});
  videoWrapper.scrollIntoView({behavior: 'smooth'});
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

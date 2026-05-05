// ── controls.js ────────────────────────────────────────────────────────────
// UI controls: inference-mode / sync-mode pill selectors, overlay toggles,
// threshold sliders, config loading, fullscreen.
// Depends on: state.js, overlays.js
// ──────────────────────────────────────────────────────────────────────────

// ── Mode label / overlay maps (filled from /api/config) ───────
const _modeLabels   = {};
const _modeOverlays = {};
const _syncModeLabels = {
  async: 'Async (realtime video)',
  sync:  'Sync (video + inference aligned)',
};

// ── Threshold state ────────────────────────────────────────────
let _densityThreshold = 0.05;
const _crowdConfidenceByMode = {};
let _crowdConfidence  = 0.5;

// ── Overlay section visibility ─────────────────────────────────
function _updateOverlaySection(mode) {
  const overlays = _modeOverlays[mode] || [];
  const $section = document.getElementById('overlay-section');
  $section.style.display = overlays.length > 0 ? '' : 'none';
  document.getElementById('overlay-bbox').style.display    = overlays.includes('bbox')    ? '' : 'none';
  document.getElementById('overlay-seg').style.display     = overlays.includes('seg')     ? '' : 'none';
  document.getElementById('overlay-heatmap').style.display = overlays.includes('heatmap') ? '' : 'none';

  document.getElementById('density-threshold-section').style.display =
    overlays.includes('heatmap') ? '' : 'none';
  document.getElementById('crowd-confidence-section').style.display =
    mode.startsWith('crowd') ? '' : 'none';

  showMask = overlays.includes('bbox');
  document.getElementById('mask-toggle').checked = showMask;

  showSeg = overlays.includes('seg');
  document.getElementById('seg-toggle').checked = showSeg;
  if (!showSeg) segCtx.clearRect(0, 0, segCanvas.width, segCanvas.height);

  showHeatmap = overlays.includes('heatmap');
  document.getElementById('heatmap-toggle').checked = showHeatmap;
  if (!showHeatmap) heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);

  const $previewRow = document.getElementById('preview-only-row');
  if ($previewRow) $previewRow.style.display = overlays.length > 0 ? '' : 'none';
}

// ── Inference mode pills ───────────────────────────────────────
function _renderModePills(availableModes, activeMode) {
  const $pills = document.getElementById('mode-pills');
  $pills.innerHTML = '';
  for (const m of availableModes) {
    const btn = document.createElement('button');
    btn.className = 'mode-pill' + (m === activeMode ? ' mode-pill--active' : '');
    btn.dataset.mode = m;
    btn.textContent = _modeLabels[m] || m;
    btn.addEventListener('click', () => _requestModeChange(m));
    $pills.appendChild(btn);
  }
}

function _setActivePill(mode) {
  document.querySelectorAll('#mode-pills .mode-pill').forEach(btn => {
    btn.classList.toggle('mode-pill--active', btn.dataset.mode === mode);
  });
}

// ── Sync mode pills ────────────────────────────────────────────
function _renderSyncModePills(options, activeSyncMode) {
  const $pills = document.getElementById('sync-mode-pills');
  $pills.innerHTML = '';
  for (const m of options) {
    const btn = document.createElement('button');
    btn.className = 'mode-pill' + (m === activeSyncMode ? ' mode-pill--active' : '');
    btn.dataset.syncMode = m;
    btn.textContent = _syncModeLabels[m] || m;
    btn.addEventListener('click', () => _requestSyncModeChange(m));
    $pills.appendChild(btn);
  }
}

function _setActiveSyncPill(mode) {
  document.querySelectorAll('#sync-mode-pills .mode-pill').forEach(btn => {
    btn.classList.toggle('mode-pill--active', btn.dataset.syncMode === mode);
  });
}

// ── Mode change request ────────────────────────────────────────
async function _requestModeChange(newMode) {
  if (newMode === _activeMode || _modeChanging) return;
  _modeChanging = true;
  document.getElementById('mode-loading').style.display = '';
  document.querySelectorAll('#mode-pills .mode-pill').forEach(b => b.disabled = true);

  try {
    const res  = await fetch('/api/mode', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ mode: newMode }),
    });
    const json = await res.json();
    if (json.ok) {
      _activeMode = newMode;
      _setActivePill(newMode);
      _updateOverlaySection(newMode);
      const _nmc = _crowdConfidenceByMode[newMode];
      if (typeof _nmc === 'number') {
        _crowdConfidence = _nmc;
        document.getElementById('crowd-confidence-slider').value = _nmc;
        document.getElementById('crowd-confidence-value').textContent = _nmc.toFixed(2);
      }
      $countValue.textContent  = '—';
      $e2eValue.textContent    = '—';
      $tilesValue.textContent  = '—';
      $globalValue.textContent = '—';
    }
  } catch (err) {
    console.warn('Mode change failed:', err);
  } finally {
    setTimeout(() => {
      _modeChanging = false;
      document.getElementById('mode-loading').style.display = 'none';
      document.querySelectorAll('#mode-pills .mode-pill').forEach(b => b.disabled = false);
    }, 1500);
  }
}

// ── Sync mode change request ───────────────────────────────────
async function _requestSyncModeChange(newSyncMode) {
  if (newSyncMode === _activeSyncMode || _syncModeChanging) return;
  _syncModeChanging = true;
  document.getElementById('sync-mode-loading').style.display = '';
  document.querySelectorAll('#sync-mode-pills .mode-pill').forEach(b => b.disabled = true);

  try {
    const res  = await fetch('/api/sync_mode', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ mode: newSyncMode }),
    });
    const json = await res.json();
    if (json.ok) {
      _activeSyncMode   = newSyncMode;
      window.__syncMode = newSyncMode;
      _setActiveSyncPill(newSyncMode);
      if (typeof window.__pcSetWebCodecsEnabled === 'function') {
        window.__pcSetWebCodecsEnabled(newSyncMode !== 'sync');
      }
    }
  } catch (err) {
    console.warn('Sync mode change failed:', err);
  } finally {
    setTimeout(() => {
      _syncModeChanging = false;
      document.getElementById('sync-mode-loading').style.display = 'none';
      document.querySelectorAll('#sync-mode-pills .mode-pill').forEach(b => b.disabled = false);
    }, 1200);
  }
}

// ── Initial config load ────────────────────────────────────────
async function _loadConfig() {
  try {
    const res = await fetch('/api/config');
    const cfg = await res.json();
    Object.assign(_modeLabels,    cfg.mode_labels    || {});
    Object.assign(_modeOverlays,  cfg.mode_overlays  || {});
    Object.assign(_syncModeLabels, cfg.sync_mode_labels || {});
    _activeMode       = cfg.mode      || 'passthrough';
    _activeSyncMode   = cfg.sync_mode || 'async';
    window.__syncMode = _activeSyncMode;
    _renderModePills(cfg.available_modes   || ['passthrough'], _activeMode);
    _renderSyncModePills(cfg.sync_mode_options || ['async', 'sync'], _activeSyncMode);
    _updateOverlaySection(_activeMode);
    if (typeof window.__pcSetWebCodecsEnabled === 'function') {
      window.__pcSetWebCodecsEnabled(_activeSyncMode !== 'sync');
    }
    if (typeof cfg.density_threshold === 'number') {
      const v = cfg.density_threshold;
      _densityThreshold = v;
      document.getElementById('density-threshold-slider').value = v;
      document.getElementById('density-threshold-value').textContent = v.toFixed(2);
    }
    if (cfg.crowd_confidence_by_mode) {
      Object.assign(_crowdConfidenceByMode, cfg.crowd_confidence_by_mode);
    }
    if (typeof cfg.crowd_confidence === 'number') {
      const v = cfg.crowd_confidence;
      _crowdConfidence = v;
      document.getElementById('crowd-confidence-slider').value = v;
      document.getElementById('crowd-confidence-value').textContent = v.toFixed(2);
    }
  } catch (e) {
    console.error('_loadConfig failed:', e);
    _renderModePills(['passthrough'], 'passthrough');
  }
}
_loadConfig();

// ── Overlay toggle listeners ───────────────────────────────────
document.getElementById('mask-toggle').addEventListener('change', e => {
  showMask = e.target.checked;
  if (!showMask) maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
});

document.getElementById('seg-toggle').addEventListener('change', e => {
  showSeg = e.target.checked;
  if (!showSeg) segCtx.clearRect(0, 0, segCanvas.width, segCanvas.height);
});

document.getElementById('heatmap-toggle').addEventListener('change', e => {
  showHeatmap = e.target.checked;
  if (!showHeatmap) heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
});

document.getElementById('preview-only-toggle').addEventListener('change', e => {
  previewOnly = e.target.checked;
  if (previewOnly) {
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    segCtx.clearRect(0, 0, segCanvas.width, segCanvas.height);
    heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
  }
});

// ── Density peak threshold slider ─────────────────────────────
(function () {
  const slider    = document.getElementById('density-threshold-slider');
  const badge     = document.getElementById('density-threshold-value');
  let _debounce   = null;

  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    _densityThreshold = v;
    badge.textContent = v.toFixed(2);
    clearTimeout(_debounce);
    _debounce = setTimeout(async () => {
      try {
        await fetch('/api/density/threshold', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ threshold: v }),
        });
      } catch (err) {
        console.warn('density threshold update failed:', err);
      }
    }, 150);
  });
})();

// ── Crowd confidence threshold slider ─────────────────────────
(function () {
  const slider  = document.getElementById('crowd-confidence-slider');
  const badge   = document.getElementById('crowd-confidence-value');
  let _debounce = null;

  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    _crowdConfidence = v;
    _crowdConfidenceByMode[_activeMode] = v;
    badge.textContent = v.toFixed(2);
    clearTimeout(_debounce);
    _debounce = setTimeout(async () => {
      try {
        await fetch('/api/crowd/confidence', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ confidence: v }),
        });
      } catch (err) {
        console.warn('crowd confidence update failed:', err);
      }
    }, 150);
  });
})();

// ── Fullscreen toggle ──────────────────────────────────────────
function toggleFullscreen() {
  const el = document.getElementById('video-wrapper');
  if (!document.fullscreenElement) el.requestFullscreen?.();
  else document.exitFullscreen?.();
}

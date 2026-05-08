// ── state.js ───────────────────────────────────────────────────────────────
// Shared constants, application state, DOM references, and rolling-average
// utilities.  Loaded first so all other scripts can read these globals.
// ──────────────────────────────────────────────────────────────────────────

// ── Constants ──────────────────────────────────────────────────
const MAX_HISTORY      = 60;
const TARGET_MS        = 33.33;   // 30 fps
const SMOOTH_WIN       = 2000;    // 2-second rolling-average window (ms)
const COUNT_SMOOTH_WIN = 1000;    // 1-second rolling-average window (ms)

// ── Per-frame history arrays ──────────────────────────────────
const e2eHist   = [];   // end-to-end latency per frame (ms)
const tilesHist = [];   // tiles inference latency per frame (ms)
const fpsTs     = [];   // raw timestamps for SSE-based FPS estimation
const gpuHist   = [];   // GPU utilisation % (1 s samples)
const countHist = [];   // person count (1 s samples for sparkline)

// ── Overlay visibility flags ──────────────────────────────────
let showMask     = true;
let showSeg      = false;
let showHeatmap  = true;
let previewOnly  = false;
let showBboxText = false;
let renderPoints = false;

// ── Inference / sync mode state ───────────────────────────────
let _activeMode       = 'passthrough';
let _modeChanging     = false;
let _activeSyncMode   = 'async';
let _syncModeChanging = false;
window.__syncMode     = 'async';

// ── Display-FPS tracking (WebCodecs rendered frames/s) ────────
const _dispFpsTs    = [];
const $displayFpsChip = document.getElementById('display-fps-chip');

// ── Rolling-average buckets  { t: timestamp_ms, v: value }[] ─
const _smooth = {
  e2e:     [],
  tiles:   [],
  global:  [],
  preproc: [],
  fusion:  [],
  count:   [],
  decode:  [],
  decodeFilter: [],
  decodeNms: [],
  decodeExport: [],
  decodePack: [],
  other:   [],
  encode:  [],
  publish: [],
  nvdec:   [],
};

// Add a sample and return the rolling mean over SMOOTH_WIN.
function rollAvg(arr, value) {
  const now = Date.now();
  arr.push({ t: now, v: value });
  while (arr.length && now - arr[0].t > SMOOTH_WIN) arr.shift();
  if (arr.length === 0) return value;
  return arr.reduce((s, x) => s + x.v, 0) / arr.length;
}

// Same as rollAvg but with an explicit window size (ms).
function rollAvgWin(arr, value, win) {
  const now = Date.now();
  arr.push({ t: now, v: value });
  while (arr.length && now - arr[0].t > win) arr.shift();
  if (arr.length === 0) return value;
  return arr.reduce((s, x) => s + x.v, 0) / arr.length;
}

// Read the current average without adding a new data point.
function peekAvg(arr, win) {
  const now   = Date.now();
  const start = now - win;
  let sum = 0, n = 0;
  for (const e of arr) { if (e.t >= start) { sum += e.v; n++; } }
  return n > 0 ? sum / n : null;
}

// ── DOM references ─────────────────────────────────────────────
const $connStatus    = document.getElementById('conn-status');
const $latencyChip   = document.getElementById('latency-chip');
const $fpsChip       = document.getElementById('fps-chip');
const $countValue    = document.getElementById('count-value');
const $countAvgValue = document.getElementById('count-avg-value');
const $frameLabel    = document.getElementById('frame-id-label');
const $e2eValue      = document.getElementById('e2e-value');
const $fpsDerived    = document.getElementById('fps-derived');
const $preprocValue  = document.getElementById('preprocess-value');
const $tilesValue    = document.getElementById('tiles-value');
const $globalValue   = document.getElementById('global-value');
const $fusionValue   = document.getElementById('fusion-value');
const $ovPreprocess  = document.getElementById('ov-preprocess');
const $ovGlobal      = document.getElementById('ov-global');
const $ovTiles       = document.getElementById('ov-tiles');
const $ovOther       = document.getElementById('ov-other');
const $displayModeChip = document.getElementById('display-mode-chip');
const $overlayChip   = document.getElementById('overlay-chip');
const $metaChip      = document.getElementById('meta-chip');
const $benchmarkToggle = document.getElementById('benchmark-toggle');
const $benchmarkStatus = document.getElementById('benchmark-status');
const $benchmarkReport = document.getElementById('benchmark-report');

const maskCanvas = document.getElementById('mask-canvas');
const maskCtx    = maskCanvas.getContext('2d');
const segCanvas  = document.getElementById('seg-canvas');
const segCtx     = segCanvas.getContext('2d');
const heatCanvas = document.getElementById('heat-canvas');
const heatCtx    = heatCanvas.getContext('2d');
const latChart   = document.getElementById('latency-chart');
const latCtx     = latChart.getContext('2d');

// Source video dimensions received from the server via telemetry.
let _videoSrcW = 0;
let _videoSrcH = 0;

// Latest packed detections received via metadata WebSocket.
// Float32Array rows of [x1, y1, x2, y2, conf].
let _packedDetectionsFrameId = -1;
let _packedDetections = null;
let _packedDetectionsRowWidth = 5;

// ── Transport/renderer status (UI chips under MJPEG/WebCodecs bubble) ───
window.__videoTransport = window.__videoTransport || 'mjpeg';
window.__overlayRenderer = window.__overlayRenderer || 'canvas2d';
window.__metaWsConnected = false;
window.__metaWsPacketRate = 0;

function _updateTransportStatusChips() {
  if ($displayModeChip) {
    const modeText = (window.__videoTransport === 'webcodecs') ? 'WebCodecs' : 'MJPEG';
    $displayModeChip.textContent = modeText;
  }

  if ($overlayChip) {
    const webglActive = !!window._webglOverlayActive;
    $overlayChip.textContent = webglActive ? 'Overlay WebGL2 ✓' : 'Overlay Canvas2D';
    $overlayChip.style.borderColor = webglActive ? 'rgba(42,223,165,0.65)' : 'rgba(255,255,255,0.22)';
  }

  if ($metaChip) {
    const connected = !!window.__metaWsConnected;
    const pps = +(window.__metaWsPacketRate || 0);
    if (connected) {
      $metaChip.textContent = `MetaWS ✓ ${pps.toFixed(0)} pkt/s`;
      $metaChip.style.borderColor = 'rgba(42,223,165,0.65)';
    } else {
      $metaChip.textContent = 'MetaWS … reconnect';
      $metaChip.style.borderColor = 'rgba(245,158,11,0.55)';
    }
  }
}

setInterval(_updateTransportStatusChips, 500);
_updateTransportStatusChips();

// ── 1-second smoothed count display ───────────────────────────
// Updated on a timer instead of per-frame to keep the value stable.
setInterval(() => {
  const avg = peekAvg(_smooth.count, COUNT_SMOOTH_WIN);
  if (avg !== null && $countAvgValue) $countAvgValue.textContent = Math.round(avg);
}, 1000);

// ── Benchmark capture (Start/Stop → CSV download) ────────────────────────
let _benchmarkActive = false;
let _benchmarkRows = [];
let _benchmarkStartEpochMs = 0;
let _benchmarkStartPerfMs = 0;
let _benchmarkLastSummaryText = 'No benchmark summary yet.';

const _benchmarkMetricCols = [
  'e2e_ms',
  'src_wait_ms',
  'preproc_ms',
  'infer_critical_ms',
  'postdecode_ms',
  'other_ms',
  'src_copy_sync_ms',
  'trt_prepare_ms',
  'trt_sync_ms',
  'decode_ms',
  'publish_total_ms',
];

const _benchmarkSpikeComponents = [
  'src_wait_ms',
  'preproc_ms',
  'infer_critical_ms',
  'postdecode_ms',
  'other_ms',
];

function _formatIso(tsMs) {
  try { return new Date(tsMs).toISOString(); } catch { return ''; }
}

function _csvEscape(value) {
  if (value === null || value === undefined) return '';
  const s = String(value);
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function _toFiniteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function _percentile(sortedValues, q) {
  if (!sortedValues.length) return null;
  if (sortedValues.length === 1) return sortedValues[0];
  const pos = (sortedValues.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedValues[lo];
  const w = pos - lo;
  return sortedValues[lo] * (1 - w) + sortedValues[hi] * w;
}

function _fmtMs(value, decimals = 2) {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  return `${value.toFixed(decimals)}ms`;
}

function _computeBenchmarkSummary(rows) {
  const metricStats = {};

  for (const col of _benchmarkMetricCols) {
    const values = rows
      .map((r) => _toFiniteNumber(r[col]))
      .filter((v) => v !== null)
      .sort((a, b) => a - b);

    if (!values.length) {
      metricStats[col] = null;
      continue;
    }

    const sum = values.reduce((acc, v) => acc + v, 0);
    metricStats[col] = {
      n: values.length,
      mean: sum / values.length,
      p50: _percentile(values, 0.50),
      p95: _percentile(values, 0.95),
      p99: _percentile(values, 0.99),
      max: values[values.length - 1],
    };
  }

  const topSpikes = [...rows]
    .map((r) => ({
      row: r,
      e2e: _toFiniteNumber(r.e2e_ms) ?? -Infinity,
    }))
    .filter((x) => Number.isFinite(x.e2e))
    .sort((a, b) => b.e2e - a.e2e)
    .slice(0, 10)
    .map(({ row, e2e }) => {
      let dominant = { key: 'n/a', value: -Infinity };
      for (const col of _benchmarkSpikeComponents) {
        const v = _toFiniteNumber(row[col]);
        if (v !== null && v > dominant.value) {
          dominant = { key: col, value: v };
        }
      }
      return {
        frame_id: row.frame_id,
        e2e_ms: e2e,
        dominant_component: dominant.key,
        dominant_ms: Number.isFinite(dominant.value) ? dominant.value : null,
      };
    });

  return { metricStats, topSpikes, rowCount: rows.length };
}

function _buildBenchmarkSummaryText(summary) {
  if (!summary) return 'No benchmark summary yet.';

  const lines = [];
  lines.push(`Rows: ${summary.rowCount}`);
  lines.push('');
  lines.push('Metric              mean     p50      p95      p99      max');
  lines.push('---------------------------------------------------------------');

  for (const col of _benchmarkMetricCols) {
    const s = summary.metricStats[col];
    if (!s) {
      lines.push(`${col.padEnd(18)} n/a      n/a      n/a      n/a      n/a`);
      continue;
    }
    lines.push(
      `${col.padEnd(18)} ${_fmtMs(s.mean).padEnd(8)} ${_fmtMs(s.p50).padEnd(8)} ${_fmtMs(s.p95).padEnd(8)} ${_fmtMs(s.p99).padEnd(8)} ${_fmtMs(s.max).padEnd(8)}`,
    );
  }

  lines.push('');
  lines.push('Top e2e spikes (dominant component):');
  summary.topSpikes.forEach((spike, i) => {
    lines.push(
      `${String(i + 1).padStart(2, ' ')}. frame ${String(spike.frame_id).padStart(5, ' ')} | e2e=${_fmtMs(spike.e2e_ms)} | ${spike.dominant_component}=${_fmtMs(spike.dominant_ms)}`,
    );
  });

  return lines.join('\n');
}

function _renderBenchmarkSummary(text) {
  if ($benchmarkReport) $benchmarkReport.textContent = text || 'No benchmark summary yet.';
}

function benchmarkCaptureStart() {
  _benchmarkRows = [];
  _benchmarkStartEpochMs = Date.now();
  _benchmarkStartPerfMs = performance.now();
  _benchmarkActive = true;
  _benchmarkLastSummaryText = 'Recording… summary will appear when you stop capture.';
  if ($benchmarkToggle) {
    $benchmarkToggle.textContent = 'Stop capture';
    $benchmarkToggle.classList.add('mode-pill--active');
  }
  if ($benchmarkStatus) $benchmarkStatus.textContent = 'Recording… 0 rows';
  _renderBenchmarkSummary(_benchmarkLastSummaryText);
}

function benchmarkCapturePush(row) {
  if (!_benchmarkActive || !row) return;
  const nowEpoch = Date.now();
  const elapsedMs = performance.now() - _benchmarkStartPerfMs;
  _benchmarkRows.push({
    ts_iso: _formatIso(nowEpoch),
    elapsed_ms: +elapsedMs.toFixed(3),
    ...row,
  });
  if ($benchmarkStatus) $benchmarkStatus.textContent = `Recording… ${_benchmarkRows.length} rows`;
}

function benchmarkCaptureStopAndDownload() {
  _benchmarkActive = false;
  if ($benchmarkToggle) {
    $benchmarkToggle.textContent = 'Start capture';
    $benchmarkToggle.classList.remove('mode-pill--active');
  }

  const rowCount = _benchmarkRows.length;
  if (!rowCount) {
    if ($benchmarkStatus) $benchmarkStatus.textContent = 'Stopped (no rows captured)';
    _benchmarkLastSummaryText = 'No rows captured.';
    _renderBenchmarkSummary(_benchmarkLastSummaryText);
    return;
  }

  const columns = [
    'ts_iso', 'elapsed_ms', 'frame_id', 'mode', 'sync_mode', 'passthrough',
    'count', 'fps', 'e2e_ms', 'preproc_ms', 'infer_tiles_ms', 'infer_global_ms',
    'infer_critical_ms', 'fusion_ms', 'postdecode_ms', 'decode_stage_filter_ms',
    'decode_stage_nms_ms', 'decode_stage_export_ms', 'decode_stage_pack_ms',
    'other_ms', 'nvdec_ms', 'src_wait_ms', 'src_age_ms', 'src_copy_sync_ms',
    'trt_prepare_ms', 'trt_sync_ms', 'decode_ms',
    'video_encode_ms', 'video_encode_wait_ms', 'video_encode_copy_ms', 'video_encode_push_ms',
    'publish_total_ms'
  ];

  const lines = [columns.join(',')];
  for (const row of _benchmarkRows) {
    lines.push(columns.map((c) => _csvEscape(row[c])).join(','));
  }
  const csv = lines.join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const startIso = _formatIso(_benchmarkStartEpochMs).replace(/[:.]/g, '-');
  const a = document.createElement('a');
  a.href = url;
  a.download = `benchmark_capture_${startIso}_${rowCount}rows.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  if ($benchmarkStatus) $benchmarkStatus.textContent = `Saved ${rowCount} rows`;

  const summary = _computeBenchmarkSummary(_benchmarkRows);
  _benchmarkLastSummaryText = _buildBenchmarkSummaryText(summary);
  _renderBenchmarkSummary(_benchmarkLastSummaryText);
}

window.__benchmarkCapture = {
  isActive: () => _benchmarkActive,
  start: benchmarkCaptureStart,
  stopAndDownload: benchmarkCaptureStopAndDownload,
  push: benchmarkCapturePush,
  getLastSummaryText: () => _benchmarkLastSummaryText,
};

_renderBenchmarkSummary(_benchmarkLastSummaryText);

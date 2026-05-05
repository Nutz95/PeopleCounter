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

// ── 1-second smoothed count display ───────────────────────────
// Updated on a timer instead of per-frame to keep the value stable.
setInterval(() => {
  const avg = peekAvg(_smooth.count, COUNT_SMOOTH_WIN);
  if (avg !== null && $countAvgValue) $countAvgValue.textContent = Math.round(avg);
}, 1000);

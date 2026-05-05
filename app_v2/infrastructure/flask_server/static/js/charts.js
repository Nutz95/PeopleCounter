// ── charts.js ──────────────────────────────────────────────────────────────
// Chart rendering: GPU sparkline, person-count sparkline, end-to-end latency
// chart.  Also drives the 1-second GPU-stats polling loop.
// Depends on: state.js
// ──────────────────────────────────────────────────────────────────────────

const gpuChart       = document.getElementById('gpu-chart');
const gpuCtx         = gpuChart.getContext('2d');
const countHistChart = document.getElementById('count-hist-chart');
const countHistCtx   = countHistChart.getContext('2d');

// ── GPU utilisation sparkline ──────────────────────────────────
function _drawGpuChart() {
  const W = gpuChart.width  = gpuChart.offsetWidth || 600;
  const H = gpuChart.height = 90;
  gpuCtx.clearRect(0, 0, W, H);
  if (gpuHist.length < 2) return;
  const xOf = i => (i / (gpuHist.length - 1)) * W;
  const yOf = v => H - (v / 100) * H;
  // 50 % guide line
  gpuCtx.beginPath();
  gpuCtx.strokeStyle = 'rgba(255,255,255,0.10)';
  gpuCtx.setLineDash([4, 4]);
  gpuCtx.moveTo(0, yOf(50)); gpuCtx.lineTo(W, yOf(50));
  gpuCtx.stroke();
  gpuCtx.setLineDash([]);
  // Area fill
  gpuCtx.beginPath();
  gpuCtx.moveTo(xOf(0), H);
  for (let i = 0; i < gpuHist.length; i++) gpuCtx.lineTo(xOf(i), yOf(gpuHist[i]));
  gpuCtx.lineTo(xOf(gpuHist.length - 1), H);
  gpuCtx.closePath();
  gpuCtx.fillStyle = 'rgba(42,223,165,0.18)';
  gpuCtx.fill();
  // Line
  gpuCtx.beginPath();
  for (let i = 0; i < gpuHist.length; i++) {
    if (i === 0) gpuCtx.moveTo(xOf(0), yOf(gpuHist[0]));
    else gpuCtx.lineTo(xOf(i), yOf(gpuHist[i]));
  }
  gpuCtx.strokeStyle = 'rgba(42,223,165,0.85)';
  gpuCtx.lineWidth = 2;
  gpuCtx.stroke();
}

// ── Person-count sparkline ────────────────────────────────────
function _drawCountHistChart() {
  const W = countHistChart.width  = countHistChart.offsetWidth || 360;
  const H = countHistChart.height = 90;
  countHistCtx.clearRect(0, 0, W, H);
  if (countHist.length < 2) return;
  const maxV = Math.max(...countHist, 1);
  const xOf  = i => (i / (countHist.length - 1)) * W;
  const yOf  = v => H - (v / maxV) * H * 0.90;
  // Area fill
  countHistCtx.beginPath();
  countHistCtx.moveTo(xOf(0), H);
  for (let i = 0; i < countHist.length; i++) countHistCtx.lineTo(xOf(i), yOf(countHist[i]));
  countHistCtx.lineTo(xOf(countHist.length - 1), H);
  countHistCtx.closePath();
  countHistCtx.fillStyle = 'rgba(165,210,255,0.14)';
  countHistCtx.fill();
  // Line
  countHistCtx.beginPath();
  for (let i = 0; i < countHist.length; i++) {
    if (i === 0) countHistCtx.moveTo(xOf(0), yOf(countHist[0]));
    else countHistCtx.lineTo(xOf(i), yOf(countHist[i]));
  }
  countHistCtx.strokeStyle = 'rgba(165,210,255,0.85)';
  countHistCtx.lineWidth = 2;
  countHistCtx.stroke();
}

// ── GPU stats polling (every 1 s) ─────────────────────────────
setInterval(async () => {
  // Smoothed count chart sample
  const _cAvg = peekAvg(_smooth.count, COUNT_SMOOTH_WIN);
  if (_cAvg !== null) {
    const _cv = Math.round(_cAvg);
    countHist.push(_cv);
    if (countHist.length > 60) countHist.shift();
    const $chBadge = document.getElementById('count-hist-badge');
    if ($chBadge) $chBadge.textContent = _cv;
    _drawCountHistChart();
  }
  // GPU stats
  try {
    const res  = await fetch('/api/gpu/stats');
    if (!res.ok) return;
    const data = await res.json();
    if (!data.available) return;
    gpuHist.push(data.gpu_util);
    if (gpuHist.length > 60) gpuHist.shift();
    const utilBadge = document.getElementById('gpu-util-badge');
    const memBadge  = document.getElementById('gpu-mem-badge');
    if (utilBadge) utilBadge.textContent = `${data.gpu_util}%`;
    if (memBadge)
      memBadge.textContent = `VRAM ${data.mem_used_mb} / ${data.mem_total_mb} MB`;
    _drawGpuChart();
  } catch (_) { /* ignore network errors */ }
}, 1000);

// ── End-to-end latency chart ──────────────────────────────────
function drawChart() {
  const W = latChart.offsetWidth || 600;
  const H = 170;
  if (latChart.width !== W || latChart.height !== H) {
    latChart.width  = W;
    latChart.height = H;
  } else {
    latCtx.clearRect(0, 0, W, H);
  }

  const N = e2eHist.length;
  if (N < 2) return;

  const allVals = [...e2eHist, ...tilesHist, TARGET_MS];
  const maxVal  = Math.max(...allVals) * 1.15;

  function xOf(i)   { return (i / (MAX_HISTORY - 1)) * W; }
  function yOf(val) { return H - (val / maxVal) * (H - 14); }

  // 33 ms reference line
  latCtx.beginPath();
  latCtx.strokeStyle = 'rgba(255,255,255,0.15)';
  latCtx.setLineDash([5, 5]);
  latCtx.moveTo(0, yOf(TARGET_MS));
  latCtx.lineTo(W, yOf(TARGET_MS));
  latCtx.stroke();
  latCtx.setLineDash([]);
  latCtx.fillStyle = 'rgba(255,255,255,0.28)';
  latCtx.font = '10px sans-serif';
  latCtx.fillText('33 ms', 4, yOf(TARGET_MS) - 3);

  function drawLine(data, color, fill) {
    if (data.length < 2) return;
    latCtx.beginPath();
    latCtx.strokeStyle = color;
    latCtx.lineWidth   = 2;
    latCtx.moveTo(xOf(0), H);
    for (let i = 0; i < data.length; i++) latCtx.lineTo(xOf(i), yOf(data[i]));
    latCtx.lineTo(xOf(data.length - 1), H);
    latCtx.closePath();
    latCtx.fillStyle = fill;
    latCtx.fill();
    latCtx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = xOf(i), y = yOf(data[i]);
      if (i === 0) latCtx.moveTo(x, y); else latCtx.lineTo(x, y);
    }
    latCtx.stroke();
  }

  drawLine(tilesHist, 'rgba(245,158,11,0.85)', 'rgba(245,158,11,0.07)');
  drawLine(e2eHist,   'rgba(42,223,165,0.9)',  'rgba(42,223,165,0.08)');
}

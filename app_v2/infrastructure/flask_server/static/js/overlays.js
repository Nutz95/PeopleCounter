// ── overlays.js ────────────────────────────────────────────────────────────
// Canvas overlay renderers: bounding-box mask, segmentation mask, density
// heatmap.  All renderers share _computeLayout() so overlays are always
// aligned to the actual letterboxed video content area.
// Depends on: state.js
// ──────────────────────────────────────────────────────────────────────────

// ── Shared layout helper ──────────────────────────────────────
// Returns the letterbox display geometry for the current video canvas state.
// Returns null if the canvas is not yet laid out or source dims are unknown.
function _computeLayout() {
  const wrapper = maskCanvas.parentElement;
  const W = wrapper.clientWidth;
  const H = wrapper.clientHeight;
  if (W === 0 || H === 0) return null;

  const canvasEl  = document.getElementById('video-canvas');
  const videoEl   = document.getElementById('video-feed');
  const useCanvas = canvasEl.style.display !== 'none';
  const fallbackW = useCanvas ? canvasEl.width  : (videoEl.naturalWidth  || 0);
  const fallbackH = useCanvas ? canvasEl.height : (videoEl.naturalHeight || 0);
  const srcW = _videoSrcW || fallbackW;
  const srcH = _videoSrcH || fallbackH;
  if (srcW <= 0 || srcH <= 0) return null;

  const dispScale = Math.min(W / srcW, H / srcH);
  const dispW = srcW * dispScale;
  const dispH = srcH * dispScale;
  const dispX = (W - dispW) / 2;
  const dispY = (H - dispH) / 2;
  return { W, H, srcW, srcH, dispW, dispH, dispX, dispY };
}

// ── Bounding-box overlay ──────────────────────────────────────
function drawMask(payload) {
  // WebGL renderer takes over mask drawing when available — skip Canvas2D.
  if (window._webglOverlayActive) return;

  const layout = _computeLayout();
  if (!layout) return;
  const { W, H, dispW, dispH, dispX, dispY } = layout;
  if (maskCanvas.width !== W || maskCanvas.height !== H) {
    maskCanvas.width  = W;
    maskCanvas.height = H;
  } else {
    maskCtx.clearRect(0, 0, W, H);
  }

  for (const p of payload) {
    if (!p || !Array.isArray(p.detections)) continue;
    for (const det of p.detections) {
      if (!det || !Array.isArray(det.bbox) || det.bbox.length < 4) continue;

      const [gx1, gy1, gx2, gy2] = det.bbox;
      const bx = dispX + gx1 * dispW;
      const by = dispY + gy1 * dispH;
      const bw = (gx2 - gx1) * dispW;
      const bh = (gy2 - gy1) * dispH;
      if (bw <= 0 || bh <= 0) continue;

      if (renderPoints) {
        const cx = bx + bw * 0.5;
        const cy = by + bh * 0.5;
        maskCtx.fillStyle = 'rgba(42,223,165,0.9)';
        maskCtx.beginPath();
        maskCtx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
        maskCtx.fill();
      } else {
        maskCtx.strokeStyle = 'rgba(42,223,165,0.9)';
        maskCtx.fillStyle   = 'rgba(42,223,165,0.12)';
        maskCtx.lineWidth   = 2;
        maskCtx.beginPath();
        maskCtx.roundRect(bx, by, bw, bh, 4);
        maskCtx.fill();
        maskCtx.stroke();
      }

      if (showBboxText && det.conf != null) {
        const label = `${(det.conf * 100).toFixed(0)}%`;
        maskCtx.fillStyle = 'rgba(5,11,23,0.75)';
        maskCtx.fillRect(bx, by - 18, label.length * 7 + 8, 18);
        maskCtx.fillStyle = '#e5fffa';
        maskCtx.font = '11px monospace';
        maskCtx.fillText(label, bx + 4, by - 4);
      }
    }
  }
}

// ── Packed detections overlay (metadata websocket) ────────────
function drawMaskPacked(rows) {
  // WebGL renderer takes over mask drawing when available — skip Canvas2D.
  if (window._webglOverlayActive) return;

  const layout = _computeLayout();
  if (!layout) return;
  const { W, H, dispW, dispH, dispX, dispY } = layout;
  if (maskCanvas.width !== W || maskCanvas.height !== H) {
    maskCanvas.width  = W;
    maskCanvas.height = H;
  } else {
    maskCtx.clearRect(0, 0, W, H);
  }
  if (!rows || rows.length < 3) return;
  const rowWidth = _packedDetectionsRowWidth || 5;
  const centersMode = rowWidth === 3;
  if (!(rowWidth === 3 || rowWidth === 5)) return;

  for (let i = 0; i + rowWidth - 1 < rows.length; i += rowWidth) {
    const conf = rows[i + rowWidth - 1];

    if (centersMode) {
      const cx = dispX + rows[i + 0] * dispW;
      const cy = dispY + rows[i + 1] * dispH;

      maskCtx.fillStyle = 'rgba(42,223,165,0.9)';
      maskCtx.beginPath();
      maskCtx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
      maskCtx.fill();

      if (showBboxText) {
        const label = `${(conf * 100).toFixed(0)}%`;
        maskCtx.fillStyle = 'rgba(5,11,23,0.75)';
        maskCtx.fillRect(cx - 2, cy - 18, label.length * 7 + 8, 18);
        maskCtx.fillStyle = '#e5fffa';
        maskCtx.font = '11px monospace';
        maskCtx.fillText(label, cx + 2, cy - 4);
      }
      continue;
    }

    const gx1 = rows[i + 0];
    const gy1 = rows[i + 1];
    const gx2 = rows[i + 2];
    const gy2 = rows[i + 3];

    const bx = dispX + gx1 * dispW;
    const by = dispY + gy1 * dispH;
    const bw = (gx2 - gx1) * dispW;
    const bh = (gy2 - gy1) * dispH;
    if (bw <= 0 || bh <= 0) continue;

    if (renderPoints) {
      const cx = bx + bw * 0.5;
      const cy = by + bh * 0.5;
      maskCtx.fillStyle = 'rgba(42,223,165,0.9)';
      maskCtx.beginPath();
      maskCtx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
      maskCtx.fill();
    } else {
      maskCtx.strokeStyle = 'rgba(42,223,165,0.9)';
      maskCtx.fillStyle   = 'rgba(42,223,165,0.12)';
      maskCtx.lineWidth   = 2;
      maskCtx.beginPath();
      maskCtx.roundRect(bx, by, bw, bh, 4);
      maskCtx.fill();
      maskCtx.stroke();
    }

    if (showBboxText) {
      const label = `${(conf * 100).toFixed(0)}%`;
      maskCtx.fillStyle = 'rgba(5,11,23,0.75)';
      maskCtx.fillRect(bx, by - 18, label.length * 7 + 8, 18);
      maskCtx.fillStyle = '#e5fffa';
      maskCtx.font = '11px monospace';
      maskCtx.fillText(label, bx + 4, by - 4);
    }
  }
}

// ── Segmentation mask overlay (YOLO-seg proto masks) ──────────
// Packed-RGBA trick: write 4 bytes in a single Int32 write via shared
// ArrayBuffer.  ~4× faster than per-channel byte assignment.
// Canvas RGBA memory order: R G B A → little-endian Int32 = 0xAABBGGRR
// Teal 50 % alpha: R=0x2A G=0xDF B=0xA5 A=0x80 → Int32LE = 0x80A5DF2A
const _SEG_PIXEL_ON  = 0x80A5DF2A;
const _segOffscreen  = document.createElement('canvas');

function drawSegMask(payload) {
  const layout = _computeLayout();
  if (!layout) return;
  const { W, H, srcW, srcH, dispW, dispH, dispX, dispY } = layout;
  if (segCanvas.width !== W || segCanvas.height !== H) {
    segCanvas.width  = W;
    segCanvas.height = H;
  } else {
    segCtx.clearRect(0, 0, W, H);
  }

  for (const p of payload) {
    if (!p || !p.seg_mask_raw || !p.seg_mask_w || !p.seg_mask_h) continue;
    const mw  = p.seg_mask_w | 0;
    const mh  = p.seg_mask_h | 0;
    const nPx = mw * mh;

    if (_segOffscreen.width !== mw || _segOffscreen.height !== mh) {
      _segOffscreen.width  = mw;
      _segOffscreen.height = mh;
    }
    const offCtx  = _segOffscreen.getContext('2d');
    const imgData = offCtx.createImageData(mw, mh);

    const bStr = atob(p.seg_mask_raw);
    if (bStr.length < nPx) continue;

    const int32 = new Int32Array(imgData.data.buffer);
    for (let i = 0; i < nPx; i++) {
      int32[i] = bStr.charCodeAt(i) > 128 ? _SEG_PIXEL_ON : 0;
    }
    offCtx.putImageData(imgData, 0, 0);

    if (p.seg_mask_is_canvas) {
      // Tiles mode: canvas covers the full frame, map directly to display area.
      segCtx.drawImage(
        _segOffscreen,
        0, 0, mw, mh,
        dispX, dispY, dispW, dispH,
      );
    } else {
      // Global model: proto mask is in letterboxed 640×640 space — strip padding.
      const modelSz = 640;
      const mScale  = Math.min(modelSz / srcW, modelSz / srcH);
      const prepW   = srcW * mScale;
      const prepH   = srcH * mScale;
      const padX    = (modelSz - prepW) / 2;
      const padY    = (modelSz - prepH) / 2;
      const mpadX   = padX  * mw / modelSz;
      const mpadY   = padY  * mh / modelSz;
      const mCntW   = prepW * mw / modelSz;
      const mCntH   = prepH * mh / modelSz;
      segCtx.drawImage(
        _segOffscreen,
        mpadX, mpadY, mCntW, mCntH,
        dispX, dispY, dispW, dispH,
      );
    }
  }
}

// ── Density heatmap overlay ────────────────────────────────────
// DensityDecoder emits hotspot coordinates {x, y, w} (normalised [0,1]).
// Each hotspot is rendered as a filled circle scaled by weight w.
function drawHeatmap(payload) {
  const layout = _computeLayout();
  if (!layout) return;
  const { W, H, dispX, dispY, dispW, dispH } = layout;
  if (heatCanvas.width !== W || heatCanvas.height !== H) {
    heatCanvas.width  = W;
    heatCanvas.height = H;
  } else {
    heatCtx.clearRect(0, 0, W, H);
  }

  heatCtx.fillStyle = 'rgba(220, 30, 30, 0.85)';
  for (const p of payload) {
    if (!p || !Array.isArray(p.hotspots) || p.hotspots.length === 0) continue;
    for (const hs of p.hotspots) {
      const cx = dispX + hs.x * dispW;
      const cy = dispY + hs.y * dispH;
      const r  = Math.max(3, Math.round(hs.w * 12));
      heatCtx.beginPath();
      heatCtx.arc(cx, cy, r, 0, 2 * Math.PI);
      heatCtx.fill();
    }
    break;  // only one density payload expected per frame
  }
}

// ── webcodecs.js ───────────────────────────────────────────────────────────
// Zero-encode video path: forwards raw H.264 Annex-B packets from the server
// WebSocket directly into the browser's VideoDecoder (WebCodecs API).
// Falls back to MJPEG transparently if WebCodecs is unavailable or sync mode
// is active.
//
// Architecture:
//   PyFFmpegDemuxer → Annex-B H.264 → WebSocket binary
//   → EncodedVideoChunk → VideoDecoder → output(frame) → ctx.drawImage → canvas
//
// Exposes: window.__pcSetWebCodecsEnabled(bool)
// Depends on: state.js (window.__syncMode, _dispFpsTs, $displayFpsChip)
//             window._SERVER_WS_PORT (set inline in index.html from Jinja2)
// ──────────────────────────────────────────────────────────────────────────

(function () {
  if (typeof VideoDecoder === 'undefined') return; // no WebCodecs support

  window.__videoTransport = window.__videoTransport || 'mjpeg';

  let _wsPort = (window._SERVER_WS_PORT || 5001);

  async function _fetchWsPort() {
    try {
      const r = await fetch('/api/ws_port');
      if (r.ok) { const j = await r.json(); if (j.ws_port) _wsPort = j.ws_port; }
    } catch (_) { /* keep last-known port */ }
  }

  const videoCanvas      = document.getElementById('video-canvas');
  const videoFeed        = document.getElementById('video-feed');
  const ctx              = videoCanvas.getContext('2d');
  const $displayModeChip = document.getElementById('display-mode-chip');

  const DECODE_QUEUE_MAX = 8;  // drop non-keyframes when queue exceeds this

  let decoder          = null;
  let ws               = null;
  let initDone         = false;
  let _lastInitKey     = null;
  let _lastConfig      = null;
  // Gate all decodes on a clean IDR boundary after any dropped P-frame.
  let _waitForKeyframe = false;
  let _stopped         = false;

  // 15-second safety timer: if no init message arrives, source is not H.264.
  const fallbackTimer = setTimeout(() => {
    if (!initDone) teardown();
  }, 15000);

  // ── Helpers ────────────────────────────────────────────────────
  function base64ToBuffer(b64) {
    const bin = atob(b64);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    return buf.buffer;
  }

  function activateCanvas() {
    videoFeed.style.display   = 'none';
    videoCanvas.style.display = 'block';
    window.__videoTransport = 'webcodecs';
    if ($displayModeChip) $displayModeChip.textContent = 'WebCodecs';
  }

  // hard=true  → permanently stopped (codec unsupported, 15 s fallback fired).
  // hard=false → soft stop for sync-mode switch; onclose will NOT reconnect.
  function teardown(hard = true) {
    _stopped = hard;
    clearTimeout(fallbackTimer);
    try { ws && ws.close(); } catch (e) {}
    try { decoder && decoder.close(); } catch (e) {}
    window.__videoTransport = 'mjpeg';
    decoder = null; ws = null; _lastConfig = null; _waitForKeyframe = false;
    if ($displayModeChip) $displayModeChip.textContent = 'MJPEG';
  }

  function forceMjpegView() {
    videoCanvas.style.display = 'none';
    videoFeed.style.display   = '';
    // The browser suspends the multipart-MJPEG connection when the <img> is
    // display:none.  Simply un-hiding it leaves a stale (dead) stream.
    // Force a fresh HTTP request with a cache-busting timestamp.
    videoFeed.src = '/api/video?_t=' + Date.now();
    window.__videoTransport = 'mjpeg';
    if ($displayModeChip) $displayModeChip.textContent = 'MJPEG';
  }

  // ── Public API ─────────────────────────────────────────────────
  function setWebCodecsEnabled(enabled) {
    if (!enabled) {
      teardown(false);
      forceMjpegView();
      return;
    }
    if (ws || (decoder && decoder.state === 'configured')) return;
    _stopped = false;
    connectWs();
  }
  window.__pcSetWebCodecsEnabled = setWebCodecsEnabled;

  // ── VideoDecoder factory ───────────────────────────────────────
  // Factored out so the error handler can inline-reset the decoder without
  // closing the WebSocket (no 500 ms reconnect gap).
  function makeVideoDecoder() {
    return new VideoDecoder({
      output: (frame) => {
        ctx.drawImage(frame, 0, 0, videoCanvas.width, videoCanvas.height);
        frame.close();
        const now = performance.now();
        _dispFpsTs.push(now);
        while (_dispFpsTs.length && now - _dispFpsTs[0] > 2000) _dispFpsTs.shift();
        const dFps = _dispFpsTs.length > 1
          ? Math.round((_dispFpsTs.length - 1) / ((now - _dispFpsTs[0]) / 1000))
          : 0;
        if ($displayFpsChip) $displayFpsChip.textContent = dFps ? `${dFps} fps vid` : '\u2014 fps vid';
      },
      error: (e) => {
        console.warn('VideoDecoder error:', e.message || e);
        if (!initDone) {
          teardown();
          videoFeed.style.display   = '';
          videoCanvas.style.display = 'none';
          return;
        }
        if (!_lastConfig) {
          try { decoder && decoder.close(); } catch (_) {}
          decoder = null;
          try { ws && ws.close(); } catch (_) {}
          return;
        }
        // Inline reset — replace the broken decoder without closing the WS.
        // With GOP=1 every upcoming packet is an IDR so decoding resumes instantly.
        try { decoder && decoder.close(); } catch (_) {}
        decoder = makeVideoDecoder();
        try { decoder.configure(_lastConfig); } catch (_) {}
        _waitForKeyframe = true;
      },
    });
  }

  // ── Init message handler ───────────────────────────────────────
  function onInitMsg(msg) {
    clearTimeout(fallbackTimer);
    initDone = true;

    const codec   = msg.codec || 'avc1';
    const initKey = `${codec}:${msg.width || 0}:${msg.height || 0}:${msg.description || ''}`;

    // Skip re-init when nothing changed and the decoder is already running.
    if (initKey === _lastInitKey && decoder && decoder.state === 'configured') return;
    _lastInitKey     = initKey;
    _waitForKeyframe = false;

    const config = {
      codec,
      optimizeForLatency:    true,
      hardwareAcceleration:  'no-preference',
    };
    if (msg.description) config.description = base64ToBuffer(msg.description);
    if (msg.width  && videoCanvas.width  !== msg.width)  { config.codedWidth  = msg.width;  videoCanvas.width  = msg.width; }
    if (msg.height && videoCanvas.height !== msg.height) { config.codedHeight = msg.height; videoCanvas.height = msg.height; }
    if (msg.width  && config.codedWidth  === undefined)  config.codedWidth  = msg.width;
    if (msg.height && config.codedHeight === undefined)  config.codedHeight = msg.height;

    if (decoder) { try { decoder.close(); } catch (_) {} decoder = null; }
    _lastConfig = null;
    decoder = makeVideoDecoder();

    VideoDecoder.isConfigSupported(config).then(support => {
      if (!support.supported) {
        console.warn('VideoDecoder: codec not supported', config);
        try { decoder && decoder.close(); } catch (_) {}
        decoder = null;
        try { ws && ws.close(); } catch (_) {}
        return;
      }
      decoder.configure(config);
      _lastConfig = config;
      activateCanvas();
    }).catch((e) => {
      console.warn('VideoDecoder.isConfigSupported error:', e);
      try { decoder && decoder.close(); } catch (_) {}
      decoder = null;
      try { ws && ws.close(); } catch (_) {}
    });
  }

  // ── Binary frame handler ───────────────────────────────────────
  function onBinaryMsg(data) {
    if (!decoder || decoder.state !== 'configured') return;
    const view       = new DataView(data);
    const flags      = view.getUint8(0);
    const isKeyframe = (flags & 1) !== 0;

    if (!isKeyframe && decoder.decodeQueueSize > DECODE_QUEUE_MAX) {
      _waitForKeyframe = true;
      return;
    }
    if (_waitForKeyframe) {
      if (!isKeyframe) return;
      _waitForKeyframe = false;
    }

    // pts_us: u64 LE read as two u32 to avoid BigInt dependency.
    const pts_lo  = view.getUint32(1, true);
    const pts_hi  = view.getUint32(5, true);
    const pts_us  = pts_hi * 4294967296 + pts_lo;
    const payload = data.slice(9);

    try {
      decoder.decode(new EncodedVideoChunk({
        type:      isKeyframe ? 'key' : 'delta',
        timestamp: pts_us,
        data:      payload,
      }));
    } catch (e) {
      // "decode called before configure" on startup — harmless.
    }
  }

  // ── WebSocket connect ──────────────────────────────────────────
  let _wsFailCount    = 0;
  const $wsErrorBanner = document.getElementById('ws-error-banner');
  const $wsErrorPort   = document.getElementById('ws-error-port');

  async function connectWs() {
    if (_stopped) return;
    if (window.__syncMode === 'sync') {
      // Sync mode: stay on MJPEG.  Retry when the user switches back to async.
      forceMjpegView();
      setTimeout(() => {
        if (!_stopped && window.__syncMode !== 'sync' && !ws) connectWs();
      }, 1200);
      return;
    }

    await _fetchWsPort();
    const wsUrl = `ws://${window.location.hostname}:${_wsPort}`;
    console.debug('[WebCodecs] connecting to', wsUrl, '(attempt', _wsFailCount + 1, ')');

    try { ws && ws.close(); } catch (_) {}
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg && msg.type === 'init') {
          _wsFailCount = 0;
          if ($wsErrorBanner) $wsErrorBanner.style.display = 'none';
          onInitMsg(msg);
        }
      } else {
        onBinaryMsg(ev.data);
      }
    };

    ws.onerror = () => {
      _wsFailCount++;
      if (_wsFailCount >= 3 && $wsErrorBanner && !initDone) {
        $wsErrorBanner.style.display = 'block';
        if ($wsErrorPort) $wsErrorPort.textContent = _wsPort;
      }
    };

    ws.onclose = () => {
      try { decoder && decoder.close(); } catch (_) {}
      decoder = null;
      if (!_stopped && window.__syncMode !== 'sync') {
        setTimeout(connectWs, initDone ? 500 : 2000);
      }
    };
  }

  connectWs();
})();

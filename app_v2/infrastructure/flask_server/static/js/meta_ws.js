// ── meta_ws.js ───────────────────────────────────────────────────────────
// Binary metadata WebSocket: receives packed detections [x1,y1,x2,y2,conf]
// and stores latest frame-aligned rows for overlay rendering.
// Depends on: state.js
// ──────────────────────────────────────────────────────────────────────────

(function () {
  let _metaWsPort = (window._SERVER_META_WS_PORT || 5003);
  let _ws = null;
  let _stopped = false;
  let _enabled = true;
  const _packetTs = [];

  window.__metaWsConnected = false;
  window.__metaWsPacketRate = 0;

  function _recordPacket() {
    const now = performance.now();
    _packetTs.push(now);
    while (_packetTs.length && now - _packetTs[0] > 2000) _packetTs.shift();
    const pps = _packetTs.length > 1
      ? (_packetTs.length - 1) / ((now - _packetTs[0]) / 1000)
      : 0;
    window.__metaWsPacketRate = pps;
  }

  async function _fetchMetaWsPort() {
    try {
      const r = await fetch('/api/meta_ws_port');
      if (r.ok) {
        const j = await r.json();
        if (j.ws_port) _metaWsPort = j.ws_port;
      }
    } catch (_) {
      // keep last-known port
    }
  }

  function _parsePackedDetections(buffer) {
    const view = new DataView(buffer);
    if (view.byteLength < 16) return null;

    const m0 = view.getUint8(0), m1 = view.getUint8(1), m2 = view.getUint8(2), m3 = view.getUint8(3);
    if (m0 !== 0x50 || m1 !== 0x43 || m2 !== 0x4d || m3 !== 0x42) return null; // "PCMB"

    const version = view.getUint8(4);
    if (version !== 1) return null;

    const flags = view.getUint16(6, true);
    const centersMode = (flags & 0x0001) !== 0;
    const rowWidth = centersMode ? 3 : 5;
    const frameId = view.getUint32(8, true);
    const count = view.getUint32(12, true);
    const expectedBytes = 16 + count * rowWidth * 4;
    if (view.byteLength < expectedBytes) return null;

    const rows = new Float32Array(buffer, 16, count * rowWidth);
    return { frameId, rows, rowWidth };
  }

  async function _connect() {
    if (_stopped || !_enabled) return;
    await _fetchMetaWsPort();

    const url = `ws://${window.location.hostname}:${_metaWsPort}`;
    try { _ws && _ws.close(); } catch (_) {}
    _ws = new WebSocket(url);
    _ws.binaryType = 'arraybuffer';

    _ws.onopen = () => {
      window.__metaWsConnected = true;
    };

    _ws.onmessage = (ev) => {
      if (!(ev.data instanceof ArrayBuffer)) return;
      const parsed = _parsePackedDetections(ev.data);
      if (!parsed) return;
      _recordPacket();
      _packedDetectionsFrameId = parsed.frameId;
      _packedDetections = parsed.rows;
      _packedDetectionsRowWidth = parsed.rowWidth;
    };

    _ws.onclose = () => {
      window.__metaWsConnected = false;
      window.__metaWsPacketRate = 0;
      if (!_stopped && _enabled) setTimeout(_connect, 1000);
    };

    _ws.onerror = () => {
      window.__metaWsConnected = false;
      try { _ws && _ws.close(); } catch (_) {}
    };
  }

  function _disconnectNow() {
    window.__metaWsConnected = false;
    window.__metaWsPacketRate = 0;
    _packedDetectionsFrameId = -1;
    _packedDetections = null;
    _packedDetectionsRowWidth = 5;
    try { _ws && _ws.close(); } catch (_) {}
    _ws = null;
  }

  window.__pcSetMetaWsEnabled = function (enabled) {
    const next = !!enabled;
    if (_enabled === next) return;
    _enabled = next;
    if (_enabled) {
      _stopped = false;
      _connect();
      return;
    }
    _disconnectNow();
  };

  _connect();
})();

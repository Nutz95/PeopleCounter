import cv2
import glob
import os
import threading
from flask import Flask, Response, render_template_string, jsonify, request


class WebServer:
    def __init__(self, port=5000):
        self.app = Flask(__name__)
        self.port = port
        self.frame = None
        self.lock = threading.Lock()
        self.metrics = {}
        self.pipeline = None
        self.available_profiles = self._discover_profiles()
        self.active_profile_view = self.available_profiles[0] if self.available_profiles else "Live Metrics"

        @self.app.route('/')
        def index():
            return render_template_string('''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PeopleCounter GPU Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg-a: #050b17;
      --bg-b: #111635;
      --panel: #0f1a33;
      --panel-strong: #1c2950;
      --accent: #2adfa5;
      --muted: #a1b0d0;
    }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: 'Space Grotesk', 'Poppins', 'Fira Sans', sans-serif;
      background: radial-gradient(circle at top right, #152b62 0%, #050a17 45%, #020409 100%);
      color: #f6f7fb;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px;
    }
    header.hero {
      width: min(1200px, 100%);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 18px;
      margin-bottom: 18px;
      padding: 16px 28px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 20px 60px rgba(5, 8, 23, 0.6);
    }
    header.hero h1 {
      margin: 4px 0;
      font-size: 2rem;
    }
    header.hero .hero-note {
      font-size: 0.9rem;
      color: var(--muted);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    main.page-grid {
      width: min(1200px, 100%);
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 24px;
      margin-bottom: 24px;
    }
    .video-shell, .metrics-shell {
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: 0 25px 45px rgba(2, 4, 10, 0.8);
      padding: 20px;
      animation: floatIn 0.6s ease-out both;
    }
    .video-wrapper {
      position: relative;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.15);
      background: var(--panel);
    }
    .video-wrapper img {
      display: block;
      width: 100%;
      height: auto;
      max-height: 70vh;
      filter: saturate(1.05);
      transition: transform 0.4s ease;
    }
    .video-wrapper:hover img {
      transform: scale(1.01);
    }
    .video-actions {
      position: absolute;
      right: 16px;
      bottom: 16px;
      display: flex;
      gap: 8px;
    }
    .video-actions button {
      background: rgba(42, 223, 165, 0.15);
      border: 1px solid rgba(42, 223, 165, 0.8);
      color: var(--accent);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: transform 0.2s ease, background 0.2s ease;
    }
    .video-actions button:hover {
      transform: translateY(-1px);
      background: rgba(42, 223, 165, 0.25);
    }
    .fps-chip {
      position: absolute;
      top: 18px;
      right: 18px;
      padding: 6px 14px;
      border-radius: 999px;
      background: rgba(6, 255, 203, 0.12);
      border: 1px solid rgba(42, 223, 165, 0.3);
      font-size: 0.9rem;
    }
    .fps-status {
      position: absolute;
      top: 18px;
      left: 18px;
      padding: 6px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      font-size: 0.8rem;
      color: var(--muted);
    }
    .video-hint {
      margin-top: 12px;
      font-size: 0.85rem;
      color: var(--muted);
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
    }
    .metric-card {
      border-radius: 16px;
      padding: 16px;
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.07);
      min-height: 100px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: inset 0 0 14px rgba(255, 255, 255, 0.03);
      animation: floatIn 0.7s ease-out both;
    }
    .metric-card.wide {
      grid-column: span 2;
    }
    .metric-title {
      font-size: 0.85rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .metric-value {
      font-size: 2.1rem;
      font-weight: 600;
    }
    .metric-footnote {
      font-size: 0.9rem;
      color: var(--muted);
    }
    .control-group {
      margin-top: 22px;
      padding: 18px;
      border-radius: 16px;
      background: var(--panel-strong);
      border: 1px solid rgba(255, 255, 255, 0.05);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.95rem;
    }
    .toggle input {
      accent-color: var(--accent);
      width: 18px;
      height: 18px;
    }
    .profile-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .profile-controls select,
    .profile-controls button {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 8px 14px;
      color: #f6f7fb;
      font-size: 0.9rem;
    }
    .profile-controls button {
      cursor: pointer;
    }
    .profile-controls button:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
    .profile-badge {
      font-size: 0.9rem;
      color: var(--muted);
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .profile-log {
      margin-top: 12px;
      border-radius: 14px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px dashed rgba(255, 255, 255, 0.15);
      min-height: 110px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .log-entry {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 10px;
      background: rgba(42, 223, 165, 0.08);
      border-radius: 10px;
      font-size: 0.85rem;
    }
    .log-time {
      font-weight: 600;
      color: #e5fffa;
    }
    .instruction-card {
      margin-top: 20px;
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 0.9rem;
      color: var(--muted);
    }
    a {
      color: var(--accent);
      text-decoration: none;
    }
    @keyframes floatIn {
      from {
        opacity: 0;
        transform: translateY(12px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    @media (max-width: 720px) {
      header.hero {
        flex-direction: column;
      }
      .metric-card.wide {
        grid-column: span 1;
      }
      .video-actions {
        position: static;
        justify-content: center;
        margin-top: 12px;
      }
    }
  </style>
</head>
<body>
  <header class="hero">
    <div>
      <p class="eyebrow">PeopleCounter GPU Stream</p>
      <h1>Live Metrics & Controls</h1>
      <p style="color: var(--muted); margin: 0;">Toggle profiles, debug logs, and fullscreen while seeing FPS, CPU, and density without the terminal.</p>
    </div>
    <div class="hero-note">
      <span>Port {{ port }}</span>
      <span>MJPEG stream</span>
    </div>
  </header>
  <main class="page-grid">
    <section class="video-shell">
      <div class="video-wrapper" id="videoWrapper">
        <img id="videoStream" src="/video_feed" alt="PeopleCounter stream">
        <div class="fps-chip" id="fpsBadge">FPS —</div>
        <div class="fps-status" id="fpsStatus">Awaiting metrics...</div>
        <div class="video-actions">
          <button id="fullscreenBtn" type="button">Fullscreen</button>
          <button id="fitBtn" type="button">Fit View</button>
        </div>
      </div>
      <p class="video-hint">Connect from any browser at http://localhost:{{ port }} to mirror the feed from this device.</p>
    </section>
    <section class="metrics-shell">
      <div class="metrics-grid">
        <article class="metric-card">
          <div class="metric-title">YOLO detections</div>
          <div class="metric-value" id="yoloCountValue">—</div>
          <div class="metric-footnote" id="yoloDeviceLabel">Device: —</div>
        </article>
        <article class="metric-card">
          <div class="metric-title">Density estimate</div>
          <div class="metric-value" id="densityCountValue">—</div>
          <div class="metric-footnote" id="densityDeviceLabel">Device: —</div>
        </article>
        <article class="metric-card">
          <div class="metric-title">Average people</div>
          <div class="metric-value" id="avgValue">—</div>
          <div class="metric-footnote">Last FPS: <span id="fpsValue">—</span></div>
        </article>
        <article class="metric-card wide">
          <div class="metric-title">Latency (ms)</div>
          <div class="metric-value"><span id="yoloTimeValue">—</span> / <span id="densityTimeValue">—</span></div>
          <div class="metric-footnote">Total: <span id="totalTimeValue">—</span></div>
        </article>
        <article class="metric-card wide">
          <div class="metric-title">CPU usage</div>
          <div class="metric-value" id="cpuValue">—%</div>
          <div class="metric-footnote" id="profileState">Idle</div>
        </article>
      </div>
      <div class="control-group">
        <label class="toggle">
          <input type="checkbox" id="debugToggle">
          <span>Enable verbose debug logs</span>
        </label>
        <div class="profile-controls">
          <select id="profileViewSelect">
            {% if profiles %}
              {% for profile in profiles %}
                <option value="{{ profile }}">{{ profile.replace('_', ' ') }}</option>
              {% endfor %}
            {% else %}
                <option value="Live Metrics">Live Metrics</option>
            {% endif %}
          </select>
          <button id="profileStart" type="button">Start Profiling</button>
          <button id="profileStop" type="button" disabled>Stop</button>
          <button id="profileClear" type="button">Clear Logs</button>
        </div>
        <div class="profile-badge">
          Active profile: <span id="profileViewBadge">{{ profiles[0] if profiles else 'Live Metrics' }}</span>
        </div>
      </div>
      <div class="profile-log" id="profileLog">
        <p class="log-empty">Profiles will appear here when profiling is active.</p>
      </div>
      <div class="instruction-card">
        <p>Use this web UI to monitor FPS, alerts, and device status while the camera feeds the backend in the same terminal.</p>
        <p>The profile selector mirrors what the launch script can configure via <code>scripts/configs</code> when you need a different backend.</p>
      </div>
    </section>
  </main>
  <script>
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
        if (data.active_profile_view) {
          profileViewBadge.textContent = data.active_profile_view.replace('_', ' ');
          if (profileViewSelect.value !== data.active_profile_view && [...profileViewSelect.options].some((opt) => opt.value === data.active_profile_view)) {
            profileViewSelect.value = data.active_profile_view;
          }
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
  </script>
</body>
</html>
''', port=self.port, profiles=self.available_profiles)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/api/metrics')
        def metrics_route():
            with self.lock:
                snapshot = dict(self.metrics)
            if self.pipeline:
                snapshot['profile_active'] = bool(self.pipeline.profile_active)
                snapshot['profile_log'] = self.pipeline.get_profile_log()
            else:
                snapshot.setdefault('profile_active', False)
                snapshot.setdefault('profile_log', [])
            snapshot.setdefault('active_profile_view', self.active_profile_view)
            return jsonify(snapshot)

        @self.app.route('/api/control', methods=['POST'])
        def control_route():
            payload = request.get_json(silent=True) or {}
            if not self.pipeline:
                return jsonify({"status": "no_pipeline"}), 400
            response = {"status": "ok"}
            if 'debug' in payload:
                self.pipeline.set_debug_mode(bool(payload['debug']))
                response['debug'] = self.pipeline.extreme_debug
            action = payload.get('profile_action')
            if action == 'start':
                self.pipeline.set_profile_logging(True)
            elif action == 'stop':
                self.pipeline.set_profile_logging(False)
            elif action == 'clear':
                self.pipeline.clear_profile_log()
            profile_name = payload.get('profile_name')
            if profile_name:
                self.active_profile_view = profile_name
                response['profile_view'] = profile_name
            return jsonify(response)

        @self.app.route('/api/profile-options')
        def profile_options_route():
            return jsonify({"profiles": self.available_profiles})

    def update_frame(self, frame):
        with self.lock:
            if frame is not None:
                self.frame = frame.copy()

    def update_metrics(self, metrics):
        with self.lock:
            if isinstance(metrics, dict):
                self.metrics = metrics.copy()
            else:
                self.metrics = metrics

    def attach_pipeline(self, pipeline):
        self.pipeline = pipeline

    def generate(self):
        while True:
            with self.lock:
                if self.frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', self.frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    def start(self):
        print(f"[*] Starting Web Server on http://0.0.0.0:{self.port}")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self.app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False)

    def _discover_profiles(self):
        configs = []
        root = os.getcwd()
        env_dir = os.path.join(root, 'scripts', 'configs')
        if os.path.isdir(env_dir):
            for path in sorted(glob.glob(os.path.join(env_dir, '*.env'))):
                name = os.path.splitext(os.path.basename(path))[0]
                if name not in configs:
                    configs.append(name)
        return configs

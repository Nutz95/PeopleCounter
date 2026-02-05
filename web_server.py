import cv2
import glob
import os
import threading
from flask import Flask, Response, jsonify, render_template, request


class WebServer:
    def __init__(self, port=5000):
        self.app = Flask(__name__)
        self.port = port
        self.frame = None
        self.encoded_frame = None
        self.lock = threading.Lock()
        self.metrics = {}
        self.pipeline = None
        self.available_profiles = self._discover_profiles()
        self.active_profile_view = self.available_profiles[0] if self.available_profiles else 'Live Metrics'

        @self.app.route('/')
        def index():
            profiles = self._gather_profiles()
            active_profile = (
                self.pipeline.active_profile_name if self.pipeline and self.pipeline.active_profile_name
                else self.active_profile_view or 'Live Metrics'
            )
            return render_template('index.html', port=self.port, profiles=profiles, active_profile=active_profile)

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
                snapshot['graph_overlay'] = bool(self.pipeline.graph_overlay_enabled)
            else:
                snapshot.setdefault('profile_active', False)
                snapshot.setdefault('profile_log', [])
                snapshot.setdefault('graph_overlay', False)
            snapshot.setdefault('active_profile_view', self.active_profile_view)
            return jsonify(snapshot)

        @self.app.route('/api/control', methods=['POST'])
        def control_route():
            payload = request.get_json(silent=True) or {}
            if not self.pipeline:
                return jsonify({'status': 'no_pipeline'}), 400
            response = {'status': 'ok'}
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
            overlay_state = payload.get('overlay')
            if overlay_state is not None:
                self.pipeline.set_graph_overlay(bool(overlay_state))
                response['overlay'] = bool(overlay_state)
            pipeline_mode = payload.get('pipeline_mode')
            if pipeline_mode is not None and hasattr(self.pipeline, 'set_yolo_pipeline_mode'):
                applied = self.pipeline.set_yolo_pipeline_mode(pipeline_mode)
                response['pipeline_mode'] = pipeline_mode
                response['pipeline_mode_applied'] = applied
            profile_name = payload.get('profile_name')
            if profile_name is not None:
                display_name = profile_name
                target_name = '' if display_name.lower() == 'live metrics' else profile_name
                applied = self.pipeline.apply_profile(target_name)
                self.active_profile_view = display_name
                response['profile_view'] = display_name
                response['profile_applied'] = applied
            return jsonify(response)

        @self.app.route('/api/profile-options')
        def profile_options_route():
            return jsonify({'profiles': self._gather_profiles()})

    def update_frame(self, frame):
        with self.lock:
            if frame is None:
                self.frame = None
                self.encoded_frame = None
                return
            if isinstance(frame, dict):
                encoded = frame.get('encoded')
                if isinstance(encoded, (bytes, bytearray)):
                    self.encoded_frame = bytes(encoded)
                else:
                    self.encoded_frame = None
                payload = frame.get('frame')
                if payload is None:
                    self.frame = None
                elif hasattr(payload, 'copy'):
                    try:
                        self.frame = payload.copy()
                    except Exception:
                        self.frame = payload
                else:
                    self.frame = payload
            elif isinstance(frame, (bytes, bytearray)):
                self.encoded_frame = bytes(frame)
                self.frame = None
            else:
                self.frame = frame.copy()
                self.encoded_frame = None

    def update_metrics(self, metrics):
        with self.lock:
            if isinstance(metrics, dict):
                self.metrics = metrics.copy()
            else:
                self.metrics = metrics

    def attach_pipeline(self, pipeline):
        self.pipeline = pipeline
        if pipeline:
            self.available_profiles = list(getattr(pipeline, 'available_profiles', []))
            active = pipeline.active_profile_name if pipeline.active_profile_name else 'Live Metrics'
            self.active_profile_view = active

    def generate(self):
        while True:
            with self.lock:
                frame = self.frame
                encoded = self.encoded_frame
            if encoded is not None:
                frame_bytes = encoded
            elif frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
            else:
                continue
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

    def _gather_profiles(self):
                pipeline_profiles = getattr(self.pipeline, 'available_profiles', None)
                if pipeline_profiles:
                        return list(pipeline_profiles)
                return list(self.available_profiles)

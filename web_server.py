import cv2
import threading
from flask import Flask, Response, render_template_string

class WebServer:
    def __init__(self, port=5000):
        self.app = Flask(__name__)
        self.port = port
        self.frame = None
        self.lock = threading.Lock()
        
        @self.app.route('/')
        def index():
            return render_template_string('''
                <html>
                    <head>
                        <title>PeopleCounter GPU Preview</title>
                        <style>
                            body { 
                                margin: 0; 
                                background-color: #1a1a1a; 
                                color: #f0f0f0; 
                                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                                justify-content: center;
                                height: 100vh;
                            }
                            h1 { margin-bottom: 20px; color: #00ffcc; text-shadow: 0 0 10px rgba(0, 255, 204, 0.3); }
                            .container {
                                position: relative;
                                border: 4px solid #333;
                                border-radius: 8px;
                                box-shadow: 0 0 30px rgba(0,0,0,0.5);
                                overflow: hidden;
                                background: #000;
                            }
                            img { display: block; max-width: 90vw; max-height: 80vh; }
                            .status { margin-top: 15px; font-size: 0.9em; color: #888; }
                        </style>
                    </head>
                    <body>
                        <h1>🚀 PeopleCounter GPU Stream</h1>
                        <div class="container">
                            <img src="/video_feed">
                        </div>
                        <div class="status">Flux en temps réel (MJPEG) - Port : {{ port }}</div>
                    </body>
                </html>
            ''', port=self.port)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    def update_frame(self, frame):
        """Met à jour le frame actuel pour le streaming."""
        with self.lock:
            if frame is not None:
                self.frame = frame.copy()

    def generate(self):
        """Générateur MJPEG."""
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
        """Lance le serveur Flask dans un thread séparé."""
        print(f"[*] Starting Web Server on http://0.0.0.0:{self.port}")
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self.app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False)

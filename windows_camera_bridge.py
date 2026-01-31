import cv2
from flask import Flask, Response
import time

# --- CONFIGURATION ---
CAMERA_INDEX = 0  # 0 pour la caméra par défaut
PORT = 5001       # Port 5001 pour ne pas entrer en conflit avec le Docker (5000)
# ---------------------

app = Flask(__name__)
camera = cv2.VideoCapture(CAMERA_INDEX)

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Réduire un peu la résolution pour le stream si nécessaire
            # frame = cv2.resize(frame, (640, 480))
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Windows Camera Bridge is Running</h1><p>Stream URL: <a href='/video_feed'>/video_feed</a></p>"

if __name__ == "__main__":
    print(f"[*] Starting bridge on http://localhost:{PORT}")
    print(f"[*] Access it from Docker using your Windows IP address.")
    app.run(host='0.0.0.0', port=PORT, threaded=True)

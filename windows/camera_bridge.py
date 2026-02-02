import cv2
from flask import Flask, Response
import socket
import logging

# Configuration
PORT = 5002
CAMERA_INDEX = 0

# Désactiver les logs Flask pour plus de clarté
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
camera = cv2.VideoCapture(CAMERA_INDEX)

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # n'a pas besoin d'être joignable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return f"<h1>Windows Camera Bridge</h1><p>Stream: <a href='/video_feed'>/video_feed</a></p>"

if __name__ == "__main__":
    ip = get_ip()
    print("\n" + "="*50)
    print("      WINDOWS CAMERA BRIDGE POUR DOCKER")
    print("="*50)
    print(f"\n[+] Serveur démarré sur : http://localhost:{PORT}")
    print(f"[+] Flux vidéo : http://localhost:{PORT}/video_feed")
    print(f"\n[!] COMMANDE A COPIER DANS LE TERMINAL WSL :")
    print(f"    ./run_app.sh http://{ip}:{PORT}/video_feed")
    print("\n" + "="*50)
    print("Laissez cette fenêtre ouverte pour streamer la caméra.")
    
    app.run(host='0.0.0.0', port=PORT, threaded=True)

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
camera = None # Sera initialisé après le choix de résolution

def select_resolution():
    # Sur Windows, CAP_DSHOW est souvent plus précis pour les réglages de résolution
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX) # Fallback
        
    if not cap.isOpened():
        print(f"[!] Erreur: Impossible d'ouvrir la caméra {CAMERA_INDEX}")
        return None

    # Liste des résolutions standards à tester
    candidates = [
        (3840, 2160, "4K / UltraHD"),
        (2560, 1440, "2K / QHD"),
        (1920, 1080, "1080p / FullHD"),
        (1280, 720,  "720p / HD"),
        (800, 600,   "SVGA"),
        (640, 480,   "VGA"),
    ]

    print("\n[i] Détection des formats supportés par votre caméra...")
    supported = []
    
    # On teste chaque résolution
    for w, h, name in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Si la caméra accepte ou se rapproche sans être déjà dans la liste
        res_key = (actual_w, actual_h)
        if res_key not in [s[0:2] for s in supported]:
            # On cherche si on a un nom pour cette résolution réelle
            label = next((c[2] for c in candidates if c[0] == actual_w and c[1] == actual_h), f"{actual_w}x{actual_h}")
            supported.append((actual_w, actual_h, label))

    # Tri par résolution décroissante
    supported.sort(key=lambda x: x[0], reverse=True)

    print("\n--- FORMATS DÉTECTÉS ---")
    for i, (w, h, label) in enumerate(supported, 1):
        print(f"{i}. {label} ({w}x{h})")
    print("0. Garder le réglage actuel")
    
    choice = input(f"\nChoisissez une option (0-{len(supported)}) [0]: ").strip() or "0"
    
    if choice != "0" and choice.isdigit() and int(choice) <= len(supported):
        w, h, name = supported[int(choice)-1]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        print(f"[+] Réglage validé sur {name}")
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Résolution active : {actual_w}x{actual_h}")
    return cap

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
    
    # Initialisation de la caméra
    camera = select_resolution()
    if camera is None:
        exit(1)

    print("\n" + "="*50)
    print("      WINDOWS CAMERA BRIDGE POUR DOCKER")
    print("="*50)
    print(f"\n[+] Serveur démarré sur : http://localhost:{PORT}")
    print(f"[+] Flux vidéo : http://localhost:{PORT}/video_feed")
    print(f"\n[!] COMMANDE A COPIER DANS LE TERMINAL WSL :")
    print(f"    ./run_app.sh http://{ip}:{PORT}/video_feed")
    print("\n" + "="*50)
    print("Laissez cette fenêtre ouverte pour streamer la caméra.")
    
    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True)
    finally:
        if camera:
            camera.release()

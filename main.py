import os
import sys
from camera_app_pipeline import CameraAppPipeline
from web_server import WebServer

if __name__ == "__main__":
    print("--- Starting PeopleCounter GPU (Web Mode) ---")
    
    # 1. Initialisation du serveur Web
    server = WebServer(port=5000)
    server.start()
    
    # 2. Initialisation du pipeline avec le callback pour le flux web
    # On récupère le mode de capture depuis l'environnement (défaut: usb)
    mode = os.environ.get('CAPTURE_MODE', 'usb')
    app = CameraAppPipeline(
        capture_mode=mode, 
        frame_callback=server.update_frame
    )
    
    # 3. Lancement du traitement
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nStopping application...")

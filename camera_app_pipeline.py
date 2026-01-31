import cv2
import sys
import screeninfo
import time
import csv
import os
import threading
import psutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from camera_capture import CameraCapture
from people_counter_processor import PeopleCounterProcessor
from yolo_people_counter import YoloPeopleCounter
from yolo_seg_people_counter import YoloSegPeopleCounter
from rtsp_capture import RTSPCapture
from mqtt_capture import MqttCapture

# --- Task for threading models ---
class ModelThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.duration = 0
    
    def run(self):
        start = time.time()
        self.result = self.func(*self.args, **self.kwargs)
        self.duration = time.time() - start

# Période de moyennage en secondes
AVERAGING_PERIOD_SECONDS = 2  # Réduit à 2s pour un graphe plus réactif

class CameraAppPipeline:
    def __init__(self, capture_mode=None, frame_callback=None):
        # capture_mode can be 'usb' (default), 'mqtt', or 'rtsp'
        cmode = capture_mode or os.environ.get('CAPTURE_MODE', 'usb')
        self.frame_callback = frame_callback
        self.use_gui = os.environ.get('USE_GUI', '0') == '1'
        
        # Common configuration from environment (can be overridden)
        cam_index = int(os.environ.get('CAMERA_INDEX', '0'))
        cam_width = int(os.environ.get('CAMERA_WIDTH', '1920'))
        cam_height = int(os.environ.get('CAMERA_HEIGHT', '1080'))
        broker_addr = os.environ.get('MQTT_BROKER', '127.0.0.1')
        broker_port = int(os.environ.get('MQTT_PORT', '1883'))
        mqtt_exposure = int(os.environ.get('MQTT_EXPOSURE', '500'))
        yolo_conf = float(os.environ.get('YOLO_CONF', '0.65'))
        self.yolo_conf = yolo_conf
        self.density_threshold = int(os.environ.get('DENSITY_THRESHOLD', '15'))
        self.denoise_strength = int(os.environ.get('DENOISE_STRENGTH', '0'))
        
        # Nouvelles options de Tiling via variables d'environnement
        # YOLO_TILING=1 pour activer (Pyramide 3 niveaux), actif par défaut pour max précision
        self.yolo_tiling = os.environ.get('YOLO_TILING', '1') == '1'
        # On active la segmentation par défaut si le modèle contient "seg" ou si YOLO_SEG=1
        yolo_model_env = os.environ.get('YOLO_MODEL', 'yolo26s-seg')
        self.yolo_seg = (os.environ.get('YOLO_SEG', '0') == '1') or ("-seg" in yolo_model_env.lower())
        # DENSITY_TILING=1 pour activer (divise en 4 quadrants), actif par défaut
        self.density_tiling = os.environ.get('DENSITY_TILING', '1') == '1'
        
        # Flag de Debug pour visualiser le tiling
        self.debug_tiling = (os.environ.get('DEBUG_TILING', '0') == '1')
        self.extreme_debug = (os.environ.get('EXTREME_DEBUG', '0') == '1')

        if cmode.lower() == 'mqtt':
            # Use MQTT-based Maixcam capture
            print("Using MqttCapture (Maixcam/MQTT)")
            self.capture = MqttCapture(broker_address=broker_addr, broker_port=broker_port, exposure=mqtt_exposure, width=cam_width, height=cam_height)
        elif cmode.lower() == 'rtsp':
            # Example: set RTSP_URL env var
            rtsp_url = os.environ.get('RTSP_URL', 'rtsp://192.168.1.95:8554/live')
            print(f"Using RTSPCapture ({rtsp_url})")
            self.capture = RTSPCapture(rtsp_url, width=cam_width, height=cam_height)
        else:
            # Default: USB camera via OpenCV
            print(f"Using CameraCapture (index={cam_index}, {cam_width}x{cam_height})")
            self.capture = CameraCapture(camera_index=cam_index, width=cam_width, height=cam_height, auto_exposure=1)

        lwcc_backend = os.environ.get('LWCC_BACKEND', 'torch').lower()
        openvino_device = os.environ.get('OPENVINO_DEVICE', 'GPU')
        yolo_backend = os.environ.get('YOLO_BACKEND', 'torch').lower()
        # On passe yolo26s-seg par défaut comme demandé par l'utilisateur
        yolo_model = os.environ.get('YOLO_MODEL', 'yolo26s-seg')

        # --- AUTO-EXPORT TENSORRT ---
        if yolo_backend == 'tensorrt_native':
            # Si c'est un nom court comme 'yolo26s-seg', on cherche dans models/
            base_model = yolo_model
            if not base_model.endswith('.engine'):
                engine_file = f"{base_model}.engine"
                engine_path = os.path.join(os.getcwd(), 'models', engine_file)
                
                if not os.path.isfile(engine_path):
                    print(f"⚠️ Engine TensorRT non trouvé : {engine_path}")
                    # On cherche le .pt correspondant
                    pt_model = f"{base_model}.pt"
                    pt_path = os.path.join(os.getcwd(), 'models', pt_model)
                    if os.path.isfile(pt_path):
                        print(f"🚀 Exportation automatique de {pt_model} vers TensorRT...")
                        # On utilise ultralytics pour l'export
                        try:
                            from ultralytics import YOLO
                            model = YOLO(pt_path)
                            # Export optimal pour le GPU actuel (RTX)
                            model.export(format='engine', dynamic=False, simplify=True, half=True)
                            # Ultralytics crée souvent le .engine à côté du .pt
                            generated_engine = pt_path.replace('.pt', '.engine')
                            if os.path.isfile(generated_engine):
                                if not os.path.exists(os.path.dirname(engine_path)):
                                    os.makedirs(os.path.dirname(engine_path))
                                if os.path.abspath(generated_engine) != os.path.abspath(engine_path):
                                    import shutil
                                    shutil.move(generated_engine, engine_path)
                                print(f"✅ Export réussi : {engine_path}")
                                yolo_model = engine_path
                        except Exception as e:
                            print(f"❌ Échec de l'export : {e}")
                    else:
                        print(f"❌ Impossible d'exporter : {pt_path} introuvable.")
                else:
                    yolo_model = engine_path
        # ----------------------------

        if yolo_backend in ('openvino', 'openvino_native'):
            ov_dir = os.environ.get('YOLO_OPENVINO_DIR')
            if ov_dir and os.path.isdir(ov_dir):
                yolo_model = ov_dir
            else:
                candidate_dir = f"{yolo_model}_openvino_model"
                if os.path.isdir(candidate_dir):
                    yolo_model = candidate_dir

        yolo_device = os.environ.get('YOLO_DEVICE')
        if not yolo_device:
            if yolo_backend == 'openvino_native':
                yolo_device = 'GPU'
            elif yolo_backend == 'tensorrt_native':
                yolo_device = 'cuda'
            else:
                yolo_device = 'cpu'

        self.processor = PeopleCounterProcessor(
            model_name="DM-Count",
            model_weights="QNRF",
            backend=lwcc_backend,
            openvino_device=openvino_device,
            density_threshold=self.density_threshold
        )
        if self.yolo_seg:
            self.yolo_counter = YoloSegPeopleCounter(
                model_path=yolo_model,
                confidence_threshold=self.yolo_conf,
                backend=yolo_backend,
                device=yolo_device
            )
        else:
            self.yolo_counter = YoloPeopleCounter(
                model_name=yolo_model,
                device=yolo_device,
                confidence_threshold=self.yolo_conf, 
                backend=yolo_backend
            )
        
        # Tentative de récupération des dimensions de l'écran, sinon défaut 1080p
        try:
            screen = screeninfo.get_monitors()[0]
            self.screen_width = screen.width
            self.screen_height = screen.height
        except Exception:
            self.screen_width = 1920
            self.screen_height = 1080

        # Création de la fenêtre uniquement si GUI activé
        if self.use_gui:
            cv2.namedWindow("People Count", cv2.WINDOW_NORMAL)
            init_w = int(self.screen_width * 0.75)
            init_h = int(self.screen_height * 0.75)
            cv2.resizeWindow("People Count", init_w, init_h)
            cv2.moveWindow("People Count", (self.screen_width - init_w) // 2, (self.screen_height - init_h) // 2)

        self.use_yolo = True  # Passe à True pour utiliser YOLO au lieu du modèle density
        self.yolo_tiling = os.environ.get('YOLO_TILING', '1') == '1' # Défaut à 1 (Tiling actif)
        self.history_len = 600  # Raw frames history (~3-5 mins at 2-3 FPS)
        self.history_data = [] # List of (yolo, density, total)

    def _draw_mini_graph(self, img):
        # Draw a small semi-transparent graph in the bottom left
        h, w = img.shape[:2]
        gw, gh = 450, 180
        margin_x = 20
        margin_y = 80 # Remonte le graphique pour éviter d'être coupé par la fenêtre
        x0, y0 = margin_x, h - margin_y - gh
        
        # Background box
        sub_img = img[y0:y0+gh, x0:x0+gw]
        rect = np.zeros_like(sub_img)
        cv2.rectangle(rect, (0, 0), (gw, gh), (30, 30, 30), -1)
        img[y0:y0+gh, x0:x0+gw] = cv2.addWeighted(sub_img, 0.4, rect, 0.6, 0)
        
        if len(self.history_data) < 2:
            cv2.putText(img, "Warming up graph...", (x0 + 10, y0 + gh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return
            
        # Scaling
        # points are (yolo, density, total)
        all_vals = []
        for d in self.history_data:
            all_vals.extend([d[0], d[1], d[2]])
            
        max_val = max(all_vals) if all_vals else 1
        curr_max = max_val
        if max_val < 5: max_val = 5 
        max_val *= 1.1 # Headroom
        
        num_points = len(self.history_data)
        # Use a fixed width distribution or scrolling? 
        # For a "live" feel, we spread samples across the 450px.
        dx = gw / (num_points - 1) if num_points > 1 else gw
        
        def to_py(val):
            return int(y0 + gh - (val / max_val * (gh - 40)) - 10)

        points_yolo = []
        points_density = []
        points_average = []
        
        for i, (y_val, d_val, a_val) in enumerate(self.history_data):
            px = int(x0 + i * dx)
            # Ajout d'offsets pour voir les 3 courbes si elles sont identiques
            points_yolo.append((px, to_py(y_val)))
            points_density.append((px, to_py(d_val) + 2))
            points_average.append((px, to_py(a_val) - 2))
            
        # Draw lines with distinct styles
        for i in range(len(points_yolo) - 1):
            # 1. Average (Cyan - Thickest)
            cv2.line(img, points_average[i], points_average[i+1], (255, 255, 0), 2)
            # 2. LWCC / Density (Red)
            cv2.line(img, points_density[i], points_density[i+1], (0, 0, 255), 1)
            # 3. YOLO (Green)
            cv2.line(img, points_yolo[i], points_yolo[i+1], (0, 255, 0), 1)
        
        # Legend and info
        cv2.putText(img, f"Live Activity (Max Sample: {int(curr_max)})", (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "YOLO", (x0 + 10, y0 + gh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(img, "LWCC (Density)", (x0 + 60, y0 + gh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, "AVERAGE", (x0 + 170, y0 + gh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def run(self):
        # Prépare le fichier CSV avec date et heure
        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f"people_count_{now_str}.csv"
        img_filename = f"people_count_{now_str}.png"
        csv_path = os.path.join(os.getcwd(), csv_filename)
        img_path = os.path.join(os.getcwd(), img_filename)
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["datetime", "YOLO_count", "Density_count", "Avg_combined", "Max_count"])
            counts_yolo = []
            counts_density = []
            last_period = int(time.time())
            log_data = []
            try:
                while True:
                    start_time = time.time()
                    # Retry mechanism for frame acquisition
                    max_retries = 3
                    retry_count = 0
                    frame = self.capture.get_frame()
                    while frame is None and retry_count < max_retries:
                        print(f"[WARNING] Frame is None. Tentative {retry_count+1}/{max_retries}...")
                        time.sleep(1)
                        frame = self.capture.get_frame()
                        retry_count += 1
                    if frame is None:
                        print("[ERROR] Frame is None. Vérifiez la connexion de la caméra et ses paramètres.")
                        print(f"Camera opened: {self.capture.is_opened}")
                        break
                    # print(f"Frame shape: {frame.shape}")
                    
                    # --- FILTRAGE DU BRUIT (Optionnel) ---
                    if self.denoise_strength > 0:
                        # On utilise un flou Gaussien léger pour lisser le bruit de capteur
                        # kernel_size doit être impair
                        k = (self.denoise_strength * 2) + 1
                        frame = cv2.GaussianBlur(frame, (k, k), 0)
                    
                    # Parallelizing YOLO and Density models
                    t_loop_start = time.time()
                    
                    # 1. YOLO INFERENCE (Tiling optionale)
                    thr_yolo = ModelThread(
                        self.yolo_counter.count_people, 
                        frame, 
                        tile_size=640, 
                        draw_boxes=True, 
                        use_tiling=self.yolo_tiling,
                        draw_tiles=self.debug_tiling
                    )

                    # 2. DENSITY INFERENCE (Tiling 2x2 Batché)
                    t_loop_start = time.time()
                    self.density_quads_coords = []
                    if self.density_tiling:
                        # On prépare les quadrants
                        h, w = frame.shape[:2]
                        mid_h, mid_w = h // 2, w // 2
                        quads = [
                            frame[0:mid_h, 0:mid_w],
                            frame[0:mid_h, mid_w:w],
                            frame[mid_h:h, 0:mid_w],
                            frame[mid_h:h, mid_w:w]
                        ]
                        self.density_quads_coords = [
                            (0, 0, mid_w, mid_h),
                            (mid_w, 0, w, mid_h),
                            (0, mid_h, mid_w, h),
                            (mid_w, mid_h, w, h)
                        ]
                        thr_density = ModelThread(self.processor.process_batch, quads)
                    else:
                        # Mode standard (Inférence globale redimensionnée)
                        thr_density = ModelThread(self.processor.process, frame)
                    
                    # On lance les deux threads
                    t_cycle_start = time.time()
                    thr_yolo.start()
                    thr_density.start()
                    
                    # Attente YOLO
                    thr_yolo.join()
                    if thr_yolo.result is not None:
                        yolo_count, frame_with_bbox = thr_yolo.result
                    else:
                        print("[ERROR] YOLO thread returned None.")
                        yolo_count, frame_with_bbox = 0, frame.copy()
                    
                    # Attente Density
                    thr_density.join()

                    # Debug Tiling Densité (Magenta)
                    if self.debug_tiling and self.density_tiling and hasattr(self, 'density_quads_coords'):
                        for x1, y1, x2, y2 in self.density_quads_coords:
                            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Récupération des temps d'exécution réels de chaque moteur via le Thread
                    yolo_time = thr_yolo.duration
                    density_time = thr_density.duration
                    
                    # Récupération des devices utilisés
                    yolo_dev = getattr(self.yolo_counter, 'last_device', 'GPU')
                    dens_dev = getattr(self.processor, 'last_device', 'NPU')
                    
                    cpu_usage = psutil.cpu_percent()
                    
                    # --- OPTIMISATION RENDU : Travail en 1080p pour libérer le CPU ---
                    # Faire du 4K bitwise/dilate/contours tue les perfs, on passe en 1080p pour l'affichage
                    h_4k, w_4k = frame_with_bbox.shape[:2]
                    render_h, render_w = 1080, 1920
                    if h_4k < render_h: render_h, render_w = h_4k, w_4k
                    
                    frame_render = cv2.resize(frame_with_bbox, (render_w, render_h))
                    density_raw_small = np.zeros((render_h, render_w, 3), dtype=np.uint8)
                    density_mask_small = np.zeros((render_h, render_w), dtype=np.uint8)
                    mid_h, mid_w = render_h // 2, render_w // 2

                    if self.density_tiling and thr_density.result is not None:
                        _, quad_res_color, quad_res_counts, quad_res_masks = thr_density.result
                        # On utilise sum sur les flottants pour éviter la troncature
                        density_count = sum(float(c) for c in quad_res_counts)
                        
                        # Stitching direct en 1080p
                        density_raw_small[0:mid_h, 0:mid_w] = cv2.resize(quad_res_color[0], (mid_w, mid_h))
                        density_raw_small[0:mid_h, mid_w:render_w] = cv2.resize(quad_res_color[1], (render_w-mid_w, mid_h))
                        density_raw_small[mid_h:render_h, 0:mid_w] = cv2.resize(quad_res_color[2], (mid_w, render_h-mid_h))
                        density_raw_small[mid_h:render_h, mid_w:render_w] = cv2.resize(quad_res_color[3], (render_w-mid_w, render_h-mid_h))
                        
                        density_mask_small[0:mid_h, 0:mid_w] = cv2.resize(quad_res_masks[0], (mid_w, mid_h), interpolation=cv2.INTER_NEAREST)
                        density_mask_small[0:mid_h, mid_w:render_w] = cv2.resize(quad_res_masks[1], (render_w-mid_w, mid_h), interpolation=cv2.INTER_NEAREST)
                        density_mask_small[mid_h:render_h, 0:mid_w] = cv2.resize(quad_res_masks[2], (mid_w, render_h-mid_h), interpolation=cv2.INTER_NEAREST)
                        density_mask_small[mid_h:render_h, mid_w:render_w] = cv2.resize(quad_res_masks[3], (render_w-mid_w, render_h-mid_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        res = thr_density.result
                        if res is not None:
                            _, dens_raw, density_count, dens_mask = res
                            density_raw_small = cv2.resize(dens_raw, (render_w, render_h))
                            density_mask_small = cv2.resize(dens_mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            density_count = 0
                    
                    # Affichage des métriques d'instrumentation si extreme_debug est actif
                    if self.extreme_debug:
                        yp = getattr(self.yolo_counter.engine, 'last_perf', {}) if hasattr(self.yolo_counter, 'engine') else {}
                        dp = getattr(self.processor, 'last_perf', {})
                        capture_time = getattr(self.capture, 'last_capture_time', 0.0)
                        
                        print(f"[DEBUG] PERF BREAKDOWN (ms):")
                        print(f"  Capture: {capture_time*1000:.1f}ms")
                        print(f"  YOLO {self.yolo_counter.model_name}: Pre={yp.get('preprocess',0)*1000:.1f} | GPU={yp.get('infer',0)*1000:.1f} | Post={yp.get('postprocess',0)*1000:.1f}")
                        print(f"  DENS {self.processor.model_name}: Pre={dp.get('preprocess',0)*1000:.1f} | GPU={dp.get('infer',0)*1000:.1f} | Post={dp.get('postprocess',0)*1000:.1f}")

                    total_process_time = time.time() - t_loop_start
                    
                    # Superposition Heatmap optimisée (en 1080p)
                    mask_3ch = cv2.cvtColor(density_mask_small, cv2.COLOR_GRAY2BGR)
                    heatmap_on_black = cv2.bitwise_and(density_raw_small, mask_3ch)
                    overlay2 = cv2.addWeighted(frame_render, 1.0, heatmap_on_black, 0.6, 0)
                    
                    # --- RENDU DES "OREILLES ROUGES" (HOTSPOTS) ---
                    # On détecte les pics de densité pour alerter l'utilisateur
                    # Utilisation d'un noyau plus petit (5x5) pour être plus précis et moins "nonsense"
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    density_mask_dilated = cv2.dilate(density_mask_small, kernel_dilate, iterations=1)
                    
                    contours, _ = cv2.findContours(density_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        # Uniquement les hotspots significatifs
                        if cv2.contourArea(cnt) > 30: 
                            (x, y), radius = cv2.minEnclosingCircle(cnt)
                            # Dessine des petits cercles rouges discrets
                            cv2.circle(overlay2, (int(x), int(y)), int(radius) + 2, (0, 0, 255), 1)

                    # Affiche les compteurs avec un style plus propre
                    cv2.putText(overlay2, f'Density Count: {density_count:.1f}', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2)
                    yolo_label = self.yolo_counter.model_name
                    cv2.putText(overlay2, f'YOLO ({yolo_label}): {int(yolo_count)}', (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)
                    
                    if log_data:
                        last_avg = log_data[-1][3]
                        cv2.putText(overlay2, f'Avg ({AVERAGING_PERIOD_SECONDS}s): {last_avg:.1f}', (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 2)

                    # --- GESTION DE LA FENÊTRE D'AFFICHAGE ET CALLBACK ---
                    if self.use_gui:
                        try:
                            win_rect = cv2.getWindowImageRect("People Count")
                            win_w, win_h = (win_rect[2], win_rect[3]) if win_rect and win_rect[2] > 0 else (1280, 720)
                        except Exception:
                            win_w, win_h = 1280, 720
                    else:
                        win_w, win_h = 1280, 720

                    h_curr, w_curr = overlay2.shape[:2]
                    scale = min(win_w / w_curr, win_h / h_curr)
                    target_w, target_h = int(w_curr * scale), int(h_curr * scale)
                    
                    preview_frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    resized = cv2.resize(overlay2, (target_w, target_h))
                    y_off, x_off = (win_h - target_h) // 2, (win_w - target_w) // 2
                    preview_frame[y_off:y_off+target_h, x_off:x_off+target_w] = resized
                    
                    self._draw_mini_graph(preview_frame)

                    # Envoi au serveur Web si actif
                    if self.frame_callback:
                        self.frame_callback(preview_frame)
                    
                    total_time = time.time() - start_time
                    fps = 1.0 / total_time if total_time > 0 else 0
                    
                    # Console logs plus détaillés avec devices
                    print(f"Metrics: YOLO={yolo_count} ({yolo_time:.3f}s on {yolo_dev}) | "
                        f"Density={density_count:.1f} ({density_time:.3f}s on {dens_dev}) | "
                        f"CPU: {cpu_usage}% | Total: {total_time:.3f}s | FPS: {fps:.2f}")

                    cv2.putText(preview_frame, f'FPS: {fps:.2f}', (win_w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Petit texte en bas pour indiquer les devices
                    cv2.putText(preview_frame, f'YOLO: {yolo_dev}', (20, win_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(preview_frame, f'DENS: {dens_dev}', (20, win_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    if self.use_gui:
                        cv2.imshow("People Count", preview_frame)
                    
                    # Console logs propres
                    capture_time = getattr(self.capture, 'last_capture_time', 0.0)
                    print(f"Metrics: YOLO={yolo_count} ({yolo_time:.3f}s) | Density={density_count:.1f} ({density_time:.3f}s) | "
                          f"Capture={capture_time:.3f}s | CPU: {cpu_usage}% | Total: {total_process_time:.3f}s | FPS: {fps:.2f}")
                    
                    # Update mini-graph history at every frame for live feedback (YOLO, Density, Average)
                    avg_instant = (yolo_count + density_count) / 2.0
                    self.history_data.append((yolo_count, density_count, avg_instant))
                    if len(self.history_data) > self.history_len:
                        self.history_data.pop(0)

                    counts_yolo.append(yolo_count)
                    counts_density.append(density_count)
                    now_sec = int(time.time())
                    if now_sec - last_period >= AVERAGING_PERIOD_SECONDS:
                        if counts_yolo and counts_density:
                            avg_yolo = sum(counts_yolo) / len(counts_yolo)
                            avg_density = sum(counts_density) / len(counts_density)
                            avg_combined = (avg_yolo + avg_density) / 2.0
                            max_count = max(avg_yolo, avg_density)
                            now_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(f"Average YOLO ({yolo_label}) People (last {AVERAGING_PERIOD_SECONDS} sec): {avg_yolo:.2f}")
                            print(f"Average Density People (last {AVERAGING_PERIOD_SECONDS} sec): {avg_density:.2f}")
                            print(f"Average Combined (last {AVERAGING_PERIOD_SECONDS} sec): {avg_combined:.2f}")
                            print(f"Max People Count (last {AVERAGING_PERIOD_SECONDS} sec): {max_count:.2f}")
                            writer.writerow([now_dt, f"{avg_yolo:.2f}", f"{avg_density:.2f}", f"{avg_combined:.2f}", f"{max_count:.2f}"])
                            csvfile.flush()
                            log_data.append((now_dt, avg_yolo, avg_density, avg_combined, max_count))
                        
                        counts_yolo = []
                        counts_density = []
                        last_period = now_sec
                    
                    if self.use_gui:
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord('q') or cv2.getWindowProperty("People Count", cv2.WND_PROP_VISIBLE) < 1:
                            break
                    else:
                        # En mode headless, on ajoute un petit sleep pour ne pas saturer le CPU si FPS > 100
                        time.sleep(0.01)
            finally:
                self.capture.release()
                if self.use_gui:
                    cv2.destroyAllWindows()
                print("Camera released and application closed.")
                # Génère la courbe à partir du CSV
                if log_data:
                    times = [dt for dt, _, _, _, _ in log_data]
                    yolo_vals = [y for _, y, _, _, _ in log_data]
                    density_vals = [d for _, _, d, _, _ in log_data]
                    avg_combined_vals = [a for _, _, _, a, _ in log_data]
                    
                    plt.figure(figsize=(14, 7))
                    plt.plot(times, yolo_vals, marker='o', markersize=4, label='YOLO', color='green', alpha=0.6)
                    plt.plot(times, density_vals, marker='x', markersize=4, label='Density', color='red', alpha=0.6)
                    plt.plot(times, avg_combined_vals, marker='d', markersize=4, label='Average (Combined)', color='cyan', linewidth=2)
                    
                    # Affiche un label de temps toutes les N points pour éviter la surcharge
                    N = max(1, len(times)//20)
                    plt.xticks([i for i in range(0, len(times), N)], [times[i] for i in range(0, len(times), N)], rotation=45)
                    plt.xlabel('Time')
                    plt.ylabel('Average People Count')
                    plt.title('People Count Over Time')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(img_path)
                    print(f"Saved plot to {img_path}")

if __name__ == "__main__":
    sys.path.append(r"E:\AI\lwcc")
    # Allow override by environment variable CAPTURE_MODE or by CLI first arg
    capture_mode = None
    if len(sys.argv) > 1:
        capture_mode = sys.argv[1]
    app = CameraAppPipeline(capture_mode=capture_mode)
    app.run()

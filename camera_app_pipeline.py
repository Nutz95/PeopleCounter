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
from rtsp_capture import RTSPCapture
from mqtt_capture import MqttCapture
from pipeline_components import MetricsCollector, ModelLoader, ProfileManager

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
    def __init__(self, capture_mode=None, frame_callback=None, metrics_callback=None):
        cmode = (capture_mode or os.environ.get('CAPTURE_MODE', 'usb')).lower()
        self.capture_mode = cmode
        self.frame_callback = frame_callback
        self.metrics_callback = metrics_callback
        self.use_gui = os.environ.get('USE_GUI', '0') == '1'
        self.graph_overlay_enabled = os.environ.get('GRAPH_OVERLAY', '1') == '1'

        cam_index = int(os.environ.get('CAMERA_INDEX', '0'))
        cam_width = int(os.environ.get('CAMERA_WIDTH', '1920'))
        cam_height = int(os.environ.get('CAMERA_HEIGHT', '1080'))
        broker_addr = os.environ.get('MQTT_BROKER', '127.0.0.1')
        broker_port = int(os.environ.get('MQTT_PORT', '1883'))
        mqtt_exposure = int(os.environ.get('MQTT_EXPOSURE', '500'))
        self.yolo_conf = float(os.environ.get('YOLO_CONF', '0.65'))
        self.density_threshold = int(os.environ.get('DENSITY_THRESHOLD', '15'))
        self.denoise_strength = int(os.environ.get('DENOISE_STRENGTH', '0'))
        self.yolo_tiling = os.environ.get('YOLO_TILING', '1') == '1'
        self.density_tiling = os.environ.get('DENSITY_TILING', '1') == '1'
        self.debug_tiling = os.environ.get('DEBUG_TILING', '0') == '1'
        self.extreme_debug = os.environ.get('EXTREME_DEBUG', '0') == '1'

        if cmode == 'mqtt':
            print("Using MqttCapture (Maixcam/MQTT)")
            self.capture = MqttCapture(broker_address=broker_addr, broker_port=broker_port, exposure=mqtt_exposure, width=cam_width, height=cam_height)
        elif cmode == 'rtsp':
            rtsp_url = os.environ.get('RTSP_URL', 'rtsp://192.168.1.95:8554/live')
            print(f"Using RTSPCapture ({rtsp_url})")
            self.capture = RTSPCapture(rtsp_url, width=cam_width, height=cam_height)
        else:
            print(f"Using CameraCapture (index={cam_index}, {cam_width}x{cam_height})")
            self.capture = CameraCapture(camera_index=cam_index, width=cam_width, height=cam_height, auto_exposure=1)

        self.profile_manager = ProfileManager()
        self.profile_manager.ensure_loaded()
        self.active_profile_name = self.profile_manager.active_profile_name or ''
        self.available_profiles = list(self.profile_manager.available_profiles)
        self.model_loader = ModelLoader(yolo_conf=self.yolo_conf, density_threshold=self.density_threshold)
        self.metrics_collector = MetricsCollector(history_len=600)
        self.profile_active = False
        self.profile_log = []
        self.latest_metrics = {}
        self.history_data = []
        self.history_len = 600
        self.yolo_pipeline_mode = os.environ.get('YOLO_PIPELINE_MODE', 'auto').lower()
        self.density_quads_coords = []
        self.yolo_device = 'cpu'
        self.density_device = 'GPU'
        self.processor = None
        self.yolo_counter = None
        self.latest_pipeline_report = {}

        self._apply_profile_settings(self.profile_manager.profile_settings)
        if not self._refresh_models(force=True):
            raise RuntimeError(f"[FATAL] Impossible d'initialiser les moteurs pour "
                               f"le profil '{self.active_profile_name or 'Live Metrics'}'.")

        try:
            screen = screeninfo.get_monitors()[0]
            self.screen_width = screen.width
            self.screen_height = screen.height
        except Exception:
            self.screen_width = 1920
            self.screen_height = 1080

        if self.use_gui:
            cv2.namedWindow("People Count", cv2.WINDOW_NORMAL)
            init_w = int(self.screen_width * 0.75)
            init_h = int(self.screen_height * 0.75)
            cv2.resizeWindow("People Count", init_w, init_h)
            cv2.moveWindow("People Count", (self.screen_width - init_w) // 2, (self.screen_height - init_h) // 2)

        self.use_yolo = True

    def _apply_profile_settings(self, settings):
        self.yolo_tiling = settings.get('YOLO_TILING', '1') == '1'
        self.density_tiling = settings.get('DENSITY_TILING', '1') == '1'
        self.debug_tiling = settings.get('DEBUG_TILING', '0') == '1'
        self.extreme_debug = settings.get('EXTREME_DEBUG', '0') == '1'
        self.yolo_conf = float(settings.get('YOLO_CONF', self.yolo_conf))
        self.density_threshold = int(settings.get('DENSITY_THRESHOLD', self.density_threshold))
        self.denoise_strength = int(settings.get('DENOISE_STRENGTH', self.denoise_strength))
        self.model_loader.yolo_conf = self.yolo_conf
        self.model_loader.density_threshold = self.density_threshold

    def _refresh_models(self, force=False):
        if not force and getattr(self, '_models_loaded', False):
            return False
        settings = self.profile_manager.profile_settings
        try:
            yolo_counter, processor, metadata = self.model_loader.load(settings)
        except Exception as exc:
            print(f"[WARN] Failed to reload models: {exc}")
            return False
        if yolo_counter is None or processor is None:
            print("[WARN] Model loader returned empty handles, keeping previous engines.")
            return False
        self.yolo_counter = yolo_counter
        self.processor = processor
        self.yolo_device = metadata.get('device', getattr(self.yolo_counter, 'last_device', 'cpu'))
        self.density_device = metadata.get('processor_device', getattr(self.processor, 'last_device', 'GPU'))
        self.model_metadata = metadata
        self._models_loaded = True
        self.set_yolo_pipeline_mode(self.yolo_pipeline_mode)
        if self.yolo_counter and hasattr(self.yolo_counter, 'set_debug_mode'):
            self.yolo_counter.set_debug_mode(self.extreme_debug)
        return True

    def apply_profile(self, profile_name):
        applied = self.profile_manager.apply_profile(profile_name)
        self.active_profile_name = self.profile_manager.pending_profile_name or ''
        return applied

    def set_graph_overlay(self, enabled):
        self.graph_overlay_enabled = bool(enabled)

    def attach_metrics_callback(self, callback):
        self.metrics_callback = callback

    def set_debug_mode(self, enabled):
        self.extreme_debug = bool(enabled)
        if self.yolo_counter and hasattr(self.yolo_counter, 'set_debug_mode'):
            self.yolo_counter.set_debug_mode(self.extreme_debug)

    def set_yolo_pipeline_mode(self, mode):
        normalized = (mode or 'auto').lower()
        if normalized not in ('auto', 'cpu', 'gpu', 'gpu_full'):
            return False
        self.yolo_pipeline_mode = normalized
        if self.yolo_counter and hasattr(self.yolo_counter, 'configure_pipeline'):
            self.yolo_counter.configure_pipeline(normalized)
        return True

    def set_profile_logging(self, active):
        self.profile_active = bool(active)

    def clear_profile_log(self):
        self.profile_log.clear()

    def get_profile_log(self, limit=40):
        return list(self.profile_log[-limit:])

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
                    if self.profile_manager.pending_profile_reload:
                        forced = self.profile_manager.ensure_loaded()
                        if forced:
                            self._apply_profile_settings(self.profile_manager.profile_settings)
                            if not self._refresh_models(force=True):
                                print("[ERROR] Impossible de recharger les modèles YOLO/Density, arrêt.")
                                break
                    self.active_profile_name = self.profile_manager.active_profile_name or 'Live Metrics'
                    preprocess_duration = 0.0
                    postprocess_duration = 0.0
                    
                    # --- FILTRAGE DU BRUIT (Optionnel) ---
                    if self.denoise_strength > 0:
                        # On utilise un flou Gaussien léger pour lisser le bruit de capteur
                        # kernel_size doit être impair
                        k = (self.denoise_strength * 2) + 1
                        frame = cv2.GaussianBlur(frame, (k, k), 0)
                    
                    # Parallelizing YOLO and Density models
                    t_loop_start = time.time()

                    if not self.yolo_counter:
                        print("[ERROR] Le compteur YOLO n'est pas initialisé, arrêt du pipeline.")
                        break
                    if not self.processor:
                        print("[ERROR] Le processeur density n'est pas initialisé, arrêt du pipeline.")
                        break
                    
                    # 1. YOLO INFERENCE (Tiling optionale)
                    preprocess_start = time.monotonic()
                    thr_yolo = ModelThread(
                        self.yolo_counter.count_people, 
                        frame, 
                        tile_size=640, 
                        draw_boxes=True, 
                        use_tiling=self.yolo_tiling,
                        draw_tiles=self.debug_tiling
                    )

                    yolo_label = getattr(self.yolo_counter, 'model_name', 'YOLO')

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

                    preprocess_duration = time.monotonic() - preprocess_start
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
                    pipeline_report = {}
                    if self.yolo_counter and hasattr(self.yolo_counter, 'get_pipeline_report'):
                        pipeline_report = self.yolo_counter.get_pipeline_report() or {}
                        self.latest_pipeline_report = pipeline_report
                        self.latest_metrics['yolo_pipeline_mode'] = pipeline_report.get('active_mode', 'unknown')
                        if self.extreme_debug and pipeline_report:
                            print(f"[DEBUG] YOLO pipeline report: {pipeline_report}")
                    
                    # Attente Density
                    thr_density.join()

                    postprocess_start = time.monotonic()

                    # Debug Tiling Densité (Magenta)
                    if self.debug_tiling and self.density_tiling and hasattr(self, 'density_quads_coords'):
                        for x1, y1, x2, y2 in self.density_quads_coords:
                            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Récupération des temps d'exécution réels de chaque moteur via le Thread
                    yolo_time = thr_yolo.duration
                    density_time = thr_density.duration
                    
                    # Récupération des devices utilisés
                    yolo_dev = getattr(self.yolo_counter, 'last_device', 'GPU')
                    dens_dev = getattr(self.processor, 'last_device', 'GPU')
                    yp = getattr(self.yolo_counter, 'last_perf', {}) if self.yolo_counter else {}
                    yolo_internal = getattr(self.yolo_counter, 'last_internal', {}) if self.yolo_counter else {}
                    dp = getattr(self.processor, 'last_perf', {}) if self.processor else {}
                    
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
                    
                    capture_time = getattr(self.capture, 'last_capture_time', 0.0)
                    # Affichage des métriques d'instrumentation si extreme_debug est actif
                    if self.extreme_debug:
                        yp_engine = getattr(self.yolo_counter, 'engine', None)
                        yp_debug = getattr(yp_engine, 'last_perf', {}) if yp_engine else yp
                        dp_debug = getattr(self.processor, 'last_perf', {})
                        
                        print(f"[DEBUG] PERF BREAKDOWN (ms):")
                        print(f"  Capture: {capture_time*1000:.1f}ms")
                        print(f"  YOLO {self.yolo_counter.model_name}: Pre={yp_debug.get('preprocess',0)*1000:.1f} | GPU={yp_debug.get('infer',0)*1000:.1f} | Post={yp_debug.get('postprocess',0)*1000:.1f}")
                        print(f"  DENS {self.processor.model_name}: Pre={dp_debug.get('preprocess',0)*1000:.1f} | GPU={dp_debug.get('infer',0)*1000:.1f} | Post={dp_debug.get('postprocess',0)*1000:.1f}")

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
                    
                    # Envoi au serveur Web si actif
                    if self.frame_callback:
                        self.frame_callback(preview_frame)
                    
                    total_time = time.time() - start_time
                    fps = 1.0 / total_time if total_time > 0 else 0
                    avg_instant = (yolo_count + density_count) / 2.0
                    self.metrics_collector.append(yolo_count, density_count, avg_instant)
                    history_snapshot = self.metrics_collector.snapshot() if self.graph_overlay_enabled else []
                    metrics = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "yolo_count": int(yolo_count),
                        "density_count": float(density_count),
                        "average": float(avg_instant),
                        "fps": float(fps),
                        "cpu_usage": float(cpu_usage),
                        "total_time": float(total_process_time),
                        "yolo_time": float(yolo_time),
                        "density_time": float(density_time),
                        "yolo_device": self.yolo_device,
                        "density_device": self.density_device,
                        "debug_mode": bool(self.extreme_debug),
                        "profile_active": bool(self.profile_active),
                        "profile_name": self.active_profile_name or 'Live Metrics',
                        "graph_overlay": bool(self.graph_overlay_enabled),
                        "capture_ms": float(capture_time * 1000),
                        "pipeline_pre_ms": float(preprocess_duration * 1000),
                        "pipeline_post_ms": float(postprocess_duration * 1000),
                        "yolo_pre_ms": float(yp.get('preprocess', 0) * 1000),
                        "yolo_gpu_ms": float(yp.get('infer', 0) * 1000),
                        "yolo_post_ms": float(yp.get('postprocess', 0) * 1000),
                        "yolo_pipeline_mode": self.yolo_pipeline_mode,
                        "yolo_pipeline_mode_effective": getattr(self.yolo_counter, 'active_pipeline_mode', self.yolo_pipeline_mode),
                        "yolo_preproc_mode": getattr(self.yolo_counter, 'preprocessor_mode', 'cpu'),
                        "yolo_renderer_mode": getattr(self.yolo_counter, 'renderer_mode', 'cpu'),
                        "yolo_internal_inf_ms": float(yolo_internal.get('t_inf', 0) * 1000),
                        "yolo_internal_draw_ms": float(yolo_internal.get('t_draw', 0) * 1000),
                        "yolo_internal_total_ms": float(yolo_internal.get('total_internal', 0) * 1000),
                        "density_pre_ms": float(dp.get('preprocess', 0) * 1000),
                        "density_gpu_ms": float(dp.get('infer', 0) * 1000),
                        "density_post_ms": float(dp.get('postprocess', 0) * 1000),
                        "history": history_snapshot
                    }
                    self.latest_metrics = metrics
                    if self.profile_active:
                        self.profile_log.append(metrics.copy())
                        if len(self.profile_log) > 300:
                            self.profile_log.pop(0)
                    if self.metrics_callback:
                        try:
                            self.metrics_callback(metrics)
                        except Exception as exc:
                            print(f"[WARN] Metrics callback failed: {exc}")
                    
                    # Console logs plus détaillés avec devices
                    print(f"Metrics: YOLO={yolo_count} ({yolo_time:.3f}s on {yolo_dev}) | "
                        f"Density={density_count:.1f} ({density_time:.3f}s on {dens_dev}) | "
                        f"CPU: {cpu_usage}% | Total: {total_time:.3f}s | FPS: {fps:.2f}")

                    
                    if self.use_gui:
                        cv2.imshow("People Count", preview_frame)

                    postprocess_duration = time.monotonic() - postprocess_start
                    
                    # Console logs propres
                    capture_time = getattr(self.capture, 'last_capture_time', 0.0)
                    print(f"Metrics: YOLO={yolo_count} ({yolo_time:.3f}s) | Density={density_count:.1f} ({density_time:.3f}s) | "
                          f"Capture={capture_time:.3f}s | CPU: {cpu_usage}% | Total: {total_process_time:.3f}s | FPS: {fps:.2f}")
                    
                    # Update mini-graph history at every frame for live feedback (YOLO, Density, Average)
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

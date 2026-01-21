import cv2
import sys
import screeninfo
import time
import csv
import os
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from camera_capture import CameraCapture
from people_counter_processor import PeopleCounterProcessor
from yolo_people_counter import YoloPeopleCounter
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
    
    def run(self):
        self.result = self.func(*self.args, **self.kwargs)

# Période de moyennage en secondes
AVERAGING_PERIOD_SECONDS = 2  # Réduit à 2s pour un graphe plus réactif

class CameraAppPipeline:
    def __init__(self, capture_mode=None):
        # capture_mode can be 'usb' (default), 'mqtt', or 'rtsp'
        cmode = capture_mode or os.environ.get('CAPTURE_MODE', 'usb')
        # Common configuration from environment (can be overridden)
        cam_index = int(os.environ.get('CAMERA_INDEX', '0'))
        cam_width = int(os.environ.get('CAMERA_WIDTH', '1920'))
        cam_height = int(os.environ.get('CAMERA_HEIGHT', '1080'))
        broker_addr = os.environ.get('MQTT_BROKER', '127.0.0.1')
        broker_port = int(os.environ.get('MQTT_PORT', '1883'))
        mqtt_exposure = int(os.environ.get('MQTT_EXPOSURE', '500'))

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
        yolo_model = os.environ.get('YOLO_MODEL', 'yolo11n')

        if yolo_backend in ('openvino', 'openvino_native'):
            ov_dir = os.environ.get('YOLO_OPENVINO_DIR')
            if ov_dir and os.path.isdir(ov_dir):
                yolo_model = ov_dir
            else:
                candidate_dir = f"{yolo_model}_openvino_model"
                if os.path.isdir(candidate_dir):
                    yolo_model = candidate_dir
        elif yolo_backend == 'tensorrt_native':
            if not yolo_model.endswith('.engine'):
                engine_file = f"{yolo_model}.engine"
                if os.path.isfile(engine_file):
                    yolo_model = engine_file
        
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
            openvino_device=openvino_device
        )
        self.yolo_counter = YoloPeopleCounter(
            model_name=yolo_model,
            device=yolo_device,
            confidence_threshold=0.7,
            backend=yolo_backend
        )  # Modifiez le seuil ici
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height
        self.use_yolo = True  # Passe à True pour utiliser YOLO au lieu du modèle density
        self.yolo_tiling = os.environ.get('YOLO_TILING', '1') == '1' # Défaut à 1 (Tiling actif)
        self.history_len = 600  # Raw frames history (~3-5 mins at 2-3 FPS)
        self.history_data = [] # List of (yolo, density, total)

    def _draw_mini_graph(self, img):
        # Draw a small semi-transparent graph in the bottom left
        h, w = img.shape[:2]
        gw, gh = 450, 180
        margin = 20
        x0, y0 = margin, h - margin - gh
        
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
        points_total = []
        
        for i, (y_val, d_val, t_val) in enumerate(self.history_data):
            px = int(x0 + i * dx)
            points_yolo.append((px, to_py(y_val)))
            points_density.append((px, to_py(d_val)))
            points_total.append((px, to_py(t_val)))
            
        # Draw lines with distinct styles
        for i in range(len(points_yolo) - 1):
            # YOLO (Green)
            cv2.line(img, points_yolo[i], points_yolo[i+1], (0, 255, 0), 1)
            # Density (Red)
            cv2.line(img, points_density[i], points_density[i+1], (0, 0, 255), 1)
            # Total (Yellow - Thicker and slightly higher to avoid overlap hiding)
            # We offset the total line by 2 pixels up if it matches the others
            p1_t = (points_total[i][0], points_total[i][1] - 1)
            p2_t = (points_total[i+1][0], points_total[i+1][1] - 1)
            cv2.line(img, p1_t, p2_t, (0, 255, 255), 2)
        
        # Legend and info
        cv2.putText(img, f"Live Activity (Max: {int(curr_max)})", (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "YOLO", (x0 + 10, y0 + gh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(img, "LWCC", (x0 + 60, y0 + gh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img, "TOTAL", (x0 + 120, y0 + gh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def run(self):
        # Prépare le fichier CSV avec date et heure
        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f"people_count_{now_str}.csv"
        img_filename = f"people_count_{now_str}.png"
        csv_path = os.path.join(os.getcwd(), csv_filename)
        img_path = os.path.join(os.getcwd(), img_filename)
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["datetime", "YOLO_count", "Density_count", "Max_count"])
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
                    
                    # Parallelizing YOLO and Density models
                    t_loop_start = time.time()
                    
                    thr_yolo = ModelThread(
                        self.yolo_counter.count_people, 
                        frame, 
                        tile_size=640, 
                        draw_boxes=True, 
                        use_tiling=self.yolo_tiling
                    )
                    thr_density = ModelThread(self.processor.process, frame)
                    
                    thr_yolo.start()
                    thr_density.start()
                    
                    thr_yolo.join()
                    thr_density.join()
                    
                    yolo_count, frame_with_bbox = thr_yolo.result
                    _, density_overlay, density_count = thr_density.result
                    
                    t_loop_end = time.time()
                    total_process_time = t_loop_end - t_loop_start
                    
                    # Resize density_overlay to match frame_with_bbox shape
                    density_overlay = cv2.resize(density_overlay, (frame_with_bbox.shape[1], frame_with_bbox.shape[0]))
                    # Superpose la carte de densité sur l'image avec bbox
                    overlay2 = cv2.addWeighted(frame_with_bbox, 0.7, density_overlay, 0.3, 0)
                    # Affiche les deux compteurs en haut
                    cv2.putText(overlay2, f'Density Count: {density_count:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    yolo_label = self.yolo_counter.model_name
                    cv2.putText(overlay2, f'{yolo_label} People: {yolo_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    # Affiche la dernière valeur moyenne max sous le compteur YOLO
                    if log_data:
                        last_max = log_data[-1][3]
                        cv2.putText(overlay2, f'Last Max Avg: {last_max:.2f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                    # Resize preview to 1080p for display
                    preview_1080p = cv2.resize(overlay2, (1920, 1080))
                    
                    # Draw mini graph
                    self._draw_mini_graph(preview_1080p)
                    
                    # Affiche le FPS en haut à droite
                    total_time = time.time() - start_time
                    fps = 1.0 / total_time if total_time > 0 else 0
                    cv2.putText(preview_1080p, f'FPS: {fps:.2f}', (1700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    
                    cv2.imshow(f"People Count", preview_1080p)
                    print(f"Parallel Inference: YOLO={yolo_count}, Density={density_count:.2f} | Time: {total_process_time:.3f}s | FPS: {fps:.2f}")
                    
                    # Update mini-graph history at every frame for live feedback
                    self.history_data.append((yolo_count, density_count, max(yolo_count, density_count)))
                    if len(self.history_data) > self.history_len:
                        self.history_data.pop(0)

                    counts_yolo.append(yolo_count)
                    counts_density.append(density_count)
                    now_sec = int(time.time())
                    if now_sec - last_period >= AVERAGING_PERIOD_SECONDS:
                        if counts_yolo and counts_density:
                            avg_yolo = sum(counts_yolo) / len(counts_yolo)
                            avg_density = sum(counts_density) / len(counts_density)
                            max_count = max(avg_yolo, avg_density)
                            now_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(f"Average YOLO ({yolo_label}) People (last {AVERAGING_PERIOD_SECONDS} sec): {avg_yolo:.2f}")
                            print(f"Average Density People (last {AVERAGING_PERIOD_SECONDS} sec): {avg_density:.2f}")
                            print(f"Max People Count (last {AVERAGING_PERIOD_SECONDS} sec): {max_count:.2f}")
                            writer.writerow([now_dt, f"{avg_yolo:.2f}", f"{avg_density:.2f}", f"{max_count:.2f}"])
                            csvfile.flush()
                            log_data.append((now_dt, avg_yolo, avg_density, max_count))
                        
                        counts_yolo = []
                        counts_density = []
                        last_period = now_sec
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q') or cv2.getWindowProperty("People Count", cv2.WND_PROP_VISIBLE) < 1:
                        break
            finally:
                self.capture.release()
                cv2.destroyAllWindows()
                print("Camera released and application closed.")
                # Génère la courbe à partir du CSV
                if log_data:
                    times = [dt for dt, _, _, _ in log_data]
                    yolo_vals = [y for _, y, _, _ in log_data]
                    density_vals = [d for _, _, d, _ in log_data]
                    max_vals = [m for _, _, _, m in log_data]
                    plt.figure(figsize=(14, 7))
                    plt.plot(times, yolo_vals, marker='o', markersize=4, label='YOLO')
                    plt.plot(times, density_vals, marker='x', markersize=4, label='Density')
                    plt.plot(times, max_vals, marker='s', markersize=4, label='Max')
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

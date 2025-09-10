import cv2
import sys
import screeninfo
import time
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from camera_capture import CameraCapture
from people_counter_processor import PeopleCounterProcessor
from yolo_people_counter import YoloPeopleCounter

# Période de moyennage en secondes
AVERAGING_PERIOD_SECONDS = 10  # Modifiez à 10 pour une moyenne sur 10 secondes

class CameraAppPipeline:
    def __init__(self):
        self.capture = CameraCapture(width=1920, height=1080, auto_exposure=1, gain=0, brightness = 0, zoom = 0, contrast = 0) #width=1920, height=1080
        self.processor = PeopleCounterProcessor(model_name="DM-Count", model_weights="QNRF")
        self.yolo_counter = YoloPeopleCounter(model_name='yolo12x', device='cuda', confidence_threshold=0.6)  # Modifiez le seuil ici
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height
        self.use_yolo = True  # Passe à True pour utiliser YOLO au lieu du modèle density

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
                    frame = self.capture.get_frame()
                    if frame is None:
                        print("[ERROR] Frame is None. Vérifiez la connexion de la caméra et ses paramètres.")
                        print(f"Camera opened: {self.capture.is_opened}")
                        break
                    # print(f"Frame shape: {frame.shape}")
                    # YOLO sur image de base
                    t0 = time.time()
                    yolo_count, frame_with_bbox = self.yolo_counter.count_people(frame, tile_size=640, draw_boxes=True)
                    t1 = time.time()
                    # Densité sur image de base (pas de bbox)
                    _, density_overlay, density_count = self.processor.process(frame)
                    t2 = time.time()
                    yolo_time = t1 - t0
                    density_time = t2 - t1
                    total_time = t2 - start_time
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
                    cv2.imshow(f"People Count", preview_1080p)
                    print(f"YOLO: {yolo_count} (time: {yolo_time:.3f}s) | Density: {density_count:.2f} (time: {density_time:.3f}s) | Total: {total_time:.3f}s")
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
    app = CameraAppPipeline()
    app.run()

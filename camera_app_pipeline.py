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

# Période de moyennage en secondes
AVERAGING_PERIOD_SECONDS = 10  # Modifiez à 10 pour une moyenne sur 10 secondes

class CameraAppPipeline:
    def __init__(self):
        self.capture = CameraCapture(auto_exposure=1, gain=0.5, brightness = 0.5, zoom = 0.1) #width=1920, height=1080
        
        # possible couples: possible models: 'CSRNet' (SHA, SHB), 'SFANet' (SHA, SHB),'Bay' (QNRF, SHA, SHB), 'DM-Count' (QNRF, SHA, SHB)
        self.processor = PeopleCounterProcessor(model_name="DM-Count", model_weights="QNRF")
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height

    def run(self):
        # Prépare le fichier CSV avec date et heure
        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f"people_count_{now_str}.csv"
        img_filename = f"people_count_{now_str}.png"
        csv_path = os.path.join(os.getcwd(), csv_filename)
        img_path = os.path.join(os.getcwd(), img_filename)
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["datetime", "average_count"])
            counts = []
            last_period = int(time.time())
            log_data = []
            try:
                while True:
                    frame = self.capture.get_frame()
                    if frame is None:
                        print("Failed to grab frame.")
                        break
                    frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                    _, overlay, count = self.processor.process(frame)
                    overlay2 = cv2.resize(overlay, (self.screen_width, self.screen_height))
                    cv2.imshow(f"People Count", overlay2)
                    counts.append(count)
                    now_sec = int(time.time())
                    if now_sec - last_period >= AVERAGING_PERIOD_SECONDS:
                        if counts:
                            avg_count = sum(counts) / len(counts)
                            now_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(f"Average People Count (last {AVERAGING_PERIOD_SECONDS} seconds): {avg_count:.2f}")
                            writer.writerow([now_dt, f"{avg_count:.2f}"])
                            csvfile.flush()
                            log_data.append((now_dt, avg_count))
                        counts = []
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
                    times = [dt for dt, _ in log_data]
                    values = [v for _, v in log_data]
                    plt.figure(figsize=(12, 6))
                    plt.plot(times, values, marker='o')
                    plt.xticks(rotation=45)
                    plt.xlabel('Time')
                    plt.ylabel('Average People Count')
                    plt.title('People Count Over Time')
                    plt.tight_layout()
                    plt.savefig(img_path)
                    print(f"Saved plot to {img_path}")

if __name__ == "__main__":
    sys.path.append(r"E:\AI\lwcc")
    app = CameraAppPipeline()
    app.run()

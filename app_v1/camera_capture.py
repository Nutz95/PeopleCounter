import cv2
import time
import requests
import numpy as np

class CameraCapture:
    def __init__(self, camera_index=0, width=3840, height=2160, auto_exposure=1, gain = 1, brightness = 1, zoom = 1, contrast = 100, saturation = 100):
        # Convert to int if it's a number, else keep as string for URL
        try:
            source = int(camera_index)
        except Exception:
            source = camera_index

        self.source = source
        self.cap = None
        self.http_stream = None
        self.http_buffer = b""

        # If source is a URL, try normal VideoCapture first, else prepare HTTP MJPEG fallback
        if isinstance(source, str) and source.startswith(('http://', 'https://', 'rtsp://')):
            self.cap = cv2.VideoCapture(source)
            # If VideoCapture couldn't open, we'll use requests-based MJPEG reader
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                try:
                    self.http_stream = requests.get(source, stream=True, timeout=5)
                    self.http_iter = self.http_stream.iter_content(chunk_size=1024)
                except Exception:
                    self.http_stream = None
        else:
            self.cap = cv2.VideoCapture(source)
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception:
                pass
        self.last_capture_time = 0

        # Enable autofocus
        try:
            #self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print("Autofocus enabled.")
        except Exception as e:
            print(f"Failed to enable autofocus: {e}")

        # Enable auto exposure
        try:
            #jself.cap.set(cv2.CAP_PROP_EXPOSURE, 0.1)
            #self.cap.set(cv2.CAP_PROP_ZOOM , zoom)
            #self.cap.set(cv2.CAP_PROP_CONTRAST , contrast)
            #self.cap.set(cv2.CAP_PROP_SATURATION , saturation)
            #self.cap.set(cv2.CAP_PROP_GAIN , gain)
            #self.cap.set(cv2.CAP_PROP_BRIGHTNESS  , brightness)
            #self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)  # 0.75 is often used for auto mode
            print("Auto exposure enabled.")
        except Exception as e:
            print(f"Failed to enable auto exposure: {e}")

        # Enable auto white balance
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            print("Auto white balance enabled.")
        except Exception as e:
            print(f"Failed to enable auto white balance: {e}")

        self.is_opened = self.cap.isOpened()

    def get_frame(self):
        if not self.is_opened:
            return None
        t_start = time.time()

        # If we have an OpenCV capture, use it
        if self.cap is not None:
            ret, frame = self.cap.read()
            self.last_capture_time = time.time() - t_start
            if ret and frame is not None:
                return frame

        # Fallback: try reading MJPEG via requests
        if self.http_stream is not None:
            try:
                for chunk in self.http_iter:
                    if not chunk:
                        continue
                    self.http_buffer += chunk
                    start = self.http_buffer.find(b'\xff\xd8')
                    end = self.http_buffer.find(b'\xff\xd9')
                    if start != -1 and end != -1 and end > start:
                        jpg = self.http_buffer[start:end+2]
                        self.http_buffer = self.http_buffer[end+2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        self.last_capture_time = time.time() - t_start
                        if frame is not None:
                            return frame
            except Exception:
                return None

        return None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

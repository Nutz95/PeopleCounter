import cv2
import time

class CameraCapture:
    def __init__(self, camera_index=0, width=3840, height=2160, auto_exposure=1, gain = 1, brightness = 1, zoom = 1, contrast = 100, saturation = 100):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
        ret, frame = self.cap.read()
        self.last_capture_time = time.time() - t_start
        
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

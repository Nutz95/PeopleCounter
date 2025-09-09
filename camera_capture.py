import cv2

class CameraCapture:
    def __init__(self, camera_index=0, width=3840, height=2160):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Enable autofocus
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print("Autofocus enabled.")
        except Exception as e:
            print(f"Failed to enable autofocus: {e}")

        # Enable auto exposure
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 is often used for auto mode
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
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

import cv2

class RTSPCapture:
    def __init__(self, rtsp_url, width=1920, height=1080):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    @property
    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

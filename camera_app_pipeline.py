import cv2
import sys
import screeninfo
from camera_capture import CameraCapture
from people_counter_processor import PeopleCounterProcessor

class CameraAppPipeline:
    def __init__(self):
        self.capture = CameraCapture() #width=1920, height=1080
        
        # possible couples: possible models: 'CSRNet' (SHA, SHB), 'SFANet' (SHA, SHB),'Bay' (QNRF, SHA, SHB), 'DM-Count' (QNRF, SHA, SHB)
        self.processor = PeopleCounterProcessor(model_name="DM-Count", model_weights="QNRF")
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height

    def run(self):
        try:
            while True:
                frame = self.capture.get_frame()
                if frame is None:
                    print("Failed to grab frame.")
                    break
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                _, overlay, count = self.processor.process(frame)
                overlay2 = cv2.resize(overlay, (self.screen_width, self.screen_height))
                # Affiche uniquement l'image annotée avec le nombre en rouge
                cv2.imshow(f"People Count", overlay2)
                print("People Count:", count)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or cv2.getWindowProperty("People Count", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.capture.release()
            cv2.destroyAllWindows()
            print("Camera released and application closed.")

if __name__ == "__main__":
    sys.path.append(r"E:\AI\lwcc")
    app = CameraAppPipeline()
    app.run()

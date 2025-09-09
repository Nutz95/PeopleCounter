import cv2
import screeninfo

class SimpleCameraApp:
    def __init__(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Set resolution to 4K
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

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

        # Get screen dimensions
        screen = screeninfo.get_monitors()[0]
        self.screen_width = screen.width
        self.screen_height = screen.height

    def start_preview(self):
        # Start the camera preview
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Resize the frame to fit the screen dimensions
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))

                # Display the frame
                cv2.imshow("Simple Camera Preview", frame)

                # Exit on pressing 'q' or clicking the close button
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or cv2.getWindowProperty("Simple Camera Preview", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        # Release the camera and close windows
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and application closed.")

# Example usage
if __name__ == "__main__":
    app = SimpleCameraApp()
    app.start_preview()

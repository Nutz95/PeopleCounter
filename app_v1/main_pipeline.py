from camera_app_pipeline import CameraAppPipeline
import os

if __name__ == "__main__":
    # Ensure capture mode is set
    mode = os.environ.get('CAPTURE_MODE', 'usb')
    app = CameraAppPipeline(capture_mode=mode)
    app.run()

import paho.mqtt.client as mqtt
import numpy as np
import cv2
import threading
import time

class MqttCapture:
    @property
    def is_opened(self):
        # Considère ouvert si le client MQTT est connecté et a reçu au moins une image
        return self.frame is not None
    def __init__(self, broker_address="127.0.0.1", broker_port=1883, exposure=500, width=1920, height=1080):
        self.cmd_topic = "maixcam/capture"
        self.img_topic = "maixcam/image"
        self.exposure_topic = "maixcam/exposure"
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.connect(self.broker_address, self.broker_port, 60)
        self.client.subscribe(self.img_topic)
        self.client.loop_start()
        # Set initial exposure
        self.set_exposure(exposure)
        # Request first image
        self.request_image()

    def on_message(self, client, userdata, msg):
        if msg.topic == self.img_topic:
            nparr = np.frombuffer(msg.payload, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img_resized = cv2.resize(img, (self.width, self.height))
                with self.lock:
                    self.frame = img_resized
            # Request next image automatically
            self.request_image()

    def request_image(self):
        self.client.publish(self.cmd_topic, "capture")

    def set_exposure(self, value):
        self.client.publish(self.exposure_topic, str(value))

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                return None

    def release(self):
        self.client.loop_stop()
        self.client.disconnect()

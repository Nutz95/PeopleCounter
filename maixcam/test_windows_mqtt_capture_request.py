

import paho.mqtt.client as mqtt
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

broker_address = "127.0.0.1"
broker_port = 1883
cmd_topic = "maixcam/capture"
img_topic = "maixcam/image"
exposure_topic = "maixcam/exposure"
received_img = None
last_img_time = None
interval_var = None

def on_message(client, userdata, msg):
    global received_img, last_img_time
    if msg.topic == img_topic:
        now = time.time()
        if last_img_time is not None:
            interval = now - last_img_time
            interval_var.set(f"Intervalle: {interval:.2f} s")
        last_img_time = now
        nparr = np.frombuffer(msg.payload, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        received_img = img
        update_image()
        send_capture()  # Demande une nouvelle image à chaque réception

def send_capture():
    client.publish(cmd_topic, "capture")

def send_exposure():
    val = exposure_var.get()
    try:
        val_int = int(val)
        client.publish(exposure_topic, str(val_int))
    except:
        pass

def update_image():
    if received_img is not None:
        # Resize to 1280x720 (720p)
        img_resized = cv2.resize(received_img, (1280, 720))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        image_label.imgtk = imgtk
        image_label.config(image=imgtk)

client = mqtt.Client()
client.on_message = on_message
client.connect(broker_address, broker_port, 60)
client.subscribe(img_topic)
client.loop_start()


root = tk.Tk()
root.title("MaixCam Capture")

frame_controls = ttk.Frame(root)
frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

capture_btn = ttk.Button(frame_controls, text="Capture", command=send_capture)
capture_btn.pack(side=tk.LEFT, padx=5)

exposure_var = tk.StringVar(value="500")
exposure_entry = ttk.Entry(frame_controls, textvariable=exposure_var, width=8)
exposure_entry.pack(side=tk.LEFT, padx=5)

exposure_btn = ttk.Button(frame_controls, text="Set Exposure", command=send_exposure)
exposure_btn.pack(side=tk.LEFT, padx=5)

interval_var = tk.StringVar(value="Intervalle: - s")
interval_label = ttk.Label(frame_controls, textvariable=interval_var)
interval_label.pack(side=tk.LEFT, padx=10)

frame_image = ttk.Frame(root)
frame_image.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

image_label = tk.Label(frame_image)
image_label.pack(fill=tk.BOTH, expand=True)

# Demande une image au démarrage
send_capture()

root.mainloop()

client.loop_stop()
client.disconnect()

import cv2
import paho.mqtt.client as mqtt
import numpy as np
from maix import image, camera, display, touchscreen
# Touchscreen and exit button logic
USE_TOUCHSCREEN = True
DRAW_ON_IMAGE = True
exit_btn_pos = (0, 0, 400, 200)
pressed_flag = [False, False, False]

def draw_exit_btn(img):
    img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3], image.Color.from_rgb(255, 255, 255), 2)
    text = "EXIT"
    font_size = 10
    text_width = len(text) * 8 * font_size
    text_height = 8 * font_size
    text_x = exit_btn_pos[0] + (exit_btn_pos[2] - text_width) // 2
    text_y = exit_btn_pos[1] + (exit_btn_pos[3] - text_height) // 2
    img.draw_string(text_x, text_y, text, image.COLOR_WHITE, scale=font_size)

def is_in_button(x, y, btn_pos):
    return x > btn_pos[0] and x < btn_pos[0] + btn_pos[2] and y > btn_pos[1] and y < btn_pos[1] + btn_pos[3]

def is_in_exit_button(x, y):
    return is_in_button(x, y, exit_btn_pos)

def handle_exit_button(x, y, pressed):
    global pressed_flag
    if pressed:
        if is_in_exit_button(x, y):
            pressed_flag[2] = True
    else:
        if pressed_flag[2]:
            print("exit btn click")
            pressed_flag[2] = False
            return True
    return False

broker_address = "192.168.1.68"
broker_port = 1883
cmd_topic = "maixcam/capture"
img_topic = "maixcam/image"
exposure_topic = "maixcam/exposure"

# Initialise la capture au démarrage
cam = camera.Camera(1920, 1080)
cam.exposure(100)
disp = display.Display()
if USE_TOUCHSCREEN:
    ts = touchscreen.TouchScreen()

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(cmd_topic)
    client.subscribe(exposure_topic)

def on_message(client, userdata, msg):
    if msg.topic == cmd_topic and msg.payload.decode() == "capture":
        print("Capture command received!")
        img = cam.read()
        if img is not None:
            # Image pour affichage (avec bouton exit)
            img_display = img.copy()
            if USE_TOUCHSCREEN:
                x, y, pressed = ts.read()
                if handle_exit_button(x, y, pressed):
                    print("Exit requested via touchscreen.")
                    raise KeyboardInterrupt
                draw_exit_btn(img_display)
            disp.show(img_display)
            # Image envoyée à MQTT (sans bouton)
            img_cv = image.image2cv(img, ensure_bgr=True, copy=True)
            _, buffer = cv2.imencode('.jpg', img_cv)
            client.publish(img_topic, buffer.tobytes())
            print("Image published.")
        else:
            print("Failed to capture image.")
    elif msg.topic == exposure_topic:
        try:
            new_exposure = int(msg.payload.decode())
            cam.exposure(new_exposure)
            print(f"Exposure updated to {new_exposure}")
        except Exception as e:
            print(f"Failed to update exposure: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker_address, broker_port, 60)
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("Arrêt demandé par exit button ou Ctrl+C")
finally:
    cam.close()
    disp.close()

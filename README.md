# PeopleCounter

Python script that uses 4K UVC camera and AI model to count people from the video stream.

## requirements

Python 3.11

## Setup

Create python environement
> python -m venv .venv
> .venv\Scripts\Activate    (ou source .venv/Scripts/activate)
> python.exe -m pip install --upgrade pip
> pip install opencv-python Pillow screeninfo numpy matplotlib

## Start application

OPENVINO_DEVICE=NPU ./run_people_counter_usb.sh
OPENVINO_DEVICE=GPU.0 YOLO_DEVICE=GPU.0 ./run_people_counter_usb.sh
OPENVINO_DEVICE=GPU.0 YOLO_DEVICE=NPU ./run_people_counter_usb.sh
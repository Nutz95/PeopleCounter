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
OPENVINO_DEVICE=GPU.0 YOLO_DEVICE=GPU.0 ./run_people_counter_usb.sh => la plus performante.
OPENVINO_DEVICE=GPU.0 YOLO_DEVICE=NPU ./run_people_counter_usb.sh 
OPENVINO_DEVICE=NPU YOLO_DEVICE=GPU.0 ./run_people_counter_usb.sh

YOLO_TILING=1.  => active le tiling yolo.


TensorRT:
YOLO_BACKEND=torch YOLO_MODEL=yolo11n.engine ./run_people_counter_usb.sh

export YOLO_MODEL=yolo12s.engine
export YOLO_MODEL=yolo12x.engine
./run_people_counter_rtx.sh 1080p
./run_people_counter_rtx.sh 4k
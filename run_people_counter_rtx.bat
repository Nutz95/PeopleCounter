@echo off
set CAPTURE_MODE=usb
set CAMERA_INDEX=0
set CAMERA_WIDTH=1920
set CAMERA_HEIGHT=1080

:: YOLO Configuration: Use TensorRT on RTX 5060 Ti
set YOLO_BACKEND=tensorrt_native
set YOLO_MODEL=yolo11n
set YOLO_DEVICE=cuda

:: LWCC Configuration: Use OpenVINO on NPU
set LWCC_BACKEND=openvino
set OPENVINO_DEVICE=NPU

:: Disable Tiling for speed (we use full frame)
set YOLO_TILING=0

echo Starting People Counter:
echo YOLO on RTX 5060 Ti (TensorRT)
echo LWCC on Intel NPU (OpenVINO)

.venv\Scripts\python.exe main_pipeline.py
pause

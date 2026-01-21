#!/bin/bash

# Choix de la résolution via argument (./run_people_counter_rtx.sh 1080p ou 4k)
RES=${1:-4k}

# Configuration de la capture
export CAPTURE_MODE=usb
export CAMERA_INDEX=0

if [ "$RES" == "1080p" ]; then
    echo "Mode : 1080p (Optimisé Performance)"
    export CAMERA_WIDTH=1920
    export CAMERA_HEIGHT=1080
    export LWCC_TRT_ENGINE=dm_count.engine
else
    echo "Mode : 4K (Optimisé Précision)"
    export CAMERA_WIDTH=3840
    export CAMERA_HEIGHT=2160
    export LWCC_TRT_ENGINE=dm_count_4k.engine
fi

# Configuration YOLO : Utilise TensorRT sur la RTX 5060 Ti
# Options disponibles : yolo11n, yolo11s, yolo12n, yolo12s, yolo12m
export YOLO_BACKEND=tensorrt_native
export YOLO_MODEL=yolo12m.engine
export YOLO_DEVICE=cuda

# Configuration LWCC : Utilise TensorRT sur la RTX 5060 Ti
export LWCC_BACKEND=tensorrt

# Réactive le découpage (tiling) pour la précision
export YOLO_TILING=1

echo "Démarrage du People Counter (MODE FULL RTX - HYPER PERFORMANCE) :"
echo " - YOLO sur RTX 5060 Ti (via TensorRT + Tiling)"
echo " - LWCC sur RTX 5060 Ti (via TensorRT)"

# Utilisation du chemin relatif pour l'exécutable python du venv
./.venv/Scripts/python.exe main_pipeline.py

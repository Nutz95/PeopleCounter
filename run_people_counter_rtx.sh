#!/bin/bash

# Choix de la résolution via argument (./run_people_counter_rtx.sh 1080p ou 4k)
RES=${1:-4k}
# Choix du modèle YOLO via argument (./run_people_counter_rtx.sh 4k yolo12s.engine)
YOLO_MODEL_ARG=${2:-yolo12m.engine}

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
export YOLO_BACKEND=tensorrt_native
export YOLO_MODEL=$YOLO_MODEL_ARG
export YOLO_DEVICE=cuda

# Configuration LWCC : Utilise TensorRT sur la RTX 5060 Ti
export LWCC_BACKEND=tensorrt

# Réactive le découpage (tiling) pour la précision
export YOLO_TILING=1

echo "Démarrage du People Counter (MODE FULL RTX) :"
echo " - Résolution : $RES"
echo " - YOLO : $YOLO_MODEL"
echo " - LWCC : $LWCC_TRT_ENGINE"

# Utilisation du chemin relatif pour l'exécutable python du venv
./.venv/Scripts/python.exe main_pipeline.py

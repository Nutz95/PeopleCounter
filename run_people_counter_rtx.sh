#!/bin/bash

# Activation de l'environnement virtuel (Git Bash / Windows)
if [ -f "./.venv/Scripts/activate" ]; then
    source ./.venv/Scripts/activate
    echo "Environnement Python (.venv) activé."
else
    echo "Attention : ./.venv/Scripts/activate non trouvé. Utilisation du python global."
fi

# Choix de la résolution via argument (./run_people_counter_rtx.sh 1080p ou 4k)
RES=${1:-4k}
# Choix du modèle YOLO via argument (./run_people_counter_rtx.sh 4k yolo12s.engine)
YOLO_MODEL_ARG=${2:-yolo12m.engine}
# Seuil de confiance YOLO (./run_people_counter_rtx.sh 4k yolo12x.engine 0.65)
CONF_ARG=${3:-0.65}
# Activation Tiling YOLO (./run_people_counter_rtx.sh 4k yolo12x.engine 0.65 1)
YOLO_TILING_ARG=${4:-1}
# Activation Tiling Densité 2x2 (./run_people_counter_rtx.sh 4k yolo12x.engine 0.65 1 0)
DENSITY_TILING_ARG=${5:-0}
# Seuil de détection Densité (./run_people_counter_rtx.sh 4k yolo12x.engine 0.65 1 0 15)
DENSITY_THRESHOLD_ARG=${6:-15}
# Réduction de bruit (./run_people_counter_rtx.sh 4k yolo12x.engine 0.65 1 0 15 0)
# 0: Désactivé, 1-5: Intensité du flou Gaussien
DENOISE_ARG=${7:-0}
# Mode Segmentation YOLO (./run_people_counter_rtx.sh 4k yolo11s-seg.engine 0.65 1 0 15 1 1)
# 0: Détection (BBox), 1: Segmentation (pour fusion précise)
YOLO_SEG_ARG=${8:-0}

# Configuration de la capture
export CAPTURE_MODE=usb
export CAMERA_INDEX=0
export YOLO_CONF=$CONF_ARG
export YOLO_TILING=$YOLO_TILING_ARG
export YOLO_SEG=$YOLO_SEG_ARG
export DENSITY_TILING=$DENSITY_TILING_ARG
export DENSITY_THRESHOLD=$DENSITY_THRESHOLD_ARG
export DENOISE_STRENGTH=$DENOISE_ARG

if [ "$RES" == "1080p" ]; then
    echo "Mode : 1080p (Optimisé Performance)"
    export CAMERA_WIDTH=1920
    export CAMERA_HEIGHT=1080
    export LWCC_TRT_ENGINE="models/tensorrt/dm_count_qnrf.engine"
else
    echo "Mode : 4K (Optimisé Précision)"
    export CAMERA_WIDTH=3840
    export CAMERA_HEIGHT=2160
    # Note : On utilise l'engine standard (1080p profile) même en 4K car on resize les quadrants à 1000px max
    export LWCC_TRT_ENGINE="models/tensorrt/dm_count_qnrf.engine"
fi

# Resolution du chemin YOLO
# Si l'argument finit par .engine et n'est pas un chemin absolu, on cherche dans models/tensorrt
if [[ "$YOLO_MODEL_ARG" == *.engine ]] && [[ "$YOLO_MODEL_ARG" != */* ]] && [[ "$YOLO_MODEL_ARG" != *\\* ]]; then
    export YOLO_MODEL="models/tensorrt/$YOLO_MODEL_ARG"
elif [[ "$YOLO_MODEL_ARG" == *.pt ]] && [[ "$YOLO_MODEL_ARG" != */* ]] && [[ "$YOLO_MODEL_ARG" != *\\* ]]; then
    export YOLO_MODEL="models/pt/$YOLO_MODEL_ARG"
else
    export YOLO_MODEL="$YOLO_MODEL_ARG"
fi

# Configuration YOLO : Sélection automatique du backend si non défini
export YOLO_BACKEND=${YOLO_BACKEND:-tensorrt_native}

# Configuration du Device : On ne force pas "cuda" si on est en OpenVINO
if [ "$YOLO_BACKEND" == "openvino_native" ]; then
    export YOLO_DEVICE=${YOLO_DEVICE:-GPU}
elif [ "$YOLO_BACKEND" == "openvino" ]; then
    export YOLO_DEVICE=${YOLO_DEVICE:-GPU}
else
    export YOLO_DEVICE=${YOLO_DEVICE:-cuda}
fi

# Configuration LWCC : Utilise TensorRT par défaut
export LWCC_BACKEND=${LWCC_BACKEND:-tensorrt}

echo "Démarrage du People Counter (MODE FULL RTX) :"
echo " - Résolution : $RES"
echo " - YOLO : $YOLO_MODEL"
echo " - LWCC : $LWCC_TRT_ENGINE"

# Lancement de l'application
python main_pipeline.py

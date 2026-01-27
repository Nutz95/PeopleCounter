#!/bin/bash

# Petit script interactif pour choisir son profil d'exécution
echo "===================================================="
echo "    PEOPLE COUNTER - LANCEUR MULTI-DEVICE"
echo "===================================================="
echo "Choisissez un profil d'exécution :"
echo "1) [EXTREME] Full RTX (YOLO Seg: RTX | Densité: RTX)"
echo "2) [HYBRID]  Zéro RTX (YOLO Seg: NPU | Densité: iGPU Arc)"
echo "3) [TRIPLE]  Triple-Play (YOLO Seg: iGPU Arc | Densité: RTX)"
echo "4) [CPU]     CPU Only (Tout sur le CPU)"
echo "5) [EXIT]    Quitter"
echo "----------------------------------------------------"
read -p "Votre choix [1-5]: " choice

case $choice in
    1)
        source ./scripts/configs/rtx_extreme.env
        ;;
    2)
        source ./scripts/configs/intel_hybrid.env
        ;;
    3)
        source ./scripts/configs/balanced_tri_chip.env
        ;;
    4)
        source ./scripts/configs/cpu_fallback.env
        ;;
    5)
        exit 0
        ;;
esac

# Utilisation des variables définies dans les fichiers .env avec valeurs par défaut si absentes
MODEL=${YOLO_MODEL:-"yolo11n.pt"}
CONF=${CONF:-0.65}
SEG=${YOLO_SEG:-0}

echo "Lancement avec le profil sélectionné..."
echo " - Modèle : $MODEL"
echo " - Backend YOLO : $YOLO_BACKEND ($YOLO_DEVICE)"
echo " - Segmentation : $SEG"

./run_people_counter_rtx.sh 4k "$MODEL" "$CONF" 1 0 15 1 "$SEG"

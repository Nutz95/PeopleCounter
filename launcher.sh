#!/bin/bash

# Petit script interactif pour choisir son profil d'exécution
echo "===================================================="
echo "    PEOPLE COUNTER - LANCEUR MULTI-DEVICE"
echo "===================================================="
echo "Choisissez un profil d'exécution :"
echo "1) [EXTREME] Full RTX (YOLO: RTX | Densité: RTX)"
echo "2) [HYBRID]  Zéro RTX (YOLO: NPU | Densité: iGPU Arc)"
echo "3) [BALANCE] Balanced (YOLO: RTX | Densité: iGPU Arc)"
echo "4) [CPU]     CPU Only (Tout sur le CPU)"
echo "5) [EXIT]    Quitter"
echo "----------------------------------------------------"
read -p "Votre choix [1-5]: " choice

case $choice in
    1)
        source ./scripts/configs/rtx_extreme.env
        MODEL="yolo12s.engine"
        SEG=0
        ;;
    2)
        source ./scripts/configs/intel_hybrid.env
        MODEL="yolo12s"
        SEG=0
        ;;
    3)
        source ./scripts/configs/balanced_tri_chip.env
        MODEL="yolo12s.engine"
        SEG=0
        ;;
    4)
        source ./scripts/configs/cpu_fallback.env
        MODEL="yolo12s.pt"
        SEG=0
        ;;
    5)
        exit 0
        ;;
    *)
        echo "Choix invalide."
        exit 1
        ;;
esac

echo "Lancement avec le profil sélectionné..."
./run_people_counter_rtx.sh 4k $MODEL 0.5 1 0 50 1 $SEG

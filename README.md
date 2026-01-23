# PeopleCounter

Python script that uses 4K UVC camera and AI model to count people from the video stream.

## requirements

Python 3.11

## Setup

Create python environement
> python -m venv .venv

Activate (Windows CMD / PowerShell)
> .venv\Scripts\Activate

Activate (Git Bash / Linux / macOS)
> source .venv/Scripts/activate
> python.exe -m pip install --upgrade pip
> pip install opencv-python Pillow screeninfo numpy matplotlib ultralytics

## Start application (RTX Version)

`./run_people_counter_rtx.sh [RESOLUTION] [MODEL] [CONF] [YOLO_TILING] [DENSITY_TILING] [DENSITY_THRESHOLD] [DENOISE] [YOLO_SEG]`

Arguments :

1. **RESOLUTION** : `4k` (défaut) ou `1080p`
2. **MODEL** : `yolo12x.engine`, `yolo11s-seg.engine`, etc.
3. **CONF** : Seuil YOLO (ex: `0.65`)
4. **YOLO_TILING** : `1` (actif par défaut) ou `0`
5. **DENSITY_TILING** : `1` (actif 2x2) ou `0` (global, défaut)
6. **DENSITY_THRESHOLD** : Seuil de détection Densité (ex: `15`, défaut). Plus bas = plus sensible.
7. **DENOISE** : Réduction de bruit (ex: `0` désactivé par défaut, ou `1-3` pour l'intensité).
8. **YOLO_SEG** : Mode Segmentation (ex: `0` désactivé défaut, `1` actif). **Nécessite un modèle -seg**.

Exemples :

```bash
# Standard 4K avec tout en automatique
./run_people_counter_rtx.sh 4k yolo12x.engine

# 4K avec Segmentation (Fusion ultra-précise via masques)
./run_people_counter_rtx.sh 4k yolo11s-seg.engine 0.50 1 0 25 1 1

# 4K avec réduction de bruit active (niveau 1)
./run_people_counter_rtx.sh 4k yolo12m.engine 0.70 1 1 15 1
```

# 4K sans tiling YOLO et sensibilité densité augmentée (seuil 10)

./run_people_counter_rtx.sh 4k yolo12x.engine 0.70 0 0 10 0 0

```

## Benchmark des Performances (YOLO12s)

| Mode | FPS | YOLO(s) | DENS(s) | CPU% | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4K Standard** | ~4.9 | 0.08s | 0.12s | 15% | Référence sans tiling |
| **4K YOLO Tiling** | ~3.0 | 0.21s | 0.12s | 17% | 45 tuiles YOLO (Précision ++) |
| **4K Density Tiling** | ~2.6 | 0.09s | 0.29s | 16% | 4 quadrants LWCC |
| **4K FULL Tiling** | **~2.0** | 0.22s | 0.27s | 19% | **Mode Précision Maximale** |

> Note: Le Tiling Densité 4K est optimisé via un resize automatique à 1000px des quadrants (Gain x3 vs mode brut).

## Interface (HUD)

- **Vert** : YOLO (Détection d'objets)
- **Rouge** : LWCC (Densité)
- **Cyan** : Moyenne combinée des deux algorithmes
- **Cercles rouges** : Foyers de densité détectés par LWCC

Appuyez sur **'q'** pour quitter et générer le graphique final.

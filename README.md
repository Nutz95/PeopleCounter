# PeopleCounter

Outil de comptage de personnes depuis une caméra (4K) utilisant YOLO et LWCC.

## Prérequis

- Python 3.11

## Installation rapide

1. Créer l'environnement virtuel:

```bash
python -m venv .venv
```

2. Activer l'environnement:

- PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- Git Bash / WSL / Linux / macOS:

```bash
source .venv/Scripts/activate
```

3. Installer les dépendances (depuis `requirements.txt`):

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Git Bash / WSL:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

> Remarque: `setup.sh` exécute la même commande et gère aussi le téléchargement / conversion des modèles.

## Préparer les modèles (optionnel / long)

Exécuter `setup.sh` (Git Bash / WSL recommandé) pour télécharger les modèles et générer des moteurs TensorRT:

```bash
./setup.sh
```

Les fichiers `.engine` et `.onnx` générés sont placés dans le dossier `models/` (modifiable via `MODELS_DIR`).

Remarques importantes:

- Pour construire des moteurs TensorRT (`.engine`) localement, une installation CUDA + TensorRT fonctionnelle est requise (tests ici effectués avec CUDA 13.1 et TensorRT 10.14). Installez CUDA 13.1 si vous prévoyez d'utiliser `setup.sh` pour générer des `.engine`.
- L'export PyTorch (`.pt -> .onnx`) peut afficher des warnings concernant la conversion d'opset (p.ex. le convertisseur essaie de rester compatible et peut ré-assigner l'opset à 18). Ces warnings sont normaux; si l'export aboutit et qu'un fichier `.onnx` est créé, la conversion vers TensorRT peut tout à fait réussir malgré les messages.
- Pour réduire ces warnings, les exports YOLO utilisent désormais l'opset 18 (plus récent) — cela évite la conversion automatique et les messages liés au version_converter.
- Les fichiers OpenVINO IR (`.xml`) sont maintenant générés dans `models/openvino/` (par défaut). Chaque modèle y est sauvegardé en tant que `<modelname>_<weights>.xml`.
- Si CUDA/TensorRT n'est pas disponible sur l'hôte, `setup.sh` exportera les `.onnx` mais sautera la conversion `.onnx -> .engine` (le script détecte l'absence de CUDA via `nvidia-smi`).
- Après export, les fichiers `.pt` téléchargés sont déplacés dans le dossier `models/` pour garder tous les artefacts au même endroit.

## Lancer l'application (exemple)

Usage:

```bash
./run_people_counter_rtx.sh [RESOLUTION] [MODEL] [CONF] [YOLO_TILING] [DENSITY_TILING] [DENSITY_THRESHOLD] [DENOISE] [YOLO_SEG]
```

Arguments:

- `RESOLUTION`: `4k` (défaut) ou `1080p`
- `MODEL`: fichier modèle (`.engine`, `.onnx`, ou `.pt`) — ex: `yolo11s-seg.engine`
- `CONF`: seuil YOLO (ex: `0.50`)
- `YOLO_TILING`: `1` active le tiling YOLO, `0` désactive
- `DENSITY_TILING`: `1` active tiling density (2x2), `0` global
- `DENSITY_THRESHOLD`: seuil LWCC/density (ex: `15`, plus bas = plus sensible)
- `DENOISE`: 0 = off, 1-3 = niveaux de denoise (Gaussian blur)
- `YOLO_SEG`: `1` active le mode segmentation (requiert modèle `-seg`), `0` désactivé

Exemples (optimisés):

### Démo principale (segmentation + tiling):
```bash
./run_people_counter_rtx.sh 4k yolo11s-seg.engine 0.5 1 0 50 1 1
```

### Standard 4K avec détection (YOLOv12):
```bash
./run_people_counter_rtx.sh 4k yolo12s.engine
```

### 4K avec réduction de bruit (niveau 1):
```bash
./run_people_counter_rtx.sh 4k yolo12m.engine 0.70 1 1 15 1
```

Paramètres courts:
- Résolution: `4k` ou `1080p`
- Modèle: nom du fichier `.engine` / `.onnx` / `.pt`
- `YOLO_SEG`: active la fusion par segmentation (nécessite un modèle `-seg`)

## Notes

- `setup.sh` installe les paquets depuis `requirements.txt`.
- Si vous n'avez pas de GPU compatible TensorRT, utilisez la version PyTorch (`.pt`) ou OpenVINO.

## Exemples multi-backend (env vars)

TensorRT (NVIDIA):
```bash
export YOLO_BACKEND=tensorrt_native
export YOLO_MODEL=yolo11s-seg.engine
./run_people_counter_rtx.sh 4k $YOLO_MODEL 0.5 1 0 50 1 1
```

OpenVINO (Intel NPU / Arc):
```bash
export YOLO_BACKEND=openvino_native
export YOLO_OPENVINO_DIR=path/to/yolo_openvino_model
./run_people_counter_rtx.sh 4k yolo11s 0.5 1 0 50 1 0
```

CPU (PyTorch / Ultralytics):
```bash
export YOLO_BACKEND=torch
export YOLO_MODEL=yolo11s.pt
./run_people_counter_rtx.sh 4k yolo11s.pt 0.5 1 0 50 1 0
```

## Support

Ouvrez une issue si vous rencontrez des problèmes en précisant OS, GPU et commande utilisée.


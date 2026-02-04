# PeopleCounter

Outil de comptage de personnes depuis une caméra (4K) utilisant YOLO et LWCC.

## Prérequis

- Python 3.11

## Installation rapide

1. Créer l'environnement virtuel :

```bash
python -m venv .venv
```

2. Activer l'environnement :

- **PowerShell** :

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Git Bash / WSL / Linux / macOS** :

```bash
source .venv/Scripts/activate
```

3. Installer les dépendances (depuis `requirements.txt`) :

**PowerShell** :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Git Bash / WSL** :

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

> Remarque: `setup.sh` exécute la même commande et gère aussi le téléchargement / conversion des modèles.

## Préparer les modèles (optionnel / long)

Exécuter `setup.sh` (Git Bash / WSL recommandé) pour télécharger les modèles et générer des moteurs TensorRT :

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
- `DEBUG_TILING`: `1` affiche les rectangles de tiling (Bleu=YOLO, Magenta=Densité), `0` par défaut
- `DENSITY_THRESHOLD`: seuil LWCC/density (ex: `15`, plus bas = plus sensible)
- `DENOISE`: 0 = off, 1-3 = niveaux de denoise (Gaussian blur)
- `YOLO_SEG`: `1` active le mode segmentation (requiert modèle `-seg`), `0` désactivé

Exemples (optimisés):

### Démo principale (segmentation + tiling)

```bash
./run_people_counter_rtx.sh 4k yolo26s-seg.engine 0.5 1 0 50 1 1
```

### Standard 4K avec détection (YOLOv26)

```bash
./run_people_counter_rtx.sh 4k yolo26s.engine
```

### 4K avec réduction de bruit (niveau 1)

```bash
./run_people_counter_rtx.sh 4k yolo26m.engine 0.70 1 1 15 1
```

Paramètres courts:

- Résolution: `4k` ou `1080p`
- Modèle: nom du fichier `.engine` / `.onnx` / `.pt`
- `YOLO_SEG`: active la fusion par segmentation (nécessite un modèle `-seg`)

## Menu de lancement (Interactif)

Pour faciliter le choix, utilisez le lanceur interactif :

```bash
./launcher.sh
```

Il propose des préréglages optimisés pour votre matériel (RTX 5060 Ti + Intel Core Ultra).

Aperçu du menu :
```text
====================================================
    PEOPLE COUNTER - LANCEUR MULTI-DEVICE
====================================================
Choisissez un profil d'exécution :
1) [EXTREME] Full RTX (YOLO Seg: RTX (26x) | Densité: RTX)
2) [HYBRID]  Zéro RTX (YOLO Seg: NPU (26s) | Densité: iGPU Arc)
3) [TRIPLE]  Triple-Play (YOLO Seg: iGPU Arc (26m) | Densité: RTX)
4) [CPU]     CPU Only (Tout sur le CPU)
5) [EXIT]    Quitter
----------------------------------------------------
Votre choix [1-5]:
```

- **EXTREME** : Utilise la puissance brute de la RTX 5060 Ti avec YOLOv26x-seg et TensorRT. Idéal pour une précision maximale en 4K.
- **HYBRID** : Libère totalement le GPU NVIDIA pour d'autres tâches (streaming/rendu) en utilisant le NPU Intel pour YOLO et l'iGPU pour la densité via OpenVINO.
- **TRIPLE** : Répartition équilibrée utilisant les trois moteurs de calcul (RTX, iGPU, NPU) pour maximiser le débit d'images.

## Optimisation de la Densité (Batching)

Le processeur de densité (`PeopleCounterProcessor`) a été optimisé pour le tiling 4K :
- **Batch Processing** : Les tuiles de l'image (tiling 2x2) sont envoyées en une seule passe au GPU via TensorRT ou OpenVINO.
- **Adaptive Resolution** : Les modèles de densité fonctionnent désormais sur une matrice optimisée (ex: 540p), offrant un gain de performance de x4 sans perte significative de précision pour le comptage de foule.
- **Multi-Streaming** : Utilisation de `ThreadPoolExecutor` pour le pré-processing parallèle des frames.

## Exemples multi-backend (env vars)

**TensorRT (NVIDIA - Par défaut) :**

```bash
export YOLO_BACKEND=tensorrt_native
export YOLO_MODEL=yolo26s-seg.engine
./run_people_counter_rtx.sh 4k $YOLO_MODEL 0.5 1 0 50 1 1
```

**OpenVINO (Intel NPU / Arc) :**

```bash
# Note: YOLO_MODEL doit correspondre au nom du dossier dans models/openvino/
export YOLO_BACKEND=openvino_native
export YOLO_OPENVINO_DIR=models/openvino/yolo26s_seg_openvino_model
./run_people_counter_rtx.sh 4k yolo26s-seg 0.5 1 0 50 1 1
```

**CPU (PyTorch / Ultralytics) :**

```bash
export YOLO_BACKEND=torch
export YOLO_MODEL=yolo26s-seg.pt
./run_people_counter_rtx.sh 4k yolo26s-seg.pt 0.5 1 0 50 1 1
```

## Configuration Hétérogène (Multi-GPU/NPU)

L'application permet de répartir la charge de calcul sur les différentes puces de votre machine.

### Paramétrage manuel (Variables d'environnement) :

| Variable | Usage | Valeurs possibles |
| :--- | :--- | :--- |
| `YOLO_BACKEND` | Moteur YOLO | `tensorrt_native`, `openvino_native`, `torch` |
| `YOLO_DEVICE` | Puce pour YOLO | `cuda` (RTX), `GPU` (Arc), `NPU` (AI Boost), `CPU` |
| `LWCC_BACKEND` | Moteur Densité | `tensorrt`, `openvino`, `torch` |
| `OPENVINO_DEVICE` | Puce pour Densité OV | `GPU`, `NPU`, `CPU` |

### Pipeline CPU / CUDA

Les classes `YoloTensorRTEngine` et `YoloSegPeopleCounter` utilisent désormais `engines/yolo/preprocessors.py` pour séparer clairement le prétraitement CPU (`cv2`) de la variante CUDA (`torch` + `F.interpolate`). Activer `YOLO_USE_GPU_PREPROC=1` dans vos profils (par exemple `scripts/configs/rtx_extreme.env`) bascule sur le pipeline GPU et déporte la mise à l’échelle/copiage dans un kernel `interpolate` qui tourne directement sur la RTX. Le flag `YOLO_USE_GPU_POST=1` reste responsable de la génération des masques via `torch` ou `cupy` afin de ne pas repasser sur le CPU.

Vous pouvez vérifier quel chemin est emprunté en relançant `./run_app.sh --profile <profil> --verbose` et en surveillant le log `[DEBUG] PERF BREAKDOWN` : `preprocess` devrait chuter vers la dizaine de millisecondes quand `YOLO_USE_GPU_PREPROC` est actif et que `torch.cuda.is_available()` retourne `True`.

Le nouveau sélecteur **YOLO pipeline** dans l’interface Web expose les modes `auto`, `gpu` et `cpu` pour tester sans modifier les `.env`. `YOLO_USE_GPU_RENDER=1` force le choix GPU par défaut et active la classe `GpuMaskRenderer`, qui fusionne les masques sur la RTX avant de repartir sur l’image. Lorsqu’`EXTREME_DEBUG=1`, un log `[DEBUG] YOLO pipeline` indique à la fois le prétraitement et le renderer sélectionnés.

### Exemples de profils :

**1. Mode Full RTX (NVIDIA uniquement) :**
```bash
export YOLO_BACKEND=tensorrt_native && export LWCC_BACKEND=tensorrt
./run_people_counter_rtx.sh 4k yolo26s-seg.engine
```

**2. Mode Hybrid Intel (Zéro charge RTX) :**
```bash
export YOLO_BACKEND=openvino_native && export YOLO_DEVICE=NPU
export LWCC_BACKEND=openvino && export OPENVINO_DEVICE=GPU
./run_people_counter_rtx.sh 4k yolo26s-seg
```

## Support

Ouvrez une issue si vous rencontrez des problèmes en précisant OS, GPU et commande utilisée.


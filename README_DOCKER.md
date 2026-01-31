# Docker : Build et Exécution GPU (Ubuntu 24.04 + CUDA 13.1)

Ce dépôt contient un `Dockerfile` multi-stage optimisé pour construire une image performante incluant **OpenCV 4.13.0 (CUDA)**, **PyTorch 2.9.1**, et **TensorRT 10.14**.

## 🏗️ Procédure de Build

Le build est divisé en 3 étapes (OpenCV -> Dépendances -> Runtime) pour minimiser la taille finale et utiliser le cache efficacement.

```bash
# Lancer le build complet (prévoyez ~1h pour la première compilation OpenCV)
docker build -t people-counter:gpu-final .
```

### 🔍 Vérification du build
Une fois l'image créée, vérifiez que le GPU est bien accessible :
```bash
docker run --rm --gpus all people-counter:gpu-final python3 -c "import cv2; print('CUDA Devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## 🚀 Exécution de l'application

Comme l'image Docker ne possède pas d'interface graphique (GUI), l'application doit être lancée en mode "headless" avec un accès réseau pour le streaming (en cours de développement).

```bash
# Lancer l'application par défaut
docker run --rm --gpus all people-counter:gpu-final python3 main.py
```

---

## 📸 Partage de Caméra USB (Windows -> WSL -> Docker)

Pour utiliser votre caméra USB locale dans le conteneur Docker sous WSL2 :

### 1. Sous Windows (PowerShell Admin)
Installez `usbipd` et attachez la caméra :
```powershell
usbipd list                          # Notez l'ID (ex: 6-2)
usbipd bind --busid <ID> --force
usbipd attach --wsl Ubuntu-24.04 --busid <ID> --auto-attach
```

### 2. Sous WSL (Linux)
Vérifiez que la caméra est bien vue dans `/dev/video*` :
```bash
ls /dev/video*
# Puis lancez Docker avec l'option --device
docker run --rm --gpus all --device /dev/video0:/dev/video0 people-counter:gpu-final python3 main.py
```

---

## 📂 Gestion des fichiers et GitHub

### Fichiers obsolètes (à supprimer)
Les fichiers suivants sont des reliquats d'anciennes versions et ne sont plus nécessaires avec le nouveau `Dockerfile` :
- `Dockerfile.probe` : Test temporaire.
- `setup.sh`, `run_docker.sh`, `setup_docker.sh` : Remplacés par le workflow Docker standard.
- `make_wheelhouse.sh` (racine) : Utilisez `scripts/make_wheelhouse.sh`.

### Que faut-il commiter ?
- **OUI** : `Dockerfile`, `requirements.cuda.txt`, `scripts/make_wheelhouse.sh`.
- **NON** : Le dossier `wheelhouse/` (trop lourd, contient des binaires `.whl` qui sont téléchargés dynamiquement durant le build Docker via le cache).
- **NON** : Les dossiers `models/` (doivent être gérés via un script de téléchargement ou stockés séparément).

```
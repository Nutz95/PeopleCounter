import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Ajouter le chemin local de lwcc
sys.path.append(r"E:\AI\lwcc")
from lwcc import LWCC

# Charger l'image
img_path = "E:/AI/lwcc/tests/dataset/img01.jpg"
img = Image.open(img_path).convert('RGB')

# Afficher l'image originale
plt.imshow(img)
plt.title("Image originale")
plt.axis('off')
plt.show()

# Prédire le nombre de personnes et récupérer la carte de densité
count, density = LWCC.get_count(img_path, model_name="Bay", model_weights="SHB", return_density=True)
print(f"Nombre estimé de personnes : {count}")

# Afficher la carte de densité
plt.imshow(density, cmap='plasma')
plt.title("Carte de densité")
plt.axis('off')
plt.show()

# Redimensionner la carte de densité à la taille de l'image
density = Image.fromarray(np.uint8(density * 255), 'L')
density = density.resize((img.size[0], img.size[1]), Image.BILINEAR)

# Superposer l'image et la carte de densité
fig = plt.figure()
plt.imshow(img, origin='upper')
plt.imshow(density, alpha=0.5, origin='upper', cmap=plt.get_cmap("plasma"))
plt.axis('off')
fig.set_size_inches(15, 8.44)
plt.title("Superposition de l'image et de la carte de densité")
plt.show()

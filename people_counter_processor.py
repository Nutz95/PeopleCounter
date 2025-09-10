import sys
sys.path.append(r"E:\AI\lwcc")
from lwcc import get_count_from_frame
import cv2
import numpy as np
from PIL import Image
import torch

class PeopleCounterProcessor:
    def __init__(self, model_name="CSRNet", model_weights="SHA"):
        """_summary_

        Args:
            model_name (str, optional): _description_. Defaults to "CSRNet".
            model_weights (str, optional): _description_. Defaults to "SHA".
            possible models: 'CSRNet' (SHA, SHB), 'SFANet' (SHA, SHB),'Bay' (QNRF, SHA, SHB), 'DM-Count' (QNRF, SHA, SHB)
        """
        self.model_name = model_name
        self.model_weights = model_weights
        self.model = None
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PeopleCounterProcessor running on device: {device_str}")

    def process(self, frame):
        # Compte les personnes sur la frame
        count, density = get_count_from_frame(
            frame,
            model_name=self.model_name,
            model_weights=self.model_weights,
            model=self.model,
            return_density=True,
            use_gpu=torch.cuda.is_available()
        )
        self.model = self.model or None
        # Redimensionner la carte de densité à la taille de la frame
        density_img = Image.fromarray(np.uint8(density * 255), 'L')
        density_img = density_img.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        density_np = np.array(density_img.convert('RGB'))
        # Appliquer une colormap rouge (COLORMAP_JET donne du rouge pour les valeurs hautes)
        density_color = cv2.applyColorMap(density_np, cv2.COLORMAP_JET)

        # Superposer la carte de densité sur la frame
        overlay = cv2.addWeighted(frame, 0.7, density_color, 0.3, 0)
        # (Suppression de l'annotation 'Count:' en rouge)
        return frame, overlay, int(count)

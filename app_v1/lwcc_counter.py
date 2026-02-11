import sys
import os

# Pour l'exécution en local ou Docker, l'import direct devrait fonctionner 
# si le package est installé ou dans le PYTHONPATH
try:
    import lwcc
except ImportError:
    # Fallback pour le container s'il est dans vendors/lwcc
    sys.path.append(os.path.join(os.getcwd(), "vendors", "lwcc"))
    import lwcc

class LWCCCounter:
    def __init__(self):
        # Initialise le modèle, par exemple CSRNet
        self.model = lwcc.models.CSRNet()

    def count_people(self, image):
        """
        Compte le nombre de personnes dans une image donnée.

        :param image: numpy array (frame OpenCV)
        :return: tuple (density_map, count)
        """
        # Prédire la carte de densité
        density_map = self.model.predict(image)

        # Calculer le nombre total de personnes
        count = density_map.sum()
        return density_map, count

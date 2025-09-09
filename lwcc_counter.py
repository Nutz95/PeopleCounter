import sys
sys.path.append(r"E:\AI\lwcc")
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

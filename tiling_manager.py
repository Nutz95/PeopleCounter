import numpy as np
import cv2

class TilingManager:
    """
    Gère le découpage d'images haute résolution (4K/1080p) en tuiles de taille fixe (640x640)
    et la fusion des résultats de segmentation/détection.
    """
    def __init__(self, tile_size=640, overlap=0.2):
        self.tile_size = tile_size
        self.overlap = overlap

    def get_tiles(self, image_shape):
        """
        Calcule les coordonnées des tuiles pour couvrir toute l'image avec un overlap optimal.
        Args:
            image_shape: (H, W, C)
        Returns:
            tiles_coords: Liste de (x1, y1, x2, y2)
        """
        h, w = image_shape[:2]
        ts = self.tile_size

        # Calcul du nombre de tuiles nécessaires
        # Pour du 4K (3840x2160), 7x4 = 28 tuiles avec une tuile globale 
        # est un meilleur compromis perf/précision car la tuile globale aide à l'agrégation.
        nx = int(np.ceil(w / ts))
        if nx > 1:
            if w == 3840: nx = 7 
            elif w == 1920: nx = 4
        
        ny = int(np.ceil(h / ts))
        if ny > 1:
            if h == 2160: ny = 5 # Augmenté de 4 à 5 pour garantir une couverture totale
            elif h == 1080: ny = 3

        tiles_coords = []
        
        # Répartition uniforme de l'overlap
        if nx > 1:
            x_step = (w - ts) / (nx - 1)
        else:
            x_step = 0
            
        if ny > 1:
            y_step = (h - ts) / (ny - 1)
        else:
            y_step = 0

        for i in range(ny):
            for j in range(nx):
                x1 = int(j * x_step)
                y1 = int(i * y_step)
                x2 = x1 + ts
                y2 = y1 + ts
                
                # Sécurité bords
                if x2 > w: x2, x1 = w, w - ts
                if y2 > h: y2, y1 = h, h - ts
                
                tiles_coords.append((x1, y1, x2, y2))
                
        return tiles_coords

    def fuse_results(self, all_boxes, all_scores, all_masks=None, iou_threshold=0.3):
        """
        Fusionne les détections des différentes tuiles en utilisant un NMS global.
        Retourne une liste de groupes d'indices [ [i1, i2], [i3], ... ]
        """
        if len(all_boxes) == 0:
            return []
            
        # Compatibilité si appelé avec 3 arguments (Yolo sans seg)
        if isinstance(all_masks, float):
            iou_threshold = all_masks
            all_masks = None

        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        clusters = [] 
        processed = np.zeros(len(boxes), dtype=bool)

        for i in range(len(order)):
            idx = order[i]
            if processed[idx]: continue
            
            cluster = [idx]
            processed[idx] = True
            
            if i + 1 < len(order):
                rem_indices = order[i+1:]
                not_proc = rem_indices[~processed[rem_indices]]
                if len(not_proc) > 0:
                    xx1 = np.maximum(x1[idx], x1[not_proc])
                    yy1 = np.maximum(y1[idx], y1[not_proc])
                    xx2 = np.minimum(x2[idx], x2[not_proc])
                    yy2 = np.minimum(y2[idx], y2[not_proc])
                    w = np.maximum(0.0, xx2 - xx1)
                    h = np.maximum(0.0, yy2 - yy1)
                    inter = w * h
                    ovr = inter / (areas[idx] + areas[not_proc] - inter)
                    
                    matches = not_proc[ovr > iou_threshold]
                    for m_idx in matches:
                        if not processed[m_idx]:
                            cluster.append(m_idx)
                            processed[m_idx] = True
            clusters.append(cluster)
        return clusters

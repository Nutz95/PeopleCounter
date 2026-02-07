## Plan: Documentation Refresh and Roadmap Alignment
Ce plan dessine la remise à jour des READMEs principaux, l'ajout d'un guide d'architecture, et l'alignement des fiches de route pour que l'état actuel des masques et mesures soit documenté et traçable.

**Phases**
1. **Phase 1: Réorganiser README.md et README_DOCKER.md**
    - **Objective:** Clarifier l'installation et l'exécution, puis relier la page d'accueil aux guides Docker et d'architecture sans dupliquer les explications.
    - **Files/Functions to Modify/Create:** README.md, README_DOCKER.md
    - **Tests to Write:** Aucun test automatisé (mise à jour documentaire).
    - **Steps:**
        1. Réécrire README.md avec des sections distinctes (prérequis, préparation des modèles, commandes de lancement, métriques/masques) et un bloc "Voir aussi" pointant vers README_DOCKER.md et README_ARCHITECTURE.md.
        2. Simplifier README_DOCKER.md pour qu'il conserve uniquement les détails conteneur (profils, CUDA/CPU) et ajoute une référence explicite à README_ARCHITECTURE.md pour la vue d'ensemble.
        3. Ajouter une note conjointe rappelant que toute modification de l'architecture ou des métriques doit se refléter dans les plans et READMEs.
2. **Phase 2: Rédiger README_ARCHITECTURE.md**
    - **Objective:** Documenter les flux CUDA/CPU, la création/encodage des masques, l'intervalle de polling adaptatif et la carte de métriques, avec schémas simples.
    - **Files/Functions to Modify/Create:** README_ARCHITECTURE.md (nouveau), éventuellement des illustrations/textes référencés par README.md
    - **Tests to Write:** Aucun test automatisé (mise à jour documentaire).
    - **Steps:**
        1. Décrire l'architecture de bout en bout : pipeline GPU/CPU, composition des buffers masques, métadonnées `created_at/created_at_ts`, et propagation vers `camera_app_pipeline.py`.
        2. Ajouter un diagramme (mermaid ou ASCII) montrant la diffusion MJPEG + masque + compositing sur le client avec la carte de latence.
        3. Mentionner l'intégration des logs `[MASK TIMING]` et du polling adaptatif dans `static/js/app.js` pour expliquer les latences visibles.
3. **Phase 3: Mettre à jour les plans et feuilles de route**
    - **Objective:** Faire mention des travaux terminés (masque alpha, métriques et carte) et des futurs jalons (graphe de latence, nettoyage des logs, atteindre 25-30 fps).
    - **Files/Functions to Modify/Create:** plans/mask_overlay_roadmap.md, plans/mask_timing-plan.md, plans/documentation-plan.md (mettre à jour), éventuellement un nouveau plan de performance
    - **Tests to Write:** Aucun test automatisé (mise à jour documentaire).
    - **Steps:**
        1. Éditer mask_overlay_roadmap.md pour ajouter les états réalisés (canvas aligné, masque calculé et downscalé, overlay limité) et les suites (fusion/densité).
        2. Éditer mask_timing-plan.md pour documenter la télémétrie actuelle (timestamp, carte de latence, polling) et inscrire les objectifs comme le graphe de latence total et la refactorisation des logs.
        3. Créer ou mettre à jour plans/documentation-plan.md pour consigner ce plan de documentation et pointer vers README_ARCHITECTURE.md, en montrant que les docs restent synchronisées avec le code.

**Open Questions**
1. Souhaitez-vous que le guide d'architecture contienne un diagramme Mermaid directement intégré ou un lien vers des images séparées ?
2. Faut-il créer un nouveau plan spécifique à la performance/latence pour séparer ce travail des plans de masques existants ?
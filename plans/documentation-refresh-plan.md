## Plan: Documentation Refresh and Roadmap Alignment
Ce plan dessine la remise à jour des READMEs principaux, l'ajout d'un guide d'architecture, et l'alignement des fiches de route pour que l'état actuel des masques et mesures soit documenté et traçable.

**Phases**
1. **Phase 1: Repenser README.md et README_DOCKER.md**
    - **Objective:** Offrir une entrée rapide qui donne envie d’essayer le projet, documenter clairement l’installation (Windows+WSL, `.env`, build Docker, setup des modèles) et rediriger vers les autres guides sans dupliquer les détails techniques obsolètes.
    - **Files/Functions to Modify/Create:** README.md, README_DOCKER.md
    - **Tests to Write:** Aucun test automatisé (mise à jour documentaire).
    - **Steps:**
        1. Restructurer README.md pour qu’il présente brièvement le projet, déroule la procédure d’installation (installation des dépendances, exécution de `setup.sh`, préparation Windows/WSL via `windows/setup_and_run.bat`, compilation Docker `./build_image.sh` avec prérequis `docker`, `nvidia-smi`, drivers NVIDIA dans WSL, etc.), puis détaille comment les profils `.env` (scripts/configs) pilotent les variables exposables.
        2. Dans le même fichier, reporter l’obsolescence des sections interactives (`launcher`, menus, backend multi-stack) vers la page d’architecture et les remplacer par des liens explicites (README_DOCKER.md, README_ARCHITECTURE.md, README_YOLO.md, plans/mask_overlay_roadmap.md, plans/mask_timing-plan.md).
        3. Nettoyer README_DOCKER.md pour qu’il reste concentré sur la génération de l’image (`./build_image.sh`), les profils `.env` ciblés dans `scripts/configs`, les prérequis à la compilation (WSL, Docker Desktop/NVIDIA Container Toolkit), et la liaison vers README_ARCHITECTURE.md pour les diagrammes mermaid et les détails du pipeline CUDA/CPU/masques.
        4. Mentionner dans les deux README qu’un changement d’architecture/masque/métriques doit être reflété simultanément dans les plans (`plans/documentation-refresh-plan.md`, `plans/mask_overlay_roadmap.md`, `plans/mask_timing-plan.md`) et dans les guides spécialisés (README_ARCHITECTURE.md, README_YOLO.md, README_DENSITY.md, etc.).
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
3. Préférez-vous que les README spécialisés (Docker, architecture, YOLO, densité) soient listés dans un tableau centralisé avec leurs objectifs et liens vers les plans correspondants ?
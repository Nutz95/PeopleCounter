# Tiling Adaptatif: Principe et Analyse

## 🎯 Principe de base

Au lieu de traiter **tous les tiles à chaque frame**, détecter dynamiquement quels tiles contiennent de l'activité et **ne traiter que ceux-là**.

### Approches possibles

#### 1. Motion-based (Basique)

```
Frame N-1: [tile status après analyse]
Frame N: [détection motion par tile] → process seulement tiles avec motion

Exemple:
┌─────┬─────┬─────┬─────┐
│ 🟢  │ ⚪  │ ⚪  │ 🟢  │  Motion detected (optical flow, frame diff)
├─────┼─────┼─────┼─────┤
│ ⚪  │ 🟢  │ 🟢  │ ⚪  │  → Process only green tiles (5/8 instead of 8/8)
└─────┴─────┴─────┴─────┘
```

**Avantages**:
- Réduction brute du nombre de tiles (gain potentiel 30-50% selon scène)
- Motion detection très rapide (GPU kernel simple)

**Inconvénients**:
- **Latence détection**: 1-2 frames avant de réagir à nouveau mouvement
- **False negatives**: Mouvement lent peut être manqué
- **Scene statique avec personnes statiques**: On rate les détections!

#### 2. ROI-based (Intermédiaire)

```
Utiliser détection global comme "guide" pour tiles:

1. Run yolo_global (640x640, rapide ~6ms) → bounding boxes
2. Mapper boxes sur grille tiles → sélectionner tiles intersectant
3. Process seulement tiles sélectionnés

Exemple:
Global détecte 2 personnes aux positions (x1,y1), (x2,y2):
┌─────┬─────┬─────┬─────┐
│ ⚪  │ 🔵  │ ⚪  │ ⚪  │  🔵 = tile intersecte bbox personne
├─────┼─────┼─────┼─────┤
│ ⚪  │ 🔵  │ 🔵  │ ⚪  │  → Process only 3/8 tiles
└─────┴─────┴─────┴─────┘
```

**Avantages**:
- Utilise l'inférence global (déjà à < 10ms)
- Détection fiable basée sur YOLO
- Pas de latence (même frame)

**Inconvénients**:
- **Paradoxe**: Si global détecte bien → pourquoi refaire tiles?
- Tiles apportent surtout **précision haute résolution**
- Risque manquer petites personnes (que global rate)

#### 3. Hybrid (Complexe)

```
Combinaison motion + ROI + historique:

1. Motion detection rapide
2. SI motion OU global détecte quelque chose:
   - Process tiles ROI (haute priorité)
   - Expansion 1-2 tiles autour (buffer sécurité)
3. SINON: Process 1 tile rotatif toutes les N frames (sécurité)
```

**Avantages**:
- Maximum d'optimisation
- Robuste contre false negatives

**Inconvénients**:
- **Complexité extrême**
- Difficile à debugger
- Maintenance overhead

---

## ⚖️ Analyse coût/bénéfice

### Coût d'implémentation

| Aspect | Effort |
|--------|--------|
| Motion detection GPU kernel | 🟡 Moyen (1-2 jours) |
| Tile selection logic | 🟡 Moyen |
| ROI mapping (global → tiles) | 🟢 Facile |
| Dynamic batching tiles | 🔴 Difficile! |
| Tests coverage | 🔴 Importante (beaucoup de edge cases) |
| Debugging | 🔴 Complexe (comportement non-déterministe) |

**Total**: 1-2 semaines développement + 3-5 jours tests

### Gain potentiel (scénario réaliste)

**Scène typique**: caméra sécurité, 2-3 personnes visibles, ~40% de la surface image

#### Motion-based
- Tiles actifs: **~50%** (optimiste)
- Gain brut: **50%** (16 tiles → 8 tiles)
- Gain net: **~40%** (overhead motion detection + dynamic batching)
- **yolo_tiles**: 23ms → ~14ms
- **Problème**: Scènes statiques = **faux négatifs**!

#### ROI-based
- Tiles actifs: **~60%** (must process zones global détecte)
- Overhead: Mapping boxes + marginsécurité
- Gain net: **~30%**
- **yolo_tiles**: 23ms → ~16ms
- **Problème**: Pourquoi refaire si global détecte déjà?

### Risques

#### 1. False negatives (CRITIQUE)

```
Scénario: Personne immobile ou mouvement très lent
Motion-based: ❌ Tile marqué inactif → personne manquée
Impact métier: Comptage FAUX!
```

#### 2. Edge cases complexes

- Personne qui entre dans frame (pas dans historique)
- Occlusion partielle (visible que dans 1 tile)
- Reflets/ombres qui créent faux motion
- Camera shake ou auto-adjust exposition

#### 3. Dynamic batching overhead

```python
# Baseline: batch fixe de 32 tiles
batch = all_tiles  # Shape: [32, 3, 640, 640]
output = model(batch)  # 1 appel TRT efficace

# Adaptatif: batch variable
active_tiles = select_active()  # 8-20 tiles (variable!)
batch = stack(active_tiles)  # Shape: [N, 3, 640, 640]  ← N change!
output = model(batch)  # TRT doit gérer batch dynamique

# Problème TRT:
# - Batch size variable = overhead réallocation
# - Perd optimisations batch fixe (graph optimization)
# - Possible overhead >= gain tiles économisés!
```

#### 4. Effet inverse possible

**Expérience déjà vécue**: Split tiles x4/x8 = PIRE que baseline!

Avec tiling adaptatif:
- Overhead motion/ROI detection: ~1-2ms
- Overhead dynamic batching: ~2-5ms
- Fragmentation mémoire GPU: ~1-2ms
- **Total overhead**: ~5-10ms

Si on économise que 30-40% des tiles (50% → 30-40% actifs en moyenne réelle avec marges sécurité):
- Gain brut: 23ms × 0.35 = ~8ms économisé
- **Overhead: 5-10ms**
- **Gain net: -2ms à +3ms** 🔴

**Verdict**: Peut être contre-productif!

---

## 🎯 Recommandation

### ❌ NE PAS implémenter tiling adaptatif MAINTENANT

**Raisons**:

1. **INT8 quantization prioritaire**
   - Gain attendu: **2-4× speedup**
   - Effort: **1 jour** (script déjà fourni!)
   - Risque: **Très faible** (technique éprouvée)
   - Si ça suffit → objectif atteint sans complexité

2. **Split tiles x2 déjà disponible**
   - Gain mesuré: **30%**
   - Effort: **0** (déjà implémenté!)
   - Risque: **Nul** (désactivable en config)

3. **Tiling adaptatif = complexité extrême**
   - Gain théorique: **30-40%** (similaire à split x2!)
   - Effort: **2-3 semaines**
   - Risque: **Élevé** (false negatives, effet inverse possible)
   - Maintenance: **Lourde** (beaucoup d'edge cases)

### ✅ Si vraiment besoin après INT8 + split x2

**Ordre d'implémentation**:

**Phase 1: Preuve de concept simple**
- Motion detection basique (frame diff GPU)
- Tile selection sans dynamic batching (padding à batch fixe)
- Mesure gain réel vs overhead
- **Critère GO/NO-GO**: Gain net >= 20% ET pas de faux négatifs

**Phase 2: ROI mapping si Phase 1 concluante**
- Utiliser détections global comme guide
- Intersection boxes ↔ tiles
- Tests exhaustifs edge cases

**Phase 3: Seulement si gains confirmés**
- Dynamic batching optimisé
- Hybrid motion + ROI
- Production-ready

---

## 📊 Scénario optimal pour tiling adaptatif

Le tiling adaptatif serait rentable SI:

1. **Scènes très vides** (< 20% surface avec activité)
   - Mall fermé la nuit
   - Parking vide
   - → Mais alors pourquoi 4K? Suffir downscale!

2. **Motion prévisible** (pas de faux négatifs)
   - Détection véhicules (mouvement rapide)
   - → Mais PeopleCounter = personnes (mouvement lent/statique)

3. **Latence tolérable** (1-2 frames délai OK)
   - Pas temps-réel strict
   - → Mais objectif 30FPS = latence critique

**Conclusion**: Le use-case PeopleCounter n'est **PAS optimal** pour tiling adaptatif.

---

## 🔬 Alternative: Downscale dynamique

Au lieu de tiling adaptatif, considérer:

```
SI scène vide (global détecte 0 personnes depuis 5 frames):
  → Passer en mode "680x680 global only" (skip tiles completely)
  → Économie: 100% des tiles!

SI global détecte >= 1 personne:
  → Revenir mode tiles (précision haute résolution)
```

**Avantages**:
- Plus simple que tiling adaptatif
- Gain maximal sur scènes vides
- Pas de false negatives (global tourne toujours)

**Inconvénient**:
- Latence 1 frame si personne entre soudainement

**Effort**: ~2-3 jours vs 2-3 semaines tiling adaptatif

---

## 📝 Conclusion

**Tiling adaptatif = over-engineering** pour ce projet.

**Stratégie recommandée**:
1. INT8 quantization (IMMÉDIAT)
2. Split tiles x2 si nécessaire
3. Downscale dynamique si scènes vides fréquentes
4. Tiling adaptatif = DERNIER RECOURS seulement

**Si objectif 10ms atteint avec INT8**: STOP là et célèbre! 🎉

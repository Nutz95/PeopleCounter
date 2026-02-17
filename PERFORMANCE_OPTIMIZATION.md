# YOLO Performance Optimization Guide

Ce guide explique comment optimiser les performances d'inférence YOLO dans app_v2.

## 📊 État actuel des performances

**Baseline (FP16, yolo26n-seg.engine, split OFF)**
- `yolo_global`: ~6.3ms ✅
- `yolo_tiles`: ~33ms 🔴 (dépasse budget 20ms)
- `end_to_end`: ~63ms 🔴 (dépasse cible 33ms pour 4K@30FPS)

**Avec split tiles x2 (split ON, 2 groupes)**
- `yolo_global`: ~9.4ms ✅
- `yolo_tiles`: ~23ms 🟡 (gain 30%, mais encore au-dessus budget)
- `end_to_end`: ~49ms 🟡 (gain 22%)

## 🎯 Optimisations disponibles

### Option 1: Quantization INT8 (RECOMMANDÉ - Gain attendu: 2-4×)

La quantization INT8 utilise les Tensor Cores INT8 du RTX 5060 Ti pour accélérer massivement l'inférence.

#### Étape 1: Préparer les images de calibration

Le script peut télécharger automatiquement un subset COCO:

```bash
./3b_prepare_int8_engines.sh yolo26n-seg 500 yes
```

OU si tu as déjà des images de calibration:

```bash
# Placer 500-1000 images dans ./coco_val_subset/
./3b_prepare_int8_engines.sh yolo26n-seg 500 no
```

⏱️ **Durée**: 10-30 minutes selon taille dataset de calibration

#### Étape 2: Activer l'engine INT8

Éditer `app_v2/config/pipeline.yaml`:

```yaml
models:
  yolo_global:
    enabled: true
    engine: models/tensorrt/yolo26n-seg-int8.engine  # ← Changé
  yolo_tiles:
    enabled: true
    engine: models/tensorrt/yolo26n-seg-int8.engine  # ← Changé
```

#### Étape 3: Tester les performances

```bash
./5_run_tests.sh --app-version v2 -k test_pipeline_e2e_real_stream_includes_inference_timings
```

Consulter le rapport HTML généré: `app_v2/tests/integration/pipeline/artifacts/pipeline_e2e_metrics.html`

**Gain attendu**: `yolo_tiles` devrait passer de ~23ms à ~6-12ms (objectif <= 10ms atteignable!)

---

### Option 2: Split tiles parallèle (Gain mesuré: 30%)

⚠️ **ATTENTION**: Cette optimisation présente des limites de scaling!

#### Comportement observé (non-linéaire!)

| Groupes | yolo_tiles | Résultat |
|---------|------------|----------|
| 1 (OFF) | ~33ms | Baseline |
| 2 | ~22ms | ✅ Gain 30% |
| 4 | ~28ms | ❌ DÉGRADATION |
| 8 | ~54ms | ❌❌ PIRE que baseline |

**Cause**: Python GIL + contention GPU + fragmentation batch size

#### Configuration

Éditer `app_v2/config/pipeline.yaml`:

```yaml
yolo_tiles_parallel:
  enabled: true   # ← Activer le split
  groups: 2       # ⚠️ NE PAS dépasser 2-3!
```

**Recommandation**: Rester à **2 groupes maximum**. Plus de groupes = dégradation garantie.

---

### Option 3: CUDA Stream Async Natif (Futur)

Remplacer `ThreadPoolExecutor` par pattern CUDA pur pour éliminer overhead Python GIL.

**Status**: Implémentation prototype disponible dans `yolo_tiling_parallel_cuda_trt.py`

**Blocage actuel**: Nécessite modification `TensorRTExecutionContext.execute()` pour mode non-blocking.

---

## 🔬 Comprendre les limitations du split parallèle

### Pourquoi ça ne scale pas linéairement?

**1. Python GIL (Global Interpreter Lock)**
- Les threads Python ne s'exécutent PAS vraiment en parallèle
- Le GIL force une exécution séquentielle avec overhead de context switching
- Overhead croît linéairement avec le nombre de threads

**2. Fragmentation du batch size**
- 32 tiles / 2 groupes = 16 tiles/batch → encore efficace ✅
- 32 tiles / 4 groupes = 8 tiles/batch → perte efficacité batch 🟡
- 32 tiles / 8 groupes = 4 tiles/batch → gaspillage total 🔴
- **TensorRT pénalise** les petits batches (overhead kernel launch)

**3. Contention GPU**
- Trop de CUDA streams concurrents = compétition pour les SMs
- Saturation memory bandwidth
- Context switching overhead TRT

**4. Overhead création contextes TRT**
- Chaque groupe = 1 contexte TRT séparé
- Plus de contextes = fragmentation mémoire GPU
- Warnings TRT = symptôme de gestion sub-optimale

### Pourquoi yolo_global ralentit aussi?

C'est la **preuve** que le problème est système:
- Contention sur ressources GPU partagées
- Fragmentation mémoire affecte TOUS les modèles
- Overhead de gestion des multiples contextes TRT

---

## 🎯 Stratégie recommandée

### Phase 1: INT8 Quantization (IMMÉDIAT)

✅ **Actions**:
1. Exécuter `./3b_prepare_int8_engines.sh yolo26n-seg`
2. Mettre à jour `pipeline.yaml` avec engine INT8
3. Valider gain avec tests e2e
4. **Si objectif atteint** (tiles <= 10ms): STOP ici ✅

### Phase 2: Split tiles x2 (SI NÉCESSAIRE)

⚠️ **Seulement SI** INT8 seul ne suffit pas:
1. Activer `yolo_tiles_parallel.enabled: true`
2. Garder `groups: 2` (ne pas augmenter!)
3. Mesurer gain additionnel

### Phase 3: CUDA Async (FUTUR)

🔮 **Si besoin ultime de perf**:
1. Refactorer `TensorRTExecutionContext` pour mode async
2. Implémenter pattern CUDA stream pur
3. Éliminer ThreadPoolExecutor

---

## 📝 Notes importantes

### Calibration INT8

- **Images nécessaires**: 500-1000 suffisent (subset COCO val2017)
- **Temps calibration**: ~10-30 minutes
- **Accuracy**: Perte < 1% en général pour YOLO
- **Cache**: La calibration est sauvegardée (pas besoin de refaire)

### Warnings TRT

Les warnings `createInferRuntime differs` sont **bénins** mais indiquent:
- Architecture sub-optimale (multiples runtimes)
- Solution future: partager un seul runtime TRT global

### Tiling adaptatif

❌ **NON RECOMMANDÉ** pour l'instant:
- Complexité importante
- ROI incertain
- Prioriser INT8 + split x2 d'abord

---

## 🧪 Validation des performances

### Tests ciblés

```bash
# Test e2e complet avec métriques détaillées
./5_run_tests.sh --app-version v2 -k test_pipeline_e2e_real_stream_includes_inference_timings

# Consulter rapport HTML
firefox app_v2/tests/integration/pipeline/artifacts/pipeline_e2e_metrics.html
```

### Métriques à surveiller

- `inference_model_yolo_tiles_ms`: objectif **<= 10ms**
- `end_to_end_ms`: objectif **<= 33ms** (4K@30FPS)
- `inference_prepare_batch_ms`: overhead préparation batch
- `inference_enqueue_ms`: overhead enqueue TRT
- `inference_stream_sync_ms`: temps synchronisation
- `inference_decode_ms`: décodage post-inference

---

## 📚 Références

- **Migration plan**: [plans/app_v2_migration_plan.md](plans/app_v2_migration_plan.md)
- **Session notes**: `/memories/session/plan.md`
- **TensorRT INT8 docs**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8
- **Ultralytics export**: https://docs.ultralytics.com/modes/export/

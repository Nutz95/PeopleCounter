# Optimization Roadmap — PeopleCounter app_v2
> Créé : 2026-02-18 | GPU : RTX 5060 Ti (Blackwell sm_120) | TRT : 10.15.1.29 | Cible : 4K@30FPS

---

## Contexte & baseline

### Clarifications architecturales
- **Format YOLO** : `images` → NCHW FP32 (`[batch, 3, height, width]`). Ni kHWC ni kNV12 supportés nativement. La conversion NV12→RGB est obligatoire, mais fusionnable.
- **Engines actuels** : `yolo26n-seg-fp8-qdq.engine` (dynamic batch, FP8 Q/DQ), `yolo26n-seg-fp8-b32.engine` (batch32 FP8). Plus de variantes batch1/16/32 séparées — profil dynamique.
- **Pipeline tiles** : 1 batch global (yolo_global) sur stream dédié + 2 groupes parallèles de ~16 tiles chacun (yolo_tiles, `YoloTilingParallelTRT`, 2 contexts TRT séparés).

### Métriques baseline (38 tests, caméra réelle, ASYNC_OVERLAY)
| Métrique | Valeur | Budget | Statut |
|---|---|---|---|
| `end_to_end_ms` | **37.6 ms** | 33 ms | 🔴 +14% |
| `fusion_wait_ms` | **20.1 ms** | 8 ms | 🔴 +250% |
| `inference_model_yolo_tiles_ms` | 19.9 ms | 20 ms | 🟢 99% |
| `inference_model_yolo_global_ms` | 3.6 ms | 15 ms | 🟢 |
| `preprocess_nv12_bridge_ms` | 8.7 ms | 16 ms | 🟢 |
| `nvdec_ms` | 1.3 ms | 42 ms | 🟢 |
| Tests passants | **42 / 42** | — | ✅ |

### Fichier de suivi des gains
→ [optimization_gains.html](optimization_gains.html) (généré automatiquement après chaque étape)

---

## Plan d'optimisation

### #1 — Timing Cache persistant ✅ PRIORITÉ IMMÉDIATE
**Effort** : ~1h | **Impact** : builds déterministes, rebuild 5-10× plus rapide

**Problème** : `convert_onnx_to_trt.py` et `prepare_yolo_modelopt_fp8.py` ne persistent pas le timing cache TRT. Les tactics peuvent varier entre rebuilds sur sm_120 (FP8 Q/DQ).

**Implémentation** :
- Modifier `convert_onnx_to_trt.py` : charger/sauvegarder `models/tensorrt/timing_cache.bin`
- Modifier `prepare_yolo_modelopt_fp8.py` : idem
- Ajouter classe `TimingCacheManager` dans `app_v2/infrastructure/`
- Tests : `test_timing_cache_manager.py` — vérifier que le cache est créé et rechargé

**Référence TRT** : `sampleEditableTimingCache`, `demo/BERT/builder.py`

---

### #2 — CUDA Graphs pour l'inférence répétitive
**Effort** : 2-3j | **Impact estimé** : −2 à −5 ms sur `enqueue_ms` tiles

**Problème** : `execute_async_v3()` re-soumet à chaque frame le graphe complet de kernels CUDA → overhead CPU non négligeable sur tiles b=16 à 30 FPS.

**Principe** : capturer le graphe une fois au warmup, puis `cudaGraphLaunch()` à chaque frame (quasi-zéro overhead CPU).

**Contraintes** :
- Nécessite des shapes fixes (incompatible avec dynamic batch en capture mode) → on capture sur `opt_batch_size`
- La TRT execution context doit supporter un mode "graph_captured"
- Pas compatible avec re-shape mid-run → fallback sur `execute_async_v3` si batch ≠ captured

**Implémentation** :
- Ajouter `CudaGraphCache` dans `app_v2/infrastructure/`
- Étendre `TensorRTExecutionContext` avec option `enable_cuda_graphs: bool`
- Tests : `test_cuda_graph_execution.py` — vérifier graph capture + launch + résultats identiques

**Référence TRT** : utilisé dans les démos Diffusion TRT, cuda-python APIs

---

### #3 — AutoCast FP32→FP16 mixte (ModelOpt AutoCast)
**Effort** : ~1j | **Impact** : alternative sans calibration, potentiellement meilleur que full-FP8 sur tiles

**Principe** : `modelopt.onnx.autocast.convert_to_mixed_precision()` identifie automatiquement les nœuds sensibles (normalization, activations) et les garde en FP32, le reste en FP16. Pas besoin d'images de calibration.

**Intérêt** : comparaison A/B FP8-QDQ vs FP16-mixte sur les tiles 640×640. Le FP16-mixte peut être plus rapide si les Q/DQ nodes introduisent de l'overhead.

**Implémentation** :
- Nouveau script `prepare_yolo_autocast_fp16.py` (similaire à `prepare_yolo_modelopt_fp8.py`)
- Output : `models/tensorrt/yolo26n-seg-autocast-fp16.engine`
- Tests : `test_autocast_fp16_engine.py` — vérifier que l'engine charge et produit des résultats cohérents

**Référence TRT** : `samples/python/strongly_type_autocast/`

---

### #4 — Weight Stripping + Engine Refit
**Effort** : ~1j | **Impact** : engine disque −60% taille, refit à chaud possible (A/B models)

**Principe** : engine "stripped" sans les poids → déploiement allégé. Au démarrage, refit depuis le fichier `.onnx` original via `IParserRefitter` (~30 ms). Performances identiques.

**Intérêt additionnel** : pouvoir swapper les poids sans rebuild complet de l'engine (ex: switch yolo26n ↔ yolo26m sans rebuild TRT).

**Implémentation** :
- Modifier `convert_onnx_to_trt.py` : ajouter option `--weight-stripped`
- Créer `TensorRTEngineRefitter` dans `app_v2/infrastructure/`
- Intégrer dans `TensorRTEngineLoader.load()` : detect stripped engine → refit auto
- Tests : `test_engine_refit.py` — vérifier refit + inférence cohérente

**Référence TRT** : `samples/python/sample_weight_stripping/`

---

### #5 — Kernel fusionné NV12→RGB+Resize+NCHW *(le plus gros gain potentiel)*
**Effort** : 3-5j | **Impact estimé** : **−4 à −8 ms** sur `preprocess_nv12_bridge_ms` (8.7 ms → ~1-2 ms)

**Problème actuel** :
1. `nv12_cuda_bridge.py` → NV12 à GPU ptr → RGB HWC uint8 tensor (8.7 ms) 
2. `preprocess.py` → letterbox/tiling → NCHW FP16 tensor (0.005 ms, déjà rapide)

Le goulot est l'étape 1. Un kernel fusionné ferait en **un seul pass** : NV12 plane ptr → letterbox/tile → NCHW FP16 normalisé [0,1]. Zéro tensor intermédiaire.

**Format YOLO confirmé** : `images` NCHW FP32 (dtype=1). L'execution context castait en FP16 à la volée → le kernel peut sortir directement en FP16.

**Dépendances Docker** :
- `cuda-python` (déjà présent depuis TRT 10.14 migration)
- Triton (`triton` package) ou compilation C extension PyTorch → à tester dans image
- Alternative pure PyTorch : `torch.ops.torchvision` ou custom kernel via `torch.utils.cpp_extension`

**Implémentation** :
- Écrire `app_v2/kernels/fused_nv12_preprocess.py` avec kernel Triton ou cuda-python
- Fallback sur `nv12_cuda_bridge` si kernel indisponible (dégradation gracieuse)
- Tests : `test_fused_nv12_preprocess.py` — comparer output avec pipeline existant (MSE < ε)

---

### #6 — IStreamWriter (sérialisation engine → mémoire/réseau)
**Effort** : ~0.5j | **Impact** : déploiement sans I/O disque, chargement engine depuis RAM

**Principe** : `builder.build_serialized_network_to_stream()` + classe `IStreamWriter` custom → sérialiser directement vers un buffer mémoire ou un socket. Utile pour Docker sans volume `models/`.

**Implémentation** :
- Créer `app_v2/infrastructure/engine_stream_serializer.py`
- Modifier `TensorRTEngineLoader` pour accepter `bytes` en plus de path

**Référence TRT** : `samples/python/stream_writer/`

---

## Ordre d'exécution recommandé

```
#1 Timing Cache     → builds déterministes (prérequis pour comparer les suivants)
#3 AutoCast FP16    → comparaison A/B vs FP8, sans risque
#4 Weight Stripping → infrastructure propre pour la suite
#2 CUDA Graphs      → optimisation runtime inference
#5 Kernel fusionné  → plus gros impact, plus risqué, en dernier
#6 IStreamWriter    → infrastructure, peut se faire en parallèle
```

---

## Tableau de suivi des gains

> Les métriques `end_to_end_ms` / `inference_tiles_ms` / `bridge_ms` nécessitent une caméra réelle (NVDEC_TEST_STREAM_URL).

| # | Optimisation | Fichiers créés / modifiés | Tests | `end_to_end_ms` | `inference_tiles_ms` | `bridge_ms` | Statut |
|---|---|---|---|---|---|---|---|
| 0 | **Baseline** | — | 42 ✅ | 37.6 ms 🔴 | 19.9 ms | 8.7 ms | ✅ référence |
| 1 | **Timing Cache** | `timing_cache_manager.py`, `convert_onnx_to_trt.py`, `prepare_yolo_modelopt_fp8.py` | +14 → 56 ✅ | *(infra, pas d'impact latence)* | *(infra)* | — | ✅ mesuré |
| 2 | **CUDA Graphs** | `cuda_graph_cache.py`, `tensorrt_execution_context.py` | +8 → 64 ✅ | à mesurer (caméra requise) | à mesurer | — | ✅ implémenté |
| 3 | **AutoCast FP16** | `prepare_yolo_autocast_fp16.py`, `2_prepare_nvdec.sh` | +7 → 71 ✅ | à mesurer | −2.4 % GPU tiles | — | ✅ mesuré trtexec |
| 4 | **Weight Stripping** | `engine_refitter.py`, `convert_onnx_to_trt.py` | +11 → 82 ✅ | *(perf = FP32)* | *(perf = FP32)* | — | ✅ mesuré trtexec |
| 5 | **Fused NV12+letterbox** | `nv12_cuda_bridge.py`, `preprocess.py` | +7 → 89 ✅ | à mesurer | à mesurer | 🎯 −4 ms visé | ✅ implémenté |
| 6 | **IStreamWriter** | `engine_stream_writer.py` → `stream_writers/` | +12 → 101 ✅ | *(infra)* | *(infra)* | — | ✅ implémenté |

**Total tests** : 42 (baseline) → **101** (+59 nouveaux tests)

---

## Benchmarks moteur — GPU Compute median (RTX 5060 Ti sm_120, 2026-02-18)

> Commande : `trtexec --loadEngine=<engine> --shapes=images:Bx3x640x640 --warmUp=200 --iterations=100 --avgRuns=10`
> Valeurs = GPU Compute Time median (hors H2D/D2H). Latency totale ≈ GPU + 1.55 ms (H2D) + 1.37 ms (D2H).

### yolo26n-seg (7.8 MB FP32) — tiles parallèles

| Format | Taille | batch=1 | batch=8 | batch=16 | batch=32 | Recommandation |
|--------|--------|---------|---------|----------|----------|----------------|
| **FP32** (baseline) | 7.8 MB | 2.63 ms | 5.21 ms | 9.38 ms | 18.59 ms | référence |
| **FP16** (opt #3) | 7.9 MB | 3.26 ms ⚠️ | **4.98 ms** | **9.08 ms** | 18.93 ms | batch≥8 seulement |
| **FP8-QDQ** | 6.5 MB | 2.66 ms | 5.01 ms | **8.66 ms** | **17.71 ms** | ✅ batch≥16 |
| **Stripped** (opt #4) | **4.1 MB** | =FP32 | =FP32 | =FP32 | =FP32 | déploiement allégé |

⚠️ FP16 est **plus lent** que FP32 à batch=1 sur yolo26n (modèle trop petit pour amortir les casts).

### yolo26m-seg (48 MB FP32) — tile globale (batch=1) + plus grand modèle

| Format | Taille | batch=1 | batch=8 | batch=16 | batch=32 | Recommandation |
|--------|--------|---------|---------|----------|----------|----------------|
| **FP32** (baseline) | 48 MB | 5.53 ms | 22.71 ms | 45.85 ms | 92.87 ms | référence |
| **FP16** (opt #3) | ~30 MB | **5.24 ms** | 22.35 ms | 46.26 ms | 94.15 ms | batch=1 uniquement |
| **FP8-QDQ** | 30 MB | 5.05 ms | **19.75 ms** | **39.34 ms** | **80.98 ms** | ✅ **tous batch** |

FP8-QDQ donne −13 % GPU compute sur yolo26m à batch≥8. Très significatif pour ce modèle.

### Analyse architecture global tile + sous-tiles

L'idée : utiliser **yolo26m-seg FP8-QDQ (batch=1, ~5 ms)** pour la tile globale, et **yolo26n-seg FP8-QDQ (batch≤16, ~8.7 ms)** pour les sous-tiles en parallèle.

```
Stream A (tile globale) :   yolo26m-seg FP8-QDQ  →  5.05 ms GPU
Stream B (sous-tiles ×N) :  yolo26n-seg FP8-QDQ  →  8.66 ms GPU (batch=16)
                                                      ↑ budget 33 ms – 5 ms (transfer) = 28 ms = confortable
```

Les deux chemins tournent en parallèle. La fusion (opt #2 à venir) doit ensuite attendre le plus lent des deux — avec ces chiffres le chemin sous-tiles est le goulot (~12 ms total avec transfers).

Pour ~16 tiles simultanées (batch=16) vs 1 frame globale :
- yolo26m FP8 batch=1 = 5 ms ← terminerait bien avant le batch de tiles
- yolo26n FP8 batch=16 = 8.66 ms ← goulot
- **Équilibre** : on pourrait grouper jusqu'à 16 tiles en un seul batch et le global tile finit ~3.6 ms avant → très bon overlap

---

---

## Commandes de référence

```bash
# Tests complets (baseline)
./5_run_tests.sh --app-version v2

# Ou directement dans Docker
./docker_exec.sh python -m pytest app_v2/tests/ -v

# Rapport e2e avec caméra (si NVDEC_TEST_STREAM_URL défini)
NVDEC_TEST_STREAM_URL=rtsp://... ./docker_exec.sh python -m pytest \
  app_v2/tests/integration/pipeline/test_pipeline_metrics_integration.py \
  -v -s

# trtexec benchmark engine actuel
./docker_exec.sh trtexec \
  --loadEngine=models/tensorrt/yolo26n-seg-fp8-qdq.engine \
  --batch=16 --warmUp=500 --iterations=100 --avgRuns=10
```

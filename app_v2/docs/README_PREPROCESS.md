# Preprocess v2 (GPU-only)

## Objectif

Maintenir un pipeline preprocess **100% GPU** entre NVDEC et TensorRT, avec réutilisation mémoire, métriques exploitables, et routage par modèle.

## Ce qui se passe concrètement après NVDEC

1. NVDEC fournit des surfaces **NV12** en mémoire GPU (pointeurs Y/UV + pitch).
2. `nv12_cuda_bridge` copie les plans en D2D (`cudaMemcpy2DAsync`) et convertit en **RGB HWC U8** sur GPU.
3. `GpuPreprocessor` calcule les plans (`global` + `tiles`) via `GpuPreprocessPlanner`.
4. Les kernels preprocess produisent des tensors **RGB NCHW FP16** depuis un `GpuTensorPool`.
5. Les inputs sont routés par modèle (`flatten_inputs(model_name)`).
6. Les leases du pool sont relâchés après publish (release différé frame-level).

## Clarification letterbox vs tiling

- **Tiling**: découpage en crops avec overlap (ton cas 4K).
- **Letterbox**: redimensionnement en conservant le ratio d’aspect + padding pour rentrer dans la taille cible (ex: 640x640 global), sans déformation.

## Métriques disponibles (FrameTelemetry)

- Décodage: `nvdec_ms`
- Bridge NV12: `preprocess_nv12_bridge_ms`
- Préprocess total: `preprocess_ms`
- Préprocess par modèle: `preprocess_model_<model>_ms`
- Préprocess par tâche: `preprocess_kernel_<model>_<idx>_ms`
- Pool:
  - `tensor_pool_hits`
  - `tensor_pool_misses`
  - `tensor_pool_allocations`
  - `tensor_pool_in_use`
  - `tensor_pool_available`
  - `tensor_pool_waits`
  - `tensor_pool_wait_ms`

## Configuration

Le pool est pilotable via `app_v2/config/pipeline.yaml`:

- `tensor_pool.max_per_key`: capacité max par clé `(shape,dtype,format,stream,device)`

Valeur courante recommandée: `64` (4K@30fps, YOLO global + tiles, 16GB VRAM).

## Test d’intégration métriques

Le test de référence est:

- `app_v2/tests/integration/pipeline/test_pipeline_metrics_integration.py`

Ce test sert de “tableau de bord de base” pour suivre le coût des étapes preprocess au fil des évolutions.

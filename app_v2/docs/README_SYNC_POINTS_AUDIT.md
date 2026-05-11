# Synchronization Points Audit — app_v2

But: lister les points de synchro/blocking qui peuvent créer des ralentissements disproportionnés et expliquer des frames sans points.

## Scope

- Pipeline principal: `app_v2/application/pipeline_orchestrator.py`
- Source RTSP/NVDEC: `app_v2/infrastructure/rtsp_frame_source.py`
- Encode vidéo: worker NVJPEG dans `PipelineOrchestrator`

## 1) Points de synchro identifiés

## A. Décodage / source

1. `RTSPFrameSource.next_frame()`
   - `self._decode_cond.wait(timeout=1.0)`
   - Nature: **thread wait** (CPU)
   - Impact visible: `frame_source_wait_latest_ms`

2. `RTSPFrameSource._copy_frame_into_slot()`
   - `self._copy_stream.synchronize()`
   - Nature: **CUDA stream sync** (par frame décodée)
   - Impact visible: `frame_source_copy_sync_ms`

## B. Pipeline principal

3. `PipelineOrchestrator.run()` en mode `RAW_STREAM_WITH_METADATA`
   - `self._video_stream.synchronize()` avant release
   - Nature: **CUDA stream sync** conditionnelle (selon fusion strategy)
   - Impact: peut réduire l’overlap vidéo/inférence

4. Passthrough branch
   - `self._video_stream.synchronize()` avant discard frame
   - Nature: **CUDA sync**

## C. Encode worker

5. `_encode_and_push_nvjpeg()`
   - `enc_event.synchronize()`
   - Nature: **attente CPU d’un event CUDA**
   - Impact visible: `video_encode_wait_event_ms`

6. `_encode_and_push_nvjpeg()`
   - `nvjpeg_stream.synchronize()`
   - Nature: **CUDA stream sync**
   - Impact visible: composante `video_encode_kernel_ms`

7. `_encode_and_push_cpujpeg()` (si backend CPU)
   - `enc_event.synchronize()`
   - Nature: **attente event CUDA**

## D. Threading / queueing

8. Encode executor mono-worker
   - `ThreadPoolExecutor(max_workers=1)` + stash latest
   - Nature: **sérialisation** volontaire du chemin encode
   - Impact visible: `video_encode_inflight`, `video_encode_stashed`

## 2) Ce qui est déjà instrumenté

- `frame_source_wait_latest_ms`
- `frame_source_copy_sync_ms`
- `nvdec_ms`
- `video_encode_wait_event_ms`
- `video_encode_gpu_hotspot_render_ms`
- `video_encode_cpu_copy_ms`
- `video_encode_push_ms`

Ajouté dans ce cycle:

- `video_hotspot_lookup_ms`
- `video_hotspot_lookup_mode_code` (0 none, 1 exact, 2 fallback, 3 miss)
- compteurs cumulés:
  - `video_hotspot_lookup_exact`
  - `video_hotspot_lookup_fallback`
  - `video_hotspot_lookup_miss`

## 3) Lecture technique (court)

- Les benchmarks unitaires NVJPEG/NVDEC montrent que la perf brute hardware est largement suffisante pour 30 fps.
- Les ralentissements disproportionnés sont plus plausibles sur:
  1. épisodes de recovery flux/décodage,
  2. points de synchro non amortis en burst,
  3. mismatch frame↔hotspots pendant les phases de backlog.

## 4) Conclusion opérationnelle

Le prochain diagnostic prioritaire est une corrélation frame par frame entre:

- `frame_source_wait_latest_ms` et `nvdec_ms`
- `video_hotspot_lookup_mode_code` (exact/fallback/miss)
- présence/absence visuelle de points sur MJPEG

C’est cette corrélation qui permet de trancher si les trous visuels viennent d’un blocage synchro, d’un fallback systémique, ou d’un problème de flux source.

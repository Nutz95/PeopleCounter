# NVDEC Benchmark Reference (RTSP, jitter & frame wait)

Objectif: mesurer **le décodeur NVDEC** sur flux réel et quantifier :
- temps entre frames (`inter_frame_ms`),
- jitter,
- `frame_source_wait_latest_ms`,
- coût NVDEC (`nvdec_ms`),
- impact d’une inférence parallèle.

---

## Setup de la mesure

- Date: 2026-05-11
- GPU: NVIDIA GeForce RTX 5060 Ti
- Container: `people-counter:gpu-final-nvdec`
- Script: `diagnostics/bench_nvdec_jitter.py`
- Flux testé: `rtsp://host.docker.internal:5002/live`
- Durée: 60 s par scénario
- Cible: 30 fps (période nominale 33.33 ms)

### Commande exécutée

```bash
docker run --rm --gpus all \
  --add-host host.docker.internal:host-gateway \
  -e PYTHONPATH=/app \
  -v "$PWD:/app" -w /app \
  people-counter:gpu-final-nvdec \
  python3 diagnostics/bench_nvdec_jitter.py \
    --stream-url rtsp://host.docker.internal:5002/live \
    --duration-s 60 \
    --fps-target 30 \
    --infer-width 1920 \
    --infer-height 1080 \
    --tag rtsp5002_60s
```

Artifacts générés:

- `diagnostics/artifacts/nvdec_bench_rtsp5002_60s/decode_only.csv`
- `diagnostics/artifacts/nvdec_bench_rtsp5002_60s/decode_plus_inference.csv`
- `diagnostics/artifacts/nvdec_bench_rtsp5002_60s/inter_frame_ms.png`
- `diagnostics/artifacts/nvdec_bench_rtsp5002_60s/frame_wait_ms.png`
- `diagnostics/artifacts/nvdec_bench_rtsp5002_60s/nvdec_copy_sync_ms.png`
- `diagnostics/artifacts/nvdec_bench_rtsp5002_60s/jitter_hist_ms.png`
- `diagnostics/artifacts/nvdec_bench_rtsp5002_60s/summary.json`

---

## Résultats synthèse

| Scenario | FPS | inter mean (ms) | inter p95 (ms) | wait mean (ms) | wait p95 (ms) | nvdec mean (ms) | nvdec p95 (ms) | infer mean (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_only | 29.46 | 33.92 | 55.49 | 33.85 | 55.39 | 30.45 | 52.78 | 0.00 |
| decode_plus_inference | 28.83 | 34.65 | 55.32 | 30.59 | 51.63 | 33.02 | 54.97 | 3.81 |

Jitter (vs 33.33 ms):

- decode_only: `jitter_std_ms = 55.26`, `jitter_abs_mean_ms = 10.62`
- decode_plus_inference: `jitter_std_ms = 45.58`, `jitter_abs_mean_ms = 9.82`

---

## Observations importantes

1. **La contention inférence parallèle n’explique pas seule les gros waits**:
   - avec inférence synthétique (~3.81 ms), on perd ~0.62 fps,
   - mais les `wait_p95` restent dans la même zone (~52–55 ms).

2. **Le flux présente des événements decode error/recreate** (logs NVDEC pendant le test):
   - `Decode Error occurred for picture ...`
   - `HW decoder faced error. Re-create instance.`

3. Les pics `inter_frame_p95` / `wait_p95` > 35 ms sont donc compatibles avec des
   perturbations du flux (ou GOP/recovery/packetization) et pas uniquement avec
   un manque de perf brute du GPU.

---

## Interprétation opérationnelle

- Le décodeur matériel est globalement capable de tenir ~30 fps en moyenne,
  mais le jitter observé et les pics > 50 ms viennent surtout de la **stabilité
  du flux + épisodes de recovery NVDEC**, pas d’un encodeur NVJPEG saturé.
- Pour expliquer les "images sans points", la piste prioritaire reste:
  1. corrélation des épisodes de `wait`/recovery,
  2. télémétrie lookup hotspots (`exact / fallback / miss`),
  3. gestion du timing frame↔overlay pendant ces épisodes.

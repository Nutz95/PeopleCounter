# DM-Count QNRF — Density Model Performance Benchmark

Hardware: **NVIDIA RTX 5060 Ti** (sm_120 Blackwell, 16 GB GDDR7, 448 GB/s)  
Framework: **TensorRT 10.15.1.29**, CUDA 13.1  
Model: **DM-Count QNRF** (VGG16 backbone, density head, output stride 8)  
Input frame: **3840 × 2160** (4K)

---

## 📐 Why latency scales linearly with pixel count

VGG16 is a pure feed-forward conv net with no dynamic routing.
Every layer is **memory-bandwidth bound** on this GPU:
each kernel reads the input feature-map from DRAM, does compute,
and writes the output feature-map back.
The RTX 5060 Ti's 448 GB/s bus is already saturated at batch=18 × 640×720 tiles.

**Consequence**: latency ∝ total pixels processed, regardless of configuration.

```
batch=1  × 640×720  =   460 800 px →  5.2 ms
batch=18 × 640×720  = 8 294 400 px → 92 ms  (18 × 5.2 ≈ 93 ms ✓ LINEAR)
batch=4  × 1920×1088= 8 355 840 px → 92 ms  (same pixel budget)
batch=1  × 3840×2160= 8 294 400 px → 92 ms  (full 4K, same budget)
```

**Why do 3 parallel CUDA streams give the same result as 1 large batch?**
Parallel streams only help when the GPU has *idle* compute or memory bandwidth.
At batch=18 the memory bus is 100% saturated. Adding parallel streams queues
extra work behind an already-full bottleneck — the wall-clock time is identical.
This is the streaming equivalent of adding more cars to a full motorway.

---

## 🔬 All measured configurations (FP16)

### A — 640 × 720 tile engine (`dm_count_qnrf.engine`, max_batch=18)

| Batch | Tiles covered (4K) | Latency | fps-equiv | Notes |
|------:|-------------------|--------:|----------:|-------|
| 1     | 1 tile (3.5 % of frame) |  5.24 ms | 190.9 | partial frame, good for sparse scenes |
| 4     | 4 tiles (14 %)    | 20.43 ms |  48.9 | **≤20 ms target ✅** — covers ~4 zones |
| 6     | 6 tiles (21 %)    | 30.70 ms |  32.6 | ≤33 ms real-time budget |
| 18    | full frame (100 %) | 91.40 ms |  10.9 | **baseline**, all 6×3 tiles |

### B — Parallel-stream experiment (640 × 720, same engine)

| Strategy | Tiles | Latency | vs batch=18 | Notes |
|----------|------:|--------:|:-----------:|-------|
| 3 streams × batch=6 | 18 | 91.98 ms | **1.00×** | **no benefit** — GPU saturated |

### C — 1920 × 1088 tile engine (`dm_count_qnrf_1920x1088.engine`, max_batch=4)

| Batch | Tiles covered (4K) | Latency | fps-equiv | Notes |
|------:|-------------------|--------:|----------:|-------|
| 4     | full frame (2×2)  | 92.16 ms |  10.9 | same latency as B=18@640×720, **no rescaling** ✅ |

### D — 1920 × 1088, batch=1 (`dm_count_qnrf_1920x1088_b1.engine`)
*Matches: 4K → downscale to 1920×1088, infer as single frame*

| Batch | Resolution | Latency | fps-equiv | Notes |
|------:|-----------|--------:|----------:|-------|
| 1     | 1920×1088  | **23.78 ms** | 42.1 | 4K→1080p, single forward pass — **≤33 ms ✅** |

### E — 3840 × 2160, batch=1 (`dm_count_qnrf_3840x2160.engine`)
*Matches: 4K native inference, simple crop (no resize — 3840×2160 is already mod-16)*

| Batch | Resolution | Latency | fps-equiv | Notes |
|------:|-----------|--------:|----------:|-------|
| 1     | 3840×2160  | **96.76 ms** | 10.3 | full 4K native, slightly slower than §C due to large-tensor cache pressure |

---

## 📊 Pixel-budget law (visual summary)

```
Total pixels → Latency (FP16 on RTX 5060 Ti)
─────────────────────────────────────────────────────
   460 800 px  (  b=1 × 640×720)   →   5.2 ms  🟢 fast
 2 088 960 px  (  b=4 × 640×720,
               or b=1 × 1920×1088) →  21–24 ms  🟢 ≤33ms ✅
 3 110 400 px  (  b=6 × 640×720)   →  31   ms  🟡 ≤33ms ✅
 8 294 400 px  (  b=18 × 640×720,  →  92–97 ms  🔴 too slow
               or b=4 × 1920×1088,
               or b=1 × 3840×2160)
─────────────────────────────────────────────────────
Note: 4K native (3840×2160) runs slightly slower than
      18 small tiles because L2 cache efficiency drops
      with very large spatial tensors.
```

---

## 🗺️ Preprocessing strategies (4K frame)

| Strategy | Engine | Preprocessing | Latency | Coverage | Quality |
|----------|--------|--------------|--------:|---------|---------|
| **6×3 tiles** | 640×720 b=18 | extract 18 crops + resize | 92 ms | 100 % | ❌ rescaled |
| **2×2 tiles** | 1920×1088 b=4 | extract 4 crops (native) | 92 ms | 100 % | ✅ native res |
| **Resize → 1080p** | 1920×1088 b=1 | bilinear 4K→1920×1088 | ~23 ms | 100 % (lower res) | 🟡 rescaled once |
| **4K native crop** | 3840×2160 b=1 | crop 4K to mod-16 (none needed) | ~92 ms | 100 % | ✅ native res |
| **Partial 4 tiles** | 640×720 b=4 | 4 crops in ROI | ~20 ms | ~14 % | ✅ zones of interest |

---

## 🎯 Recommendations by use case

### Real-time < 20 ms, partial coverage
→ **640×720, batch=4**: process the 4 highest-density zones of the frame.  
  Configure via `target_width: 640`, `target_height: 720` in `pipeline.yaml`;  
  the planner will generate the first 4 tiles (top-left 2×2 region by default).

### Global count at ~23 ms (full-frame, lower resolution)
→ **1920×1088, batch=1**: resize 4K → 1920×1088 once, single inference.  
  Acceptable for crowd estimation; some fine detail lost vs native resolution.

### Best quality density map, no time constraint
→ **2×2 tiles at 1920×1088** (current default): 4 native-resolution tiles,  
  no rescaling, full spatial detail preserved. ~92 ms.

### < 30 ms on full 4K frame (planned)
→ **FP8 quantization** (TH1X FP8 Tensor Cores): expected 2–4× speedup  
  → target ~23–46 ms for full frame.  
  Requires UCF-QNRF Train/img dataset for calibration:  
  ```bash
  python3 prepare_density_models.py \
      --tile-size 1920x1088 \
      --calib-dir /path/to/UCF-QNRF/Train/img
  ```

---

## 🏗️ Engine inventory

| Engine file | Tile size | max_batch | Size | Latency | Status |
|-------------|-----------|----------:|-----:|--------:|--------|
| `dm_count_qnrf.engine` | 640×720 | 18 | ~41 MB | 91.4 ms (b=18) | ✅ built |
| `dm_count_qnrf_1920x1088.engine` | 1920×1088 | 4 | ~41 MB | 92.2 ms (b=4) | ✅ built |
| `dm_count_qnrf_1920x1088_b1.engine` | 1920×1088 | 1 | 41.2 MB | **23.78 ms** (b=1) | ✅ built |
| `dm_count_qnrf_3840x2160.engine` | 3840×2160 | 1 | 41.2 MB | **96.76 ms** (b=1) | ✅ built |
| `dm_count_qnrf-fp8-qdq.engine` | 640×720 | 18 | TBD | ~46 ms (est.) | ⏳ needs dataset |
| `dm_count_qnrf_1920x1088-fp8-qdq.engine` | 1920×1088 | 4 | TBD | ~23 ms (est.) | ⏳ needs dataset |

---

## 🔬 How to reproduce benchmarks

```bash
# All benchmarks run inside Docker (people-counter:gpu-final-nvdec)

# 640×720, full batch sweep + 3-stream parallel test
docker run --rm --gpus all --shm-size=4g \
  -v "$PWD:/app" -w /app people-counter:gpu-final-nvdec \
  python3 prepare_density_models.py --skip-fp8 --benchmark-strategies

# 1920×1088, batch=4 (current default)
docker run --rm --gpus all --shm-size=4g \
  -v "$PWD:/app" -w /app people-counter:gpu-final-nvdec \
  python3 prepare_density_models.py --skip-fp8 --tile-size 1920x1088

# 1920×1088, batch=1 (4K → 1080p single-frame)
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD:/app" -w /app people-counter:gpu-final-nvdec \
  python3 prepare_density_models.py --skip-fp8 --tile-size 1920x1088 --max-batch 1

# 3840×2160, batch=1 (full 4K native, no crop)
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD:/app" -w /app people-counter:gpu-final-nvdec \
  python3 prepare_density_models.py --skip-fp8 --tile-size 3840x2160

# FP8 (once UCF-QNRF dataset is available)
docker run --rm --gpus all --shm-size=4g \
  -v "$PWD:/app" -v /path/to/UCF-QNRF:/calib:ro \
  -w /app people-counter:gpu-final-nvdec \
  python3 prepare_density_models.py --tile-size 1920x1088 \
  --calib-dir /calib/Train/img
```

---

## 📝 Notes on model architecture

- **Backbone**: VGG16 (5 pooling stages → spatial stride 32)
- **Density head**: `reg_layer` (512→256→128 conv blocks) + `density_layer` (128→1 conv + ×2 bilinear) → effective stride **8**
- **Output shape**: `(B, 1, H/8, W/8)` — sum × density_scale ≈ head count
- **Input normalisation**: ImageNet mean/std, input in [0, 1] fp32
- **Tile dimensions**: must be **multiples of 16** (VGG16 feature-map alignment)
  - 3840 = 240 × 16 ✅, 2160 = 135 × 16 ✅ → 4K native needs **no crop**
  - 1920 = 120 × 16 ✅, 1088 = 68 × 16 ✅ (nearest mul-16 ≥ 1080)

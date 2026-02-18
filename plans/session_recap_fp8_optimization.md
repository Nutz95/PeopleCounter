# Session Recap — FP8 ModelOpt PTQ Optimization
> Date: 2026-02-18 | Branch: main | Context: Blackwell RTX 5060 Ti (sm_120)

---

## 1. What Was Accomplished

### TensorRT 10.15.1.29 Upgrade
- Upgraded from TRT 10.14 to **TRT 10.15.1.29** (via pip, nvidia NGC)
- TRT 10.14 FP8 was broken on Blackwell sm_120 — 10.15 fixes it
- Both Docker images updated: `people-counter:gpu-final` and `people-counter:gpu-final-nvdec`

### FP8 PTQ Quantization via nvidia-modelopt 0.41.0
- Implemented **ONNX-space PTQ FP8** using `modelopt.onnx.quantization.quantize()`
- Script: `prepare_yolo_modelopt_fp8.py` (generic, supports any `--model` arg)
- Calibration: 50 COCO val2017 images (or random fallback)
- ModelOpt flow: `PTQ INT8 calibration → INT8 to FP8 conversion → Q/DQ ONNX`
- **STRONGLY_TYPED network** mode in TRT builder (required for FP8 Q/DQ, mutually exclusive with `BuilderFlag.FP8` and `BuilderFlag.FP16`)

### Engines Produced
| Engine | Size | Type | Ratio vs FP16 |
|--------|------|------|---------------|
| `yolo26n-seg.engine` | 8.2 MB | FP16, TRT 10.15 | — |
| `yolo26n-seg-fp8-qdq.engine` | **6.8 MB** | FP8 Q/DQ STRONGLY_TYPED | 1.2x |
| `yolo26m-seg.engine` | 50.0 MB | FP16, TRT 10.15 | — |
| `yolo26m-seg-fp8-qdq.engine` | **31.2 MB** | FP8 Q/DQ STRONGLY_TYPED | **1.6x** |

---

## 2. Performance Results

### trtexec GPU Compute (no CPU overhead, batch=16)
| Model | FP16 | FP8 Q/DQ | Gain |
|-------|------|----------|------|
| yolo26n-seg b=16 | 9.33 ms | **8.73 ms** | **+6.4%** |
| yolo26n-seg b=32 | 18.56 ms | **8.17 ms** | **+7.8%** |
| yolo26m-seg b=16 | 45.6 ms | **39.7 ms** | **+13.0%** |

### Pipeline E2E (HTML benchmark report, current config)
| Metric | Value | Budget | Status |
|--------|-------|--------|--------|
| `end_to_end_ms` | 37.6 ms | 33 ms | 🔴 +14% |
| `fusion_wait_ms` | 20.1 ms | 8 ms | 🔴 250% |
| `inference_model_yolo_tiles_ms` | 19.9 ms | 20 ms | 🟢 99% |
| `inference_model_yolo_global_ms` | 3.6 ms | 15 ms | 🟢 24% |
| `preprocess_critical_path_ms` | 8.7 ms | 27 ms | 🟢 32% |
| `preprocess_nv12_bridge_ms` | 8.7 ms | 16 ms | 🟢 54% |
| `nvdec_ms` | 1.3 ms | 42 ms | 🟢 3% |

**42 tests passing** ✅

> **Note**: `fusion_wait_ms=20ms` is structural in `ASYNC_OVERLAY` mode — it's the tile merge wait time.
> Not reducible without changing the fusion strategy architecture.

---

## 3. Active Configuration

### `app_v2/config/pipeline.yaml`
```yaml
models:
  yolo_global:
    engine: models/tensorrt/yolo26n-seg-fp8-qdq.engine  # FP8 Q/DQ STRONGLY_TYPED
  yolo_tiles:
    engine: models/tensorrt/yolo26n-seg-fp8-qdq.engine  # FP8 Q/DQ STRONGLY_TYPED
yolo_tiles_parallel:
  enabled: True
  groups: 2
```

---

## 4. Modified Files

| File | Change |
|------|--------|
| `prepare_yolo_modelopt_fp8.py` | **CREATED** — full ModelOpt PTQ FP8 script |
| `1_prepare.sh` | Added: `lief`, `onnx~=1.19.0`, `onnx-graphsurgeon` (NGC) |
| `app_v2/config/pipeline.yaml` | Using FP8 Q/DQ engines for both yolo_global + yolo_tiles |

---

## 5. Known Issues & Gotchas

### Dependency Hell (solved)
```bash
# WRONG (doesn't exist on PyPI):
pip install nvidia-onnx-graphsurgeon

# CORRECT (NGC registry):
pip install onnx-graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

### onnx Version Pinning — CRITICAL
- `onnx 1.17`: missing `FLOAT4E2M1` → crash
- `onnx 1.19.x`: works with monkey-patches (see script header)
- `onnx 1.20`: `_mapping` API changed → incompatible with modelopt 0.41
- **Always use**: `pip install 'onnx~=1.19.0' --no-deps`

### Required Monkey-Patches (in `prepare_yolo_modelopt_fp8.py`)
```python
# Must be applied BEFORE importing modelopt
import onnx
onnx.TensorProto.FLOAT4E2M1 = 23  # added in onnx 1.20, backport needed

import onnx._mapping
from onnx._mapping import TENSOR_TYPE_TO_NP_TYPE
NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}
# ... (see script for full FakeMapping class)
```

### STRONGLY_TYPED is Mutually Exclusive
```python
# WRONG — BuilderFlag.FP8 + STRONGLY_TYPED = TRT error
config.set_flag(trt.BuilderFlag.FP8)
network.set_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

# CORRECT — STRONGLY_TYPED only, no FP8/FP16 flags
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
)
```

### ⚠️ Pending: Deps Not in nvdec Image
`lief` and `onnx~=1.19.0` were added to `1_prepare.sh` but the nvdec image hasn't been rebuilt.
Until that's done, prefix any quantization command with:
```bash
pip install --quiet --no-cache-dir lief 'onnx~=1.19.0' --no-deps
```

### Cosmetic TRT Warning (ignorable)
```
[SCALE] has invalid precision FP8, ignored
```
This is a TRT internal analysis bug — zero_points are correctly FP8, TRT incorrectly propagates FP8 type to scale tensors in its diagnostics. No performance impact. 434 occurrences for yolo26n, 480 for yolo26m.

---

## 6. Script Usage

### Generate FP8 Q/DQ Engine for any YOLO model
```bash
# Run inside Docker (people-counter:gpu-final)
docker run --rm --gpus all --shm-size=4g \
  -v "$PWD:/app" -w /app \
  people-counter:gpu-final \
  bash -c "pip install --quiet --no-cache-dir lief 'onnx~=1.19.0' --no-deps && \
    python3 prepare_yolo_modelopt_fp8.py --model yolo26n-seg"
```

### Script auto-detects:
- If `models/onnx/yolo26n-seg-dynamic.onnx` missing → exports from `models/pt/yolo26n-seg.pt` via ultralytics
- If COCO calibration images unavailable → uses random Gaussian noise
- Output: `models/tensorrt/yolo26n-seg-fp8-qdq.engine` + `models/onnx/yolo26n-seg-fp8-qdq.onnx`

---

## 7. What's Left To Do

### Priority 1 — Fix fusion bottleneck (biggest remaining gain)
`fusion_wait_ms=20ms` is the #1 blocker for hitting the 33ms E2E target.
- Strategy: investigate `RAW_STREAM_WITH_METADATA` fusion mode in `app_v2/application/`
- The current `ASYNC_OVERLAY` forces synchronization between tile groups
- **TensorRT cloned repo**: consult `/home/nra/TensorRT/` (or wherever it's cloned) for:
  - CUDA streams examples (`samples/python/`)
  - Multi-context/multi-stream best practices
  - Builder optimization flags for sm_120

### Priority 2 — Camera RTSP end-to-end
```bash
./4_run_app.sh --app-version v2 rtsp://<camera_ip>:<port>/stream
```

### Priority 3 — MQTT Output Publisher
- Check `app_v2/infrastructure/` for MQTT client implementation
- Verify message schema with the existing `app_v1/mqtt_capture.py` for compatibility

### Priority 4 — Web UI Streaming
- Flask server with segmentation masks overlay
- Reference: `app_v1/web_server.py` for the v1 implementation pattern

### Priority 5 — Rebuild nvdec image with new deps
```bash
./1_prepare.sh && ./2_prepare_nvdec.sh
docker run --rm people-counter:gpu-final-nvdec python3 -c "import lief, onnx; print(onnx.__version__)"
# Expected: 1.19.x
```

---

## 8. Environment Reference

| Component | Version |
|-----------|---------|
| GPU | RTX 5060 Ti (Blackwell sm_120) |
| CUDA | 13.1.1 |
| TensorRT | **10.15.1.29** (pip, nvidia NGC) |
| nvidia-modelopt | 0.41.0 |
| onnx | 1.19.1 (pinned) |
| onnx-graphsurgeon | 0.3.27 (NGC) |
| lief | 0.17.3 |
| Docker base image | `people-counter:gpu-final` (build) |
| Docker runtime image | `people-counter:gpu-final-nvdec` (tests) |
| Python | 3.11 |

---

## 9. Key Command Reference

```bash
# Run full test suite (42 tests)
docker exec people-counter-nvdec python3 -m pytest app_v2/tests/ -v

# Benchmark current pipeline
docker exec people-counter-nvdec python3 app_v2/main.py --benchmark

# trtexec benchmark FP8 engine
docker run --rm --gpus all -v "$PWD:/app" -w /app people-counter:gpu-final-nvdec \
  trtexec --loadEngine=models/tensorrt/yolo26n-seg-fp8-qdq.engine \
  --batch=16 --warmUp=500 --iterations=100 --avgRuns=10

# Check TRT version
docker run --rm --gpus all people-counter:gpu-final-nvdec \
  python3 -c "import tensorrt as trt; print(trt.__version__)"
# Expected: 10.15.1.29
```

---

*Generated 2026-02-18 — handoff document for new agent session*

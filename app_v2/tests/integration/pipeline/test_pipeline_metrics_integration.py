from __future__ import annotations

import importlib
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from app_v2.config.test_config import load_test_config
from app_v2.application.pipeline_orchestrator import PipelineOrchestrator
from app_v2.core.fusion_strategy import FusionStrategy, SimpleFusionStrategy
from app_v2.enums import FusionStrategyType
from app_v2.core.result_publisher import ResultPublisher
from app_v2.infrastructure.gpu_preprocessor import GpuPreprocessor
from app_v2.infrastructure.nvdec_decoder import NvdecDecodeConfig, NvdecDecoder
from app_v2.infrastructure.rtsp_frame_source import RTSPFrameSource
from app_v2.tests.integration.pipeline.perf_budget import (
    evaluate_perf_budget,
    render_perf_budget_table,
    write_perf_budget_html_report,
)

CONFIG = load_test_config()


class CapturePublisher(ResultPublisher):
    def __init__(self) -> None:
        self.published: list[tuple[int, list[dict[str, Any]]]] = []

    def publish(self, frame_id: int, payload: list[dict[str, Any]]) -> None:
        self.published.append((frame_id, payload))


class PublishAfterThree(FusionStrategy):
    def __init__(self) -> None:
        super().__init__(FusionStrategyType.STRICT_SYNC)

    def should_publish(self, frame_id: int, collected_ids: list[int]) -> bool:
        del frame_id
        return len(collected_ids) >= 3

    def merge(self, payloads: list[Any]) -> list[Any]:
        return list(payloads)


def _normalize_stream_url(value: Any) -> str | None:
    url = value or ""
    url = str(url).strip()
    return url if url else None


def _resolve_stream_url() -> str | None:
    return (
        _normalize_stream_url(os.environ.get("NVDEC_TEST_STREAM_URL"))
        or _normalize_stream_url(CONFIG.get("nvdec_test_stream_url"))
    )


def _pipeline_like_config() -> dict[str, Any]:
    return {
        "fusion_strategy": "ASYNC_OVERLAY",
        "models": {
            "yolo_global": {"enabled": True},
            "yolo_tiles": {"enabled": True},
        },
        "trt_execution": {
            "enabled": True,
            "strict_shape_check": True,
        },
        "streams": {
            "transfer": 0,
            "yolo": 1,
            "density": 2,
            "yolo_global_preprocess": 3,
            "yolo_tiles_preprocess": 4,
            "density_preprocess": 5,
        },
        "preprocess_branches": {
            "yolo_global_preprocess": True,
            "yolo_tiles_preprocess": True,
            "density_preprocess": True,
        },
        "tensor_pool": {
            "max_per_key": 64,
        },
        "preprocess": {
            "yolo_global": {
                "target_width": 640,
                "target_height": 640,
                "mode": "global",
                "overlap": 0.0,
            },
            "yolo_tiles": {
                "target_width": 640,
                "target_height": 640,
                "mode": "tiles",
                "overlap": 0.2,
            },
        },
    }


def test_pipeline_metrics_snapshot_includes_stage_timings_and_pool_stats() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch must be installed to exercise pipeline metrics integration")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise pipeline metrics integration")

    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to exercise pipeline metrics integration")

    try:
        importlib.import_module("PyNvCodec")
    except ModuleNotFoundError:
        pytest.skip("PyNvCodec must be installed to exercise pipeline metrics integration")

    try:
        decoder = NvdecDecoder(stream_url, NvdecDecodeConfig(ring_capacity=4))
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if (
            "libnvcuvid" in message
            or "cuda codec" in message
            or "pynvdecoder" in message
            or "unsupported ffmpeg pixel format" in message
        ):
            pytest.skip(f"NVDEC decoder unavailable: {exc}")
        raise
    preprocessor = GpuPreprocessor()
    preprocessor.configure(_pipeline_like_config())

    decoder.start()
    try:
        popped = None
        for fid in range(1, 4):
            decoder.decode_next_into_ring(frame_id=fid, timestamp_ns=int(time.time_ns()))
            popped = decoder.ring.pop_ready(timeout_s=5)
            assert popped is not None
            if fid < 3:
                warm_slot, _ = popped
                decoder.ring.release(warm_slot)
        assert popped is not None
        slot, frame = popped

        output = preprocessor.build_output(frame_id=1, frame=frame)
        telemetry = output.telemetry
        assert telemetry is not None
        snapshot = telemetry.snapshot()

        required_keys = [
            "frame_id",
            "nvdec_ms",
            "preprocess_ms",
            # preprocess_nv12_bridge_ms removed: NV12 decode is now fused inside each
            # kernel (preprocess_nv12_fused / preprocess_nv12_crop_fused stage names).
            "preprocess_model_yolo_global_ms",
            "preprocess_model_yolo_tiles_ms",
            "preprocess_model_sum_ms",
            "preprocess_model_max_ms",
            "preprocess_critical_path_ms",
            "preprocess_serial_overhead_ms",
            "tensor_pool_allocations",
            "tensor_pool_in_use",
            "tensor_pool_wait_ms",
        ]
        for key in required_keys:
            assert key in snapshot, f"Missing telemetry key: {key}"

        for key, value in snapshot.items():
            if key.endswith("_ms"):
                assert float(value) >= 0.0, f"Telemetry timing must be >= 0 for {key}"

        assert "preprocess_stream_model_yolo_global" in snapshot
        assert "preprocess_stream_model_yolo_tiles" in snapshot
        assert snapshot["preprocess_model_max_ms"] <= snapshot["preprocess_model_sum_ms"]
        assert snapshot["preprocess_critical_path_ms"] <= snapshot["preprocess_ms"]

        budget_report = evaluate_perf_budget(snapshot, fusion_strategy=_pipeline_like_config().get("fusion_strategy"))
        if budget_report.mode != "off":
            print("perf_budget_checked:", budget_report.checked)
            print("perf_budget_summary:", budget_report.summary)
            print("perf_budget_table:\n" + render_perf_budget_table(budget_report))
            report_file = write_perf_budget_html_report(budget_report, report_name="pipeline_preprocess_metrics")
            print("perf_budget_html_report:", str(report_file))
            if budget_report.violations:
                print("perf_budget_violations:", budget_report.violations)
            if budget_report.should_fail:
                assert (
                    not budget_report.violations
                ), f"Performance budget exceeded: {budget_report.violations}"

        print("pipeline_metrics_snapshot:", {k: snapshot[k] for k in sorted(snapshot.keys()) if k.endswith("_ms") or k.startswith("tensor_pool_")})

        output.release_all()
        decoder.ring.release(slot)
    finally:
        decoder.stop()


def test_pipeline_e2e_real_stream_includes_inference_timings() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch must be installed to run e2e inference integration")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to run e2e inference integration")

    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to exercise e2e inference integration")

    try:
        importlib.import_module("PyNvCodec")
    except ModuleNotFoundError:
        pytest.skip("PyNvCodec must be installed to run e2e inference integration")

    publisher = CapturePublisher()
    try:
        source = RTSPFrameSource(stream_url)
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if (
            "libnvcuvid" in message
            or "cuda codec" in message
            or "pynvdecoder" in message
            or "unsupported ffmpeg pixel format" in message
        ):
            pytest.skip(f"NVDEC frame source unavailable: {exc}")
        raise

    orchestrator = PipelineOrchestrator(
        frame_source=source,
        max_frames=8,
        publisher=publisher,
        fusion_strategy=SimpleFusionStrategy(FusionStrategyType.STRICT_SYNC),
    )
    orchestrator.run()

    assert publisher.published, "Pipeline should publish at least one payload"

    telemetry_snapshot: dict[str, Any] | None = None
    telemetry_frame_id: int = -1
    telemetry_by_frame: dict[int, dict[str, Any]] = {}
    inference_by_frame: dict[int, dict[str, float]] = defaultdict(dict)
    inference_breakdown_by_frame: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    yolo_payloads: list[dict[str, Any]] = []
    for published_frame_id, payload in publisher.published:
        for item in payload:
            if "telemetry" in item and isinstance(item["telemetry"], dict):
                current_frame = int(float(item["telemetry"].get("frame_id", -1)))
                telemetry_by_frame[current_frame] = dict(item["telemetry"])
                if current_frame >= telemetry_frame_id:
                    telemetry_snapshot = item["telemetry"]
                    telemetry_frame_id = current_frame
            if item.get("model") in {"yolo_global", "yolo_tiles"}:
                yolo_payloads.append(item)
                infer_ms = item.get("inference_ms")
                model_name = item.get("model")
                if isinstance(model_name, str) and isinstance(infer_ms, (int, float)):
                    per_frame = inference_by_frame[published_frame_id]
                    per_frame[model_name] = max(per_frame.get(model_name, 0.0), float(infer_ms))
                    breakdown = inference_breakdown_by_frame[published_frame_id][model_name]
                    breakdown["prepare_batch_ms"] = float(item.get("prepare_batch_ms", 0.0))
                    breakdown["enqueue_ms"] = float(item.get("enqueue_ms", 0.0))
                    breakdown["stream_sync_ms"] = float(item.get("stream_sync_ms", 0.0))
                    breakdown["decode_ms"] = float(item.get("decode_ms", 0.0))

    assert telemetry_snapshot is not None, "Expected telemetry in published payload"
    for key in [
        "fusion_wait_ms",
        "overlay_lag_ms",
        "end_to_end_ms",
        "inference_model_sum_ms",
        "inference_model_max_ms",
    ]:
        assert key in telemetry_snapshot, f"Missing e2e telemetry key: {key}"

    assert yolo_payloads, "Expected at least one YOLO payload"
    assert any("inference_ms" in item for item in yolo_payloads), "Expected inference_ms in YOLO payload"
    status_values = [
        item.get("prediction", {}).get("status")
        for item in yolo_payloads
        if isinstance(item.get("prediction"), dict)
    ]
    if status_values:
        assert all(status == "ok" for status in status_values), f"Unexpected inference status values: {status_values}"

    latest_frame_inference = inference_by_frame.get(telemetry_frame_id, {})

    telemetry_snapshot["inference_model_yolo_global_ms"] = float(latest_frame_inference.get("yolo_global", 0.0))
    telemetry_snapshot["inference_model_yolo_tiles_ms"] = float(latest_frame_inference.get("yolo_tiles", 0.0))
    telemetry_snapshot["inference_model_sum_ms"] = float(
        telemetry_snapshot["inference_model_yolo_global_ms"] + telemetry_snapshot["inference_model_yolo_tiles_ms"]
    )
    telemetry_snapshot["inference_model_max_ms"] = float(
        max(telemetry_snapshot["inference_model_yolo_global_ms"], telemetry_snapshot["inference_model_yolo_tiles_ms"])
    )

    warmup_frames = 5
    history_metrics = [
        "nvdec_ms",
        "preprocess_ms",
        # preprocess_nv12_bridge_ms: NV12 decode now fused inside kernel stage timings
        "preprocess_serial_overhead_ms",
        "inference_model_sum_ms",
        "inference_model_max_ms",
        "inference_model_yolo_global_ms",
        "inference_model_yolo_tiles_ms",
        "inference_prepare_batch_yolo_global_ms",
        "inference_prepare_batch_yolo_tiles_ms",
        "inference_enqueue_yolo_global_ms",
        "inference_enqueue_yolo_tiles_ms",
        "inference_stream_sync_yolo_global_ms",
        "inference_stream_sync_yolo_tiles_ms",
        "inference_decode_yolo_global_ms",
        "inference_decode_yolo_tiles_ms",
        "end_to_end_ms",
    ]
    history: dict[str, list[float]] = defaultdict(list)
    ordered_frames = sorted(frame for frame in telemetry_by_frame.keys() if frame >= 0)
    steady_frames = ordered_frames[warmup_frames:]
    for frame_id in steady_frames:
        sample = dict(telemetry_by_frame[frame_id])
        model_values = inference_by_frame.get(frame_id, {})
        sample["inference_model_yolo_global_ms"] = float(model_values.get("yolo_global", 0.0))
        sample["inference_model_yolo_tiles_ms"] = float(model_values.get("yolo_tiles", 0.0))
        sample["inference_model_sum_ms"] = float(sample["inference_model_yolo_global_ms"]) + float(sample["inference_model_yolo_tiles_ms"])
        sample["inference_model_max_ms"] = max(float(sample["inference_model_yolo_global_ms"]), float(sample["inference_model_yolo_tiles_ms"]))
        
        breakdown = inference_breakdown_by_frame.get(frame_id, {})
        if "yolo_global" in breakdown:
            sample["inference_prepare_batch_yolo_global_ms"] = breakdown["yolo_global"].get("prepare_batch_ms", 0.0)
            sample["inference_enqueue_yolo_global_ms"] = breakdown["yolo_global"].get("enqueue_ms", 0.0)
            sample["inference_stream_sync_yolo_global_ms"] = breakdown["yolo_global"].get("stream_sync_ms", 0.0)
            sample["inference_decode_yolo_global_ms"] = breakdown["yolo_global"].get("decode_ms", 0.0)
        if "yolo_tiles" in breakdown:
            sample["inference_prepare_batch_yolo_tiles_ms"] = breakdown["yolo_tiles"].get("prepare_batch_ms", 0.0)
            sample["inference_enqueue_yolo_tiles_ms"] = breakdown["yolo_tiles"].get("enqueue_ms", 0.0)
            sample["inference_stream_sync_yolo_tiles_ms"] = breakdown["yolo_tiles"].get("stream_sync_ms", 0.0)
            sample["inference_decode_yolo_tiles_ms"] = breakdown["yolo_tiles"].get("decode_ms", 0.0)
        
        for key in history_metrics:
            value = sample.get(key)
            if isinstance(value, (int, float)):
                history[key].append(float(value))

    budget_report = evaluate_perf_budget(telemetry_snapshot, fusion_strategy=orchestrator.config.get("fusion_strategy"))
    if budget_report.mode != "off":
        print("e2e_perf_budget_checked:", budget_report.checked)
        print("e2e_perf_budget_summary:", budget_report.summary)
        print("e2e_perf_budget_table:\n" + render_perf_budget_table(budget_report))
        report_file = write_perf_budget_html_report(
            budget_report,
            report_name="pipeline_e2e_metrics",
            history=history,
            warmup_frames=warmup_frames,
        )
        print("e2e_perf_budget_html_report:", str(report_file))


def test_pipeline_e2e_real_stream_trt_opt_in_shape_guard() -> None:
    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to exercise TRT opt-in e2e integration")

    previous_value = os.environ.get("PEOPLE_COUNTER_ENABLE_TRT_EXECUTION")
    os.environ["PEOPLE_COUNTER_ENABLE_TRT_EXECUTION"] = "1"
    publisher = CapturePublisher()
    try:
        source = RTSPFrameSource(stream_url)
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "unsupported ffmpeg pixel format" in message or "pynvdecoder" in message:
            pytest.skip(f"NVDEC frame source unavailable: {exc}")
        raise

    try:
        orchestrator = PipelineOrchestrator(
            frame_source=source,
            max_frames=1,
            publisher=publisher,
            fusion_strategy=SimpleFusionStrategy(FusionStrategyType.STRICT_SYNC),
        )
        orchestrator.run()
    finally:
        if previous_value is None:
            os.environ.pop("PEOPLE_COUNTER_ENABLE_TRT_EXECUTION", None)
        else:
            os.environ["PEOPLE_COUNTER_ENABLE_TRT_EXECUTION"] = previous_value

    assert publisher.published, "Expected at least one publish in TRT opt-in run"
    flattened = [entry for _, payload in publisher.published for entry in payload if isinstance(entry, dict)]
    model_entries = [entry for entry in flattened if entry.get("model") in {"yolo_global", "yolo_tiles"}]
    assert model_entries, "Expected YOLO entries in TRT opt-in run"
    for entry in model_entries:
        prediction = entry.get("prediction")
        if isinstance(prediction, dict):
            status = prediction.get("status")
            assert status in {"ok", "gpu_unavailable"}


def test_e2e_visual_detection_snapshot() -> None:
    """Decode 1 frame from the live stream, run global YOLO, save annotated PNG.

    Output: app_v2/tests/integration/pipeline/artifacts/e2e_visual_snapshot.png
    (visible on the host — the workspace is bind-mounted to /app inside Docker).

    GPU→CPU copy happens only AFTER all GPU inference is finished.
    """
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch must be installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    stream_url = _resolve_stream_url()
    if stream_url is None:
        pytest.skip("Set NVDEC_TEST_STREAM_URL to run visual detection snapshot")

    try:
        importlib.import_module("PyNvCodec")
    except ModuleNotFoundError:
        pytest.skip("PyNvCodec must be installed")

    from app_v2.config import load_pipeline_config
    from app_v2.application.inference_stream_controller import InferenceStreamController
    from app_v2.application.model_builder import ModelBuilder
    from app_v2.infrastructure.stream_pool import SimpleStreamPool

    real_cfg = load_pipeline_config()
    global_model_cfg = real_cfg.get("models", {}).get("yolo_global", {})
    if not global_model_cfg.get("enabled", False):
        pytest.skip("yolo_global not enabled in pipeline.yaml")

    engine_path = global_model_cfg.get("engine", "")
    project_root = Path(__file__).parents[4]  # /app inside Docker
    engine_full = project_root / engine_path if not os.path.isabs(engine_path) else Path(engine_path)
    if not engine_full.exists():
        pytest.skip(f"Engine not found: {engine_full}")

    # Global-only config — tiles off, no separate tiles preprocess branch
    global_only_cfg: dict[str, Any] = {
        **real_cfg,
        "models": {"yolo_global": global_model_cfg},
        "preprocess_branches": {"yolo_global_preprocess": True},
        "preprocess": {
            "yolo_global": real_cfg.get("preprocess", {}).get(
                "yolo_global",
                {"target_width": 640, "target_height": 640, "mode": "global", "overlap": 0.0},
            )
        },
        "yolo_tiles_parallel": {"enabled": False},
    }

    controller = InferenceStreamController(global_only_cfg)
    models = ModelBuilder(global_only_cfg, controller, SimpleStreamPool()).build_models()
    if not models:
        pytest.skip("No models built from real config")

    # Enable seg-mask extraction for visualization (GPU→CPU copy deferred to after all inference)
    for m in models:
        if hasattr(m, "_decoder"):
            m._decoder.seg_mask_enabled = True

    # Decode 2 frames: 1 warm-up + 1 for inference
    try:
        decoder = NvdecDecoder(stream_url, NvdecDecodeConfig(ring_capacity=4))
    except (RuntimeError, ValueError) as exc:
        msg = str(exc).lower()
        if "libnvcuvid" in msg or "pynvdecoder" in msg or "unsupported ffmpeg pixel format" in msg:
            pytest.skip(f"NVDEC unavailable: {exc}")
        raise

    slot: Any = None
    frame: Any = None
    frame_w = 0
    frame_h = 0
    results: list[dict[str, Any]] = []
    frame_rgb = None

    decoder.start()
    try:
        for fid in range(1, 3):
            decoder.decode_next_into_ring(frame_id=fid, timestamp_ns=int(time.time_ns()))
            popped = decoder.ring.pop_ready(timeout_s=5)
            assert popped is not None, f"Timeout waiting for frame fid={fid}"
            if fid < 2:
                decoder.ring.release(popped[0])
            else:
                slot, frame = popped

        assert frame is not None
        frame_w = getattr(frame, "width", 0) or 1920
        frame_h = getattr(frame, "height", 0) or 1080

        preprocessor = GpuPreprocessor()
        preprocessor.configure(global_only_cfg)
        output = preprocessor.build_output(frame_id=1, frame=frame)

        # --- All GPU work first ---
        for model in models:
            processed = output.flatten_inputs(model.name)
            tile_plan = output.plans.get(model.name)
            pred = model.infer(
                1, processed,
                preprocess_events=list(output.cuda_events.values()),
                tile_plan=tile_plan,
            )
            n_det = len(pred.get("detections", []))
            print(f"  [e2e-snapshot] {model.name}: {n_det} detections")
            results.append(pred)

        # --- GPU→CPU copy only after ALL inference is done ---
        try:
            from app_v2.kernels.nv12_cuda_bridge import nv12_to_rgb_hwc_resized_cuda
            rgb_t = nv12_to_rgb_hwc_resized_cuda(frame, frame_h, frame_w, stream_id=99)
            torch.cuda.synchronize()
            frame_rgb = rgb_t.cpu().numpy()
        except Exception as exc_rgb:
            print(f"  [e2e-snapshot] RGB conversion skipped: {exc_rgb}")

        output.release_all()
        decoder.ring.release(slot)
    finally:
        decoder.stop()

    # --- Annotate and save PNG ---
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    total_dets = sum(len(r.get("detections", [])) for r in results)
    print(f"  [e2e-snapshot] total detections across all models: {total_dets}")

    if frame_rgb is not None:
        try:
            import base64
            import json
            from PIL import Image, ImageDraw  # type: ignore[import]
            img = Image.fromarray(frame_rgb).convert("RGBA")

            # --- Seg-mask overlay (semi-transparent green) ---
            for r in results:
                mask_b64 = r.get("seg_mask_raw")
                mw = r.get("seg_mask_w", 0)
                mh = r.get("seg_mask_h", 0)
                if isinstance(mask_b64, str) and mw > 0 and mh > 0:
                    try:
                        import numpy as np_
                        raw = np_.frombuffer(base64.b64decode(mask_b64), dtype=np_.uint8).reshape(mh, mw)
                        # Un-letterbox: the model letterboxes the frame into 640×640
                        # space with black padding.  The mask is in that same padded
                        # space (scaled to 160×160).  Crop out the content region
                        # first so the mask aligns with the actual frame pixels.
                        _lbox_s = min(640.0 / frame_w, 640.0 / frame_h)
                        _cw     = frame_w * _lbox_s          # content px in 640-space
                        _ch     = frame_h * _lbox_s
                        _px     = (640.0 - _cw) / 2.0       # left/right pad
                        _py     = (640.0 - _ch) / 2.0       # top/bottom pad
                        _m2m    = mw / 640.0                 # mask-space/640
                        _cx1    = int(round(_px * _m2m))
                        _cy1    = int(round(_py * _m2m))
                        _cx2    = min(int(round((_px + _cw) * _m2m)), mw)
                        _cy2    = min(int(round((_py + _ch) * _m2m)), mh)
                        mask_content = raw[_cy1:_cy2, _cx1:_cx2]
                        mask_img = Image.fromarray(mask_content, mode="L").resize(
                            (frame_w, frame_h), Image.BILINEAR
                        )
                        overlay = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
                        green = Image.new("RGBA", (frame_w, frame_h), (42, 223, 100, 90))
                        overlay.paste(green, mask=mask_img)
                        img = Image.alpha_composite(img, overlay)
                    except Exception as exc_mask:
                        print(f"  [e2e-snapshot] mask overlay skipped: {exc_mask}")

            # --- Bounding boxes ---
            draw = ImageDraw.Draw(img)
            for r in results:
                for det in r.get("detections", []):
                    bbox = det.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    bx1 = int(bbox[0] * frame_w)
                    by1 = int(bbox[1] * frame_h)
                    bx2 = int(bbox[2] * frame_w)
                    by2 = int(bbox[3] * frame_h)
                    draw.rectangle([bx1, by1, bx2, by2], outline=(42, 223, 165), width=3)
                    draw.text((bx1 + 3, max(0, by1 - 15)), f"{det.get('conf', 0):.0%}", fill=(230, 255, 250))

            png_path = artifacts_dir / "e2e_visual_snapshot.png"
            img.convert("RGB").save(str(png_path))
            print(f"  [e2e-snapshot] annotated PNG saved → {png_path}")

            # --- JSON result file ---
            json_report: dict[str, Any] = {
                "frame_w": frame_w,
                "frame_h": frame_h,
                "total_detections": total_dets,
                "models": {},
            }
            for r in results:
                model_name = r.get("model", "unknown")
                json_report["models"][model_name] = {
                    "detections": [
                        {
                            "bbox": det.get("bbox", []),
                            "conf": det.get("conf", 0),
                            "label": det.get("label", ""),
                        }
                        for det in r.get("detections", [])
                    ],
                    "detection_count": len(r.get("detections", [])),
                    "seg_mask_present": bool(r.get("seg_mask_raw")),
                    "seg_mask_w": r.get("seg_mask_w", 0),
                    "seg_mask_h": r.get("seg_mask_h", 0),
                    "inference_ms": r.get("inference_ms"),
                    "prepare_batch_ms": r.get("prepare_batch_ms"),
                }
            json_path = artifacts_dir / "e2e_visual_snapshot.json"
            json_path.write_text(json.dumps(json_report, indent=2, default=str))
            print(f"  [e2e-snapshot] JSON report saved → {json_path}")
        except ImportError:
            print("  [e2e-snapshot] PIL not available — PNG annotation skipped")
    else:
        print("  [e2e-snapshot] no RGB frame captured — PNG skipped")

    assert results, "Expected at least one inference result"

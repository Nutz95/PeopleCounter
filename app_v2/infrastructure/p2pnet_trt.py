from __future__ import annotations

import time
from typing import Any, Sequence

from app_v2.core.inference_model import InferenceModel

# ImageNet normalization constants (same as Density/LWCC path)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _apply_imagenet_norm(batch_fp16: Any) -> Any:
    """Normalize a [B, 3, H, W] fp16 [0..1] batch to ImageNet fp32 on GPU."""
    try:
        import torch
    except ModuleNotFoundError:
        return batch_fp16

    batch = batch_fp16.float()
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


class P2PNetTRT(InferenceModel):
    """TensorRT-backed P2PNet point-based crowd counting model.

    Output contract for frontend compatibility:
      - hotspots: [{x, y, w}] with x/y normalized to full frame [0,1]
      - hotspot_count / density_count / count: integer person estimate
    """

    def __init__(
        self,
        engine_context: Any,
        stream_id: int,
        inference_params: dict[str, Any] | None = None,
        model_name: str = "p2pnet",
    ) -> None:
        self._context = engine_context
        self._stream_id = stream_id
        self._name = model_name
        self._inference_params = dict(inference_params or {})
        self._confidence_threshold = float(self._inference_params.get("confidence_threshold", 0.5))
        self._max_points = int(self._inference_params.get("max_points", 5000))
        self._display_max_points = int(self._inference_params.get("display_max_points", self._max_points))
        self._export_hotspots_for_ui = bool(self._inference_params.get("export_hotspots_for_ui", True))

    @property
    def name(self) -> str:
        return self._name

    @property
    def stream_id(self) -> int:
        return self._stream_id

    def warm_up(self, batch_size: int) -> None:
        pass

    def infer(
        self,
        frame_id: int,
        inputs: Sequence[Any],
        *,
        preprocess_events: Sequence[Any] | None = None,
        tile_plan: Any | None = None,
    ) -> dict[str, Any]:
        stream_key = f"model:{self._name}"
        start_ns = time.perf_counter_ns()
        self._context.bind_stream(stream_key)
        try:
            batch_fp32 = self._build_batch(inputs)
            raw_outputs = self._context.execute(
                {
                    "frame_id": frame_id,
                    "inputs": [batch_fp32] if batch_fp32 is not None else list(inputs),
                    "model": self._name,
                    "params": dict(self._inference_params),
                    "preprocess_events": list(preprocess_events or []),
                }
            )
            gpu_done_ns = time.perf_counter_ns()
            decode_start_ns = gpu_done_ns
            hotspots, point_count, hotspots_gpu_tensor = self._decode_points(raw_outputs, tile_plan)
            decode_ms = (time.perf_counter_ns() - decode_start_ns) / 1_000_000.0
            infer_ms = (gpu_done_ns - start_ns) / 1_000_000.0

            return {
                "frame_id": frame_id,
                "model": self._name,
                "hotspots": hotspots,
                "hotspot_count": int(point_count),
                "_hotspots_gpu_tensor": hotspots_gpu_tensor,
                # Keep compatibility with existing count extraction and dashboards
                "density_count": float(point_count),
                "count": int(point_count),
                "tile_plan": tile_plan,
                "inference_ms": float(infer_ms),
                "prepare_batch_ms": float(raw_outputs.get("prepare_batch_ms", 0.0)) if isinstance(raw_outputs, dict) else 0.0,
                "enqueue_ms": float(raw_outputs.get("enqueue_ms", 0.0)) if isinstance(raw_outputs, dict) else 0.0,
                "stream_sync_ms": float(raw_outputs.get("stream_sync_ms", 0.0)) if isinstance(raw_outputs, dict) else 0.0,
                "decode_ms": float(decode_ms),
                "inference_params": self._inference_params,
            }
        finally:
            self._context.release_stream(stream_key)

    def close(self) -> None:
        pass

    @staticmethod
    def _build_batch(inputs: Sequence[Any]) -> Any:
        try:
            import torch
        except ModuleNotFoundError:
            return None

        if not inputs:
            return None

        tiles = []
        for t in inputs:
            tensor = getattr(t, "tensor", t)
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                try:
                    tensor = torch.as_tensor(tensor)
                except Exception:
                    continue
            tiles.append(tensor)

        if not tiles:
            return None

        batch = torch.stack(tiles, dim=0)
        return _apply_imagenet_norm(batch)

    def _decode_points(self, raw_outputs: Any, tile_plan: Any | None) -> tuple[list[dict[str, float]], int, Any | None]:
        try:
            import torch
        except ModuleNotFoundError:
            return [], 0, None

        if not isinstance(raw_outputs, dict):
            return [], 0, None

        tensors = raw_outputs.get("output_tensors")
        if not isinstance(tensors, list) or len(tensors) < 2:
            return [], 0, None

        logits = tensors[0]
        points = tensors[1]
        if not isinstance(logits, torch.Tensor) or not isinstance(points, torch.Tensor):
            return [], 0, None

        # Expected shapes: logits [B, N, 2], points [B, N, 2]
        if logits.dim() != 3 or points.dim() != 3:
            return [], 0, None

        if logits.shape[0] < 1 or points.shape[0] < 1:
            return [], 0, None

        scores = torch.softmax(logits[0], dim=-1)[:, 1]  # crowd class prob
        pts = points[0]

        mask = scores >= self._confidence_threshold
        if not torch.any(mask):
            return [], 0

        sel_pts = pts[mask]
        sel_scores = scores[mask]
        selected_count = int(sel_pts.shape[0])

        if self._max_points > 0 and sel_pts.shape[0] > self._max_points:
            topk_scores, topk_idx = torch.topk(sel_scores, self._max_points, largest=True, sorted=False)
            sel_pts = sel_pts[topk_idx]
            sel_scores = topk_scores
            selected_count = self._max_points

        # UI/display cap: keep count fidelity while avoiding giant payloads and
        # expensive JSON/draw calls on ultra-dense scenes.
        disp_pts = sel_pts
        disp_scores = sel_scores
        if self._display_max_points > 0 and disp_pts.shape[0] > self._display_max_points:
            # IMPORTANT: avoid score-only top-k for display, it creates visible
            # "holes" in dense scenes (keeps only the most confident clusters).
            # Use deterministic uniform sampling over decoder order instead.
            total = int(disp_pts.shape[0])
            target = int(self._display_max_points)
            sample_pos = torch.linspace(0, total - 1, steps=target, device=disp_pts.device)
            sample_idx = sample_pos.round().long().clamp(0, total - 1)
            disp_pts = disp_pts[sample_idx]
            disp_scores = disp_scores[sample_idx]

        hotspots, hotspots_gpu_tensor = self._to_hotspots(
            disp_pts,
            disp_scores,
            tile_plan,
            export_hotspots_for_ui=self._export_hotspots_for_ui,
        )
        return hotspots, selected_count, hotspots_gpu_tensor

    @staticmethod
    def _to_hotspots(
        points_xy: Any,
        scores: Any,
        tile_plan: Any | None,
        *,
        export_hotspots_for_ui: bool,
    ) -> tuple[list[dict[str, float]], Any | None]:
        """Map model-space points to normalized full-frame points [0,1]."""
        try:
            import torch
        except ModuleNotFoundError:
            return [], None

        if points_xy.numel() == 0:
            return [], None

        # Default mapping: normalized in model input space
        x = points_xy[:, 0]
        y = points_xy[:, 1]

        if tile_plan is not None and getattr(tile_plan, "tasks", None):
            task = tile_plan.tasks[0]
            fw = float(getattr(tile_plan, "frame_width", 1) or 1)
            fh = float(getattr(tile_plan, "frame_height", 1) or 1)

            sw = float(getattr(task, "source_width", 1) or 1)
            sh = float(getattr(task, "source_height", 1) or 1)
            sx = float(getattr(task, "source_x", 0) or 0)
            sy = float(getattr(task, "source_y", 0) or 0)
            tw = float(getattr(task, "target_width", sw) or sw)
            th = float(getattr(task, "target_height", sh) or sh)

            # Reverse letterbox from model-space pixels to source crop pixels.
            scale = min(tw / max(1.0, sw), th / max(1.0, sh))
            prep_w = sw * scale
            prep_h = sh * scale
            pad_x = (tw - prep_w) * 0.5
            pad_y = (th - prep_h) * 0.5

            ox = ((x - pad_x) / max(1.0, prep_w)).clamp(0.0, 1.0)
            oy = ((y - pad_y) / max(1.0, prep_h)).clamp(0.0, 1.0)

            gx = ((sx + ox * sw) / max(1.0, fw)).clamp(0.0, 1.0)
            gy = ((sy + oy * sh) / max(1.0, fh)).clamp(0.0, 1.0)
        else:
            # Fallback when plan is unavailable
            max_x = float(torch.max(x).item()) if x.numel() > 0 else 1.0
            max_y = float(torch.max(y).item()) if y.numel() > 0 else 1.0
            gx = (x / max(1.0, max_x)).clamp(0.0, 1.0)
            gy = (y / max(1.0, max_y)).clamp(0.0, 1.0)

        hotspot_tensor = torch.stack(
            (gx, gy, scores.clamp(0.0, 1.0)),
            dim=1,
        )
        if not export_hotspots_for_ui:
            return [], hotspot_tensor

        packed = hotspot_tensor.detach().cpu().tolist()
        return [{"x": float(px), "y": float(py), "w": float(pw)} for px, py, pw in packed], hotspot_tensor

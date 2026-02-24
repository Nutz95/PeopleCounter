from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from app_v2.core.inference_model import InferenceModel

# ImageNet normalization constants (DM-Count / LWCC preprocessing convention)
# Applied on-GPU after the [0,1] fp16 tiling preprocessing output:
#   output = (input - mean) / std
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _apply_imagenet_norm(batch_fp16: Any) -> Any:
    """Normalise a [B, 3, H, W] fp16 [0-1] batch to ImageNet stats, return fp32.

    DM-Count (and all LWCC models) were trained with torchvision ImageNet
    preprocessing.  The GPU preprocessor outputs fp16 in [0, 1]; this step
    shifts and scales in fp32 on the same GPU device — no CPU round-trip.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return batch_fp16

    batch = batch_fp16.float()  # fp16 → fp32, stays on GPU
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=batch.device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


class DensityTRT(InferenceModel):
    """TensorRT DM-Count density model.

    Default tiling: 2×2 = 4 tiles of 1920×1088 over 4K (no rescaling, native resolution).
    Also supports 6×3 = 18 tiles of 640×720 (legacy, same latency, rescaled).

    Preprocessing contract (from ``GpuPreprocessor``):
        • N GpuTensor tiles, each [3, H, W] fp16 in [0, 1].
        • Stacked → [N, 3, H, W], then ImageNet-normalised → fp32.
        • TRT engine input: [B, 3, H, W] fp32.

    TRT output: [B, 1, H_out, W_out] fp32 density map (stride 8).
    For 1920×1088 input: H_out=136, W_out=240 (1088/8, 1920/8).
    For  640×720 input: H_out= 90, W_out= 80  ( 720/8,  640/8).

    Benchmark (RTX 5060 Ti, FP16):
        • batch=4  @ 1920×1088 → ~92 ms  (full 4K, no rescaling) ← default
        • batch=18 @  640×720  → ~92 ms  (full 4K, rescaled)
        • For < 30 ms at full 4K → FP8 quantization required.

    Execution strategies (``pipeline.yaml → models.density.strategy``):
        ``sequential`` (default)
            One ``execute()`` call with the full batch of N tiles.

        ``parallel_rows``
            Split tiles into ``n_row_streams`` sub-batches on independent
            CUDA streams/TRT contexts.  Only useful if GPU has spare SMs;
            benchmark shows no benefit on RTX 5060 Ti (GPU already saturated).
    """

    _STRATEGY_SEQUENTIAL    = "sequential"
    _STRATEGY_PARALLEL_ROWS = "parallel_rows"

    def __init__(
        self,
        engine_context: Any,
        stream_id: int,
        *,
        strategy: str = "sequential",
        n_row_streams: int = 3,
        tiles_per_row: int = 6,
    ) -> None:
        self._context = engine_context
        self._stream_id = stream_id
        self._name = "density"
        self._strategy = strategy
        self._n_row_streams = n_row_streams
        self._tiles_per_row = tiles_per_row
        # Extra execution contexts for parallel_rows mode — lazy-created
        self._extra_contexts: list[Any] = []

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
        """Run DM-Count on a batch of tiles and return per-tile density maps."""
        if self._strategy == self._STRATEGY_PARALLEL_ROWS:
            return self._infer_parallel_rows(frame_id, inputs, tile_plan=tile_plan)
        return self._infer_sequential(frame_id, inputs,
                                      preprocess_events=preprocess_events,
                                      tile_plan=tile_plan)

    # ── sequential (default) ──────────────────────────────────────────────────

    def _infer_sequential(
        self,
        frame_id: int,
        inputs: Sequence[Any],
        *,
        preprocess_events: Sequence[Any] | None = None,
        tile_plan: Any | None = None,
    ) -> dict[str, Any]:
        """Single TRT execute() call with all tiles batched together."""
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
                    "params": {},
                    "preprocess_events": list(preprocess_events or []),
                }
            )
            infer_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

            tensors = raw_outputs.get("output_tensors", []) if isinstance(raw_outputs, dict) else []
            density_tiles, total_count = self._decode_density(tensors, len(inputs))

            return {
                "frame_id": frame_id,
                "model": self._name,
                "density_tiles": density_tiles,
                "density_count": float(total_count),
                "tile_plan": tile_plan,
                "inference_ms": float(infer_ms),
                "strategy": self._strategy,
                "prepare_batch_ms": float(raw_outputs.get("prepare_batch_ms", 0.0)) if isinstance(raw_outputs, dict) else 0.0,
                "enqueue_ms": float(raw_outputs.get("enqueue_ms", 0.0)) if isinstance(raw_outputs, dict) else 0.0,
                "stream_sync_ms": float(raw_outputs.get("stream_sync_ms", 0.0)) if isinstance(raw_outputs, dict) else 0.0,
            }
        finally:
            self._context.release_stream(stream_key)

    # ── parallel rows ─────────────────────────────────────────────────────────

    def _infer_parallel_rows(
        self,
        frame_id: int,
        inputs: Sequence[Any],
        *,
        tile_plan: Any | None = None,
    ) -> dict[str, Any]:
        """Split tiles into row sub-batches and run on concurrent CUDA streams.

        Each row sub-batch is fed to a separate TRT execution context on its own
        CUDA stream, so the GPU can schedule them concurrently.  Results are
        gathered and concatenated before returning.
        """
        try:
            import torch
        except ModuleNotFoundError:
            return self._infer_sequential(frame_id, inputs, tile_plan=tile_plan)

        n_tiles   = len(inputs)
        tpr       = self._tiles_per_row   # tiles per row (e.g. 6)
        n_streams = self._n_row_streams   # number of rows (e.g. 3)

        # Split inputs into row sub-lists
        rows: list[list[Any]] = []
        for i in range(0, n_tiles, tpr):
            rows.append(list(inputs[i : i + tpr]))
        if len(rows) > n_streams:
            # More rows than streams: merge excess into last slot
            extra = rows[n_streams:]
            rows = rows[:n_streams]
            for ex in extra:
                rows[-1].extend(ex)

        # Build normalised fp32 batches (CPU-side stacking, still fast)
        batches = [self._build_batch(row) for row in rows]

        # --- Ensure we have enough extra contexts (lazy creation) ---
        # This requires access to the raw TRT engine.  We get it via the
        # context wrapper's `engine` attribute (implementation-specific).
        # Falls back to sequential if engine not accessible.
        raw_engine = getattr(getattr(self._context, "_engine", None), "_engine", None) \
                     or getattr(self._context, "engine", None)
        if raw_engine is None or not hasattr(raw_engine, "create_execution_context"):
            # Can't create extra contexts — fall back
            return self._infer_sequential(frame_id, inputs, tile_plan=tile_plan)

        while len(self._extra_contexts) < len(rows):
            self._extra_contexts.append(raw_engine.create_execution_context())

        start_ns = time.perf_counter_ns()

        # --- Launch rows concurrently via ThreadPoolExecutor ---
        row_results: list[Any] = [None] * len(rows)
        cuda_streams = [torch.cuda.Stream() for _ in rows]

        def _run_row(idx: int) -> Any:
            batch = batches[idx]
            if batch is None:
                return None
            ctx   = self._extra_contexts[idx]
            s     = cuda_streams[idx]
            inp_name = raw_engine.get_tensor_name(0)
            out_name = raw_engine.get_tensor_name(1)
            ctx.set_input_shape(inp_name, tuple(batch.shape))
            out_shape = tuple(ctx.get_tensor_shape(out_name))
            out = torch.zeros(*out_shape, dtype=torch.float32, device="cuda")
            ctx.set_tensor_address(inp_name, batch.data_ptr())
            ctx.set_tensor_address(out_name, out.data_ptr())
            with torch.cuda.stream(s):
                ctx.execute_async_v3(s.cuda_stream)
            return out

        with ThreadPoolExecutor(max_workers=len(rows)) as ex:
            futures = {ex.submit(_run_row, i): i for i in range(len(rows))}
            for fut in as_completed(futures):
                idx = futures[fut]
                row_results[idx] = fut.result()

        torch.cuda.synchronize()
        infer_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        # Concatenate row outputs → [N_tiles, 1, H_out, W_out]
        valid = [r for r in row_results if r is not None and isinstance(r, torch.Tensor)]
        if valid:
            full_out = torch.cat(valid, dim=0)
            density_tiles, total_count = self._decode_density([full_out], n_tiles)
        else:
            density_tiles, total_count = [], 0.0

        return {
            "frame_id": frame_id,
            "model": self._name,
            "density_tiles": density_tiles,
            "density_count": float(total_count),
            "tile_plan": tile_plan,
            "inference_ms": float(infer_ms),
            "strategy": self._strategy,
            "prepare_batch_ms": 0.0,
            "enqueue_ms": 0.0,
            "stream_sync_ms": 0.0,
        }

    def close(self) -> None:
        self._extra_contexts.clear()

    @staticmethod
    def _build_batch(inputs: Sequence[Any]) -> Any:
        """Stack GpuTensors into a single [B, 3, H, W] fp32 batch."""
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

        batch = torch.stack(tiles, dim=0)  # [B, 3, H, W] fp16
        return _apply_imagenet_norm(batch)   # → fp32

    @staticmethod
    def _decode_density(tensors: list[Any], n_tiles: int) -> tuple[list[Any], float]:
        """Split [B, 1, H_out, W_out] output back into per-tile maps + total count."""
        try:
            import torch
        except ModuleNotFoundError:
            return [], 0.0

        if not tensors:
            return [], 0.0

        out = tensors[0] if isinstance(tensors, (list, tuple)) else tensors
        if not isinstance(out, torch.Tensor):
            return [], 0.0

        # TRT may return [B, 1, H, W]; ensure batch dim matches
        if out.dim() == 3:
            out = out.unsqueeze(0)

        # sum() over spatial dims gives person count per tile
        total_count = float(out.sum().item())
        density_tiles = [out[i] for i in range(min(n_tiles, out.shape[0]))]
        return density_tiles, total_count

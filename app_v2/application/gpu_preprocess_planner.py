from __future__ import annotations

from math import ceil

from app_v2.core.preprocessor_types import InputSpec, PreprocessMode, PreprocessPlan, PreprocessTask


class GpuPreprocessPlanner:
    """Generates preprocess plans (global/tiled) for configured model specs."""

    def build_plan(self, frame_width: int, frame_height: int, spec: InputSpec) -> PreprocessPlan:
        if spec.mode is PreprocessMode.GLOBAL:
            task = PreprocessTask(
                model_name=spec.model_name,
                task_index=0,
                source_x=0,
                source_y=0,
                source_width=frame_width,
                source_height=frame_height,
                target_width=spec.target_width,
                target_height=spec.target_height,
                metadata={"kind": "letterbox"},
            )
            return PreprocessPlan(
                model_name=spec.model_name,
                frame_width=frame_width,
                frame_height=frame_height,
                tasks=(task,),
                metadata={"mode": PreprocessMode.GLOBAL.value, "task_count": 1},
            )

        if spec.mode is PreprocessMode.TILES:
            return self._build_tiling_plan(frame_width, frame_height, spec)

        raise ValueError(f"Unsupported preprocess mode: {spec.mode}")

    def _build_tiling_plan(self, frame_width: int, frame_height: int, spec: InputSpec) -> PreprocessPlan:
        # When prescale_width/height are set, the tile grid is computed on a
        # virtual smaller frame (e.g. 1920×1080) and each tile's crop coordinates
        # are then scaled back to the actual frame.  This guarantees SQUARE crops
        # (no aspect-ratio distortion for the model) regardless of source
        # resolution, while fixing the tile count to a predictable value.
        if spec.prescale_width > 0 and spec.prescale_height > 0:
            vw = spec.prescale_width
            vh = spec.prescale_height
            sx: float = frame_width  / vw
            sy: float = frame_height / vh
        else:
            vw = frame_width
            vh = frame_height
            sx = sy = 1.0

        # Effective tile dimensions in the *virtual* frame.
        # When source_tile_width/height > 0, each tile is a larger virtual crop
        # that gets downscaled to target_width/height for the model input.
        # Example: source 1920×1080 virtual crop → model input 640×720.
        eff_w = spec.source_tile_width  if spec.source_tile_width  > 0 else spec.target_width
        eff_h = spec.source_tile_height if spec.source_tile_height > 0 else spec.target_height

        step_x = max(1, int(round(eff_w * (1.0 - spec.overlap))))
        step_y = max(1, int(round(eff_h * (1.0 - spec.overlap))))

        cols = max(1, ceil(max(1, vw - eff_w) / step_x) + 1)
        rows = max(1, ceil(max(1, vh - eff_h) / step_y) + 1)

        tasks: list[PreprocessTask] = []
        task_index = 0
        for row in range(rows):
            vy = min(row * step_y, max(0, vh - eff_h))
            for col in range(cols):
                vx = min(col * step_x, max(0, vw - eff_w))
                # Scale virtual tile coordinates back to actual frame space.
                ax = int(round(vx * sx))
                ay = int(round(vy * sy))
                aw = int(round(eff_w * sx))
                ah = int(round(eff_h * sy))
                # Clamp to actual frame boundaries.
                ax = max(0, min(ax, frame_width  - 1))
                ay = max(0, min(ay, frame_height - 1))
                aw = min(aw, frame_width  - ax)
                ah = min(ah, frame_height - ay)
                tasks.append(
                    PreprocessTask(
                        model_name=spec.model_name,
                        task_index=task_index,
                        source_x=ax,
                        source_y=ay,
                        source_width=max(1, aw),
                        source_height=max(1, ah),
                        target_width=spec.target_width,
                        target_height=spec.target_height,
                        metadata={"row": row, "col": col},
                    )
                )
                task_index += 1

        return PreprocessPlan(
            model_name=spec.model_name,
            frame_width=frame_width,
            frame_height=frame_height,
            tasks=tuple(tasks),
            metadata={
                "mode": PreprocessMode.TILES.value,
                "task_count": len(tasks),
                "rows": rows,
                "cols": cols,
                "overlap": spec.overlap,
            },
        )

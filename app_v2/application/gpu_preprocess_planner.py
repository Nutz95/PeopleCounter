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
        # Effective tile dimensions in the source frame.
        # When source_tile_width/height > 0, each tile is a larger crop that
        # gets downscaled to target_width/height for the model input.
        # Example: 1920×1080 crop → 640×720 model input (2×2 tiles over 4K).
        eff_w = spec.source_tile_width  if spec.source_tile_width  > 0 else spec.target_width
        eff_h = spec.source_tile_height if spec.source_tile_height > 0 else spec.target_height

        step_x = max(1, int(round(eff_w * (1.0 - spec.overlap))))
        step_y = max(1, int(round(eff_h * (1.0 - spec.overlap))))

        cols = max(1, ceil(max(1, frame_width  - eff_w) / step_x) + 1)
        rows = max(1, ceil(max(1, frame_height - eff_h) / step_y) + 1)

        tasks: list[PreprocessTask] = []
        task_index = 0
        for row in range(rows):
            y = min(row * step_y, max(0, frame_height - eff_h))
            for col in range(cols):
                x = min(col * step_x, max(0, frame_width - eff_w))
                tasks.append(
                    PreprocessTask(
                        model_name=spec.model_name,
                        task_index=task_index,
                        source_x=x,
                        source_y=y,
                        source_width=min(eff_w, frame_width),
                        source_height=min(eff_h, frame_height),
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

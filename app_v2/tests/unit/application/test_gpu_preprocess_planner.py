from __future__ import annotations

from app_v2.application.gpu_preprocess_planner import GpuPreprocessPlanner
from app_v2.core.preprocessor_types import InputSpec, PreprocessMode


def test_gpu_preprocess_planner_global_plan() -> None:
    planner = GpuPreprocessPlanner()
    spec = InputSpec(
        model_name="yolo_global",
        target_width=640,
        target_height=640,
        mode=PreprocessMode.GLOBAL,
        overlap=0.0,
    )

    plan = planner.build_plan(frame_width=1920, frame_height=1080, spec=spec)

    assert plan.model_name == "yolo_global"
    assert plan.metadata["mode"] == "global"
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.source_x == 0
    assert task.source_y == 0
    assert task.source_width == 1920
    assert task.source_height == 1080


def test_gpu_preprocess_planner_tiling_plan_has_multiple_tasks() -> None:
    planner = GpuPreprocessPlanner()
    spec = InputSpec(
        model_name="yolo_tiles",
        target_width=640,
        target_height=640,
        mode=PreprocessMode.TILES,
        overlap=0.2,
    )

    plan = planner.build_plan(frame_width=1920, frame_height=1080, spec=spec)

    assert plan.model_name == "yolo_tiles"
    assert plan.metadata["mode"] == "tiles"
    assert plan.metadata["task_count"] > 1
    assert len(plan.tasks) == plan.metadata["task_count"]
    assert plan.metadata["rows"] >= 1
    assert plan.metadata["cols"] >= 1
    assert plan.tasks[0].metadata["row"] == 0
    assert plan.tasks[0].metadata["col"] == 0

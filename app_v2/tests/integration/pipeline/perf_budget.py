from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


DEFAULT_STAGE_BUDGET_MS: dict[str, float] = {
    "nvdec_ms": 35.0,
    "preprocess_nv12_bridge_ms": 15.0,
    "preprocess_ms": 30.0,
    "preprocess_model_yolo_global_ms": 5.0,
    "preprocess_model_yolo_tiles_ms": 5.0,
    "tensor_pool_wait_ms": 2.0,
}


@dataclass(frozen=True)
class PerfBudgetReport:
    mode: str
    checked: dict[str, tuple[float, float]]
    violations: dict[str, tuple[float, float]]

    @property
    def should_fail(self) -> bool:
        return self.mode == "fail" and bool(self.violations)


def perf_budget_mode() -> str:
    value = os.environ.get("PEOPLE_COUNTER_PERF_BUDGET_MODE", "warn").strip().lower()
    if value in {"off", "warn", "fail"}:
        return value
    return "warn"


def evaluate_perf_budget(
    snapshot: Mapping[str, float | int],
    stage_budget_ms: Mapping[str, float] | None = None,
) -> PerfBudgetReport:
    mode = perf_budget_mode()
    budget = dict(stage_budget_ms or DEFAULT_STAGE_BUDGET_MS)

    checked: dict[str, tuple[float, float]] = {}
    violations: dict[str, tuple[float, float]] = {}

    for metric_key, limit_ms in budget.items():
        if metric_key not in snapshot:
            continue
        value = float(snapshot[metric_key])
        checked[metric_key] = (value, float(limit_ms))
        if value > float(limit_ms):
            violations[metric_key] = (value, float(limit_ms))

    return PerfBudgetReport(mode=mode, checked=checked, violations=violations)
